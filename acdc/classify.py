"""Step 4 - 5-class diagnosis (NOR / MINF / DCM / HCM / RV) from features with classical classifiers.

Feature sources
  gt            features from the ground-truth masks (oracle: what perfect segmentation would give)
  <pred cfg>    features from a predicted segmentation (e.g. unet2d, classical_lgbm). For the 100 training patients
                these are OUT-OF-FOLD predictions, so they carry realistic segmentation errors.
  image         no segmentation: down-sampled central crops of the ED / ES / |ED-ES| mid-ventricular slices -> PCA.
                The "classical image features" rung of the ladder.

Scenarios  "<train source>-><test source>":  gt->gt (oracle), gt->unet2d (train on clean features, deploy on predicted),
           unet2d->unet2d (train and deploy on predicted features: fully automatic pipeline), image->image, ...

Classifiers: multinomial logistic regression (standardised features, C tuned) and random forest (tuned).
Evaluation:
  CV   = nested cross-validation on the 100 training patients: outer loop = the frozen 5 folds, inner loop = 5-fold
         grid search on the 80 remaining patients. Gives out-of-fold probabilities for all 100.
  test = grid search (5-fold) on all 100 training patients, refit, predict the 50 test patients once.
Metrics: accuracy (95% bootstrap CI), macro-F1, negative log-likelihood, Brier score, top-label ECE (10 bins).

    python -m acdc.classify --sources unet2d classical_lgbm
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from acdc.common import GROUPS, RESULTS, WORK
from acdc.features import feature_table

OUT = RESULTS / "cls"


def make_model(kind, n_features):
    if kind == "logreg":
        steps = [StandardScaler()]
        if n_features > 50:  # image features: reduce dimension first
            steps.append(PCA(n_components=20, random_state=0))
        return GridSearchCV(make_pipeline(*steps, LogisticRegression(max_iter=10000)),
                            {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                            cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="neg_log_loss", n_jobs=4)
    if kind == "rf":
        return GridSearchCV(RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=4),
                            {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3, 5]},
                            cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="neg_log_loss", n_jobs=1)
    raise ValueError(kind)


def image_features():
    """Classical image features (no segmentation): 3 central slices of ED, ES and |ED-ES|, 128x128 px (160 mm) centre
    crop, block-averaged to 16x16 -> 3 x 3 x 256 = 2,304 values per patient (PCA inside the classifier pipeline)."""
    cache = WORK / "features" / "image.csv"
    if cache.exists():
        return pd.read_csv(cache)
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    rows = []
    for r in meta.itertuples():
        d = np.load(WORK / "cache" / f"{r.pid}.npz")
        ed, es = d["img_ED"].astype(np.float32), d["img_ES"].astype(np.float32)
        Z, H, W = ed.shape
        zc, yc, xc = Z // 2, H // 2, W // 2
        vec = []
        for im in (ed, es, np.abs(ed - es)):
            for z in (max(zc - 1, 0), zc, min(zc + 1, Z - 1)):
                crop = im[z, yc - 64:yc + 64, xc - 64:xc + 64]
                vec.append(crop.reshape(16, 8, 16, 8).mean((1, 3)).ravel())
        v = np.concatenate(vec)
        rows.append({"pid": r.pid, "split": r.split, "fold": r.fold, "group": r.group, **{f"f{i}": x for i, x in enumerate(v)}})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def table(source):
    return image_features() if source == "image" else feature_table(source)


def ece_toplabel(prob, y, n_bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(prob, y, rng_seed=0, n_boot=2000):
    prob = np.clip(prob, 1e-7, 1)
    prob = prob / prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    acc = accuracy_score(y, pred)
    rng = np.random.default_rng(rng_seed)
    boot = [accuracy_score(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    onehot = np.eye(len(GROUPS))[y]
    return {"n": int(len(y)), "acc": float(acc), "acc_lo": float(np.percentile(boot, 2.5)), "acc_hi": float(np.percentile(boot, 97.5)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "nll": float(log_loss(y, prob, labels=range(len(GROUPS)))),
            "brier": float(((prob - onehot) ** 2).sum(1).mean()), "ece": ece_toplabel(prob, y)}


def run_scenario(train_src, test_src, kind):
    tr_tab, te_tab = table(train_src), table(test_src)
    feats = [c for c in tr_tab.columns if c not in ("pid", "split", "fold", "group")]
    enc = {g: i for i, g in enumerate(GROUPS)}
    tr_all = tr_tab[tr_tab.split == "train"].reset_index(drop=True)
    te_cv = te_tab[te_tab.split == "train"].set_index("pid").loc[tr_all.pid].reset_index()  # same patients, test-source features
    # nested CV on the training patients
    oof = np.zeros((len(tr_all), len(GROUPS)))
    for f in range(5):
        fit, val = tr_all.fold != f, te_cv.fold == f
        m = make_model(kind, len(feats)).fit(tr_all.loc[fit, feats].values, tr_all.loc[fit, "group"].map(enc).values)
        oof[val.values] = m.predict_proba(te_cv.loc[val, feats].values)
    # final model on all 100 -> 50 test patients
    m = make_model(kind, len(feats)).fit(tr_all[feats].values, tr_all.group.map(enc).values)
    te = te_tab[te_tab.split == "test"].reset_index(drop=True)
    prob_te = m.predict_proba(te[feats].values)
    res, probs = [], []
    for split, df, prob in (("cv", te_cv, oof), ("test", te, prob_te)):
        y = df.group.map(enc).values
        res.append({"scenario": f"{train_src}->{test_src}", "train_source": train_src, "test_source": test_src,
                    "classifier": kind, "eval": split, **metrics(prob, y), "best_params": json.dumps(m.best_params_),
                    "confusion": json.dumps(confusion_matrix(y, prob.argmax(1), labels=range(5)).tolist())})
        probs.append(pd.DataFrame({"pid": df.pid, "eval": split, "group": df.group,
                                   **{f"p_{g}": prob[:, i] for i, g in enumerate(GROUPS)}}))
    name = f"{train_src}__{test_src}__{kind}"
    pd.concat(probs).to_csv(OUT / "probs" / f"{name}.csv", index=False)
    if kind == "rf" and train_src == test_src and train_src != "image":
        imp = pd.Series(m.best_estimator_.feature_importances_, index=feats).sort_values(ascending=False)
        imp.to_csv(OUT / f"rf_importance_{train_src}.csv", header=["importance"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["unet2d"], help="predicted-segmentation sources to evaluate")
    args = ap.parse_args()
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    scen = [("gt", "gt"), ("image", "image")]
    for s in args.sources:
        scen += [("gt", s), (s, s)]
    path = OUT / "summary.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = []
    for tr, te in scen:
        for kind in ("logreg", "rf"):
            r = run_scenario(tr, te, kind)
            rows += r
            for x in r:
                print(f"{x['scenario']:32s} {kind:7s} {x['eval']:4s} acc {x['acc']:.3f} [{x['acc_lo']:.2f},{x['acc_hi']:.2f}] "
                      f"F1 {x['macro_f1']:.3f} NLL {x['nll']:.3f} ECE {x['ece']:.3f}", flush=True)
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.set_index(["scenario", "classifier", "eval"]).index.isin(new.set_index(["scenario", "classifier", "eval"]).index)]
    pd.concat([old, new]).to_csv(path, index=False)


if __name__ == "__main__":
    main()
