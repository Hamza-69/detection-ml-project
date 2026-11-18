"""Step 8 - diagnosis evaluation: compare all pipelines (results/cls/).

    python -m acdc.evaluate_cls --seg_cfgs unet2d unet25d classical_lgbm

Pipelines compared (test = 50 held-out patients, CV = out-of-fold on the 100 training patients):
  image features -> LR             classical, no segmentation
  CNN end-to-end (ResNet-18)       per seed (5 seeds, test = 5-fold ensemble), mean +- sd over seeds, and the
                                   ensemble of all 25 models; + temperature scaling fitted on the out-of-fold predictions
  GT masks -> features -> LR / RF  oracle
  <seg> masks -> features -> LR/RF automatic two-stage pipeline (seg = U-Net or classical segmentation)
Outputs: pipelines.csv (main table), cnn_seeds.csv, mcnemar.csv, coverage.csv, fig_coverage.png, fig_confusion.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import confusion_matrix

from acdc.classify import metrics
from acdc.common import GROUPS, RESULTS

OUT = RESULTS / "cls"
ENC = {g: i for i, g in enumerate(GROUPS)}
PCOLS = [f"p_{g}" for g in GROUPS]


def fit_temperature(prob, y):
    """Single temperature T minimising NLL of softmax(log p / T)."""
    logp = np.log(np.clip(prob, 1e-7, 1))

    def nll(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        q = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
    return optimize.minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def apply_temperature(prob, t):
    z = np.log(np.clip(prob, 1e-7, 1)) / t
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def cnn_results(tag="resnet18"):
    """Returns {name: (cv_df, test_df)} prob tables for each seed and for the all-seed ensemble, plus a per-seed table."""
    runs = [pd.read_csv(p) for p in sorted((OUT / "cnn_runs").glob(f"{tag}_s*_f*.csv"))]
    if not runs:
        return {}, pd.DataFrame()
    df = pd.concat(runs)
    out, rows = {}, []
    for seed, d in df.groupby("seed"):
        if d.fold.nunique() < 5:
            continue
        cv = d[d["eval"] == "cv"].sort_values("pid")
        te = d[d["eval"] == "test"].groupby(["pid", "group"])[PCOLS].mean().reset_index()  # 5-fold ensemble
        out[f"cnn_s{seed}"] = (cv, te)
        y_cv, y_te = cv.group.map(ENC).values, te.group.map(ENC).values
        t = fit_temperature(cv[PCOLS].values, y_cv)
        r = {"seed": seed, "T": t}
        for ev, dd, y in (("cv", cv, y_cv), ("test", te, y_te)):
            r.update({f"{ev}_{k}": v for k, v in metrics(dd[PCOLS].values, y).items() if k in ("acc", "macro_f1", "nll", "brier", "ece")})
        m_ts = metrics(apply_temperature(te[PCOLS].values, t), y_te)
        r.update({f"test_ts_{k}": m_ts[k] for k in ("acc", "nll", "brier", "ece")})
        rows.append(r)
    seeds = [s for s in df.seed.unique() if f"cnn_s{s}" in out]
    if seeds:
        dd = df[df.seed.isin(seeds)]
        cv = dd[dd["eval"] == "cv"].groupby(["pid", "group"])[PCOLS].mean().reset_index()
        te = dd[dd["eval"] == "test"].groupby(["pid", "group"])[PCOLS].mean().reset_index()
        out["cnn_ensemble"] = (cv, te)
    return out, pd.DataFrame(rows)


def feature_results(seg_cfgs):
    out = {}
    scen = [("image", "image")] + [("gt", "gt")] + [(a, b) for s in seg_cfgs for a, b in (("gt", s), (s, s))]
    for tr, te in scen:
        for kind in ("logreg", "rf"):
            p = OUT / "probs" / f"{tr}__{te}__{kind}.csv"
            if p.exists():
                d = pd.read_csv(p)
                out[f"{tr}->{te} {kind}"] = (d[d["eval"] == "cv"], d[d["eval"] == "test"])
    return out


def mcnemar(a, b, y):
    """Exact McNemar test on paired correct/incorrect predictions."""
    ca, cb = a.argmax(1) == y, b.argmax(1) == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p


def coverage_curve(prob, y, fracs=np.linspace(0.5, 1.0, 11)):
    """Accuracy on the most confident fraction of patients (selective prediction)."""
    conf = prob.max(1)
    order = np.argsort(-conf)
    correct = (prob.argmax(1) == y)[order]
    return [(f, correct[: max(1, int(round(f * len(y))))].mean()) for f in fracs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_cfgs", nargs="+", default=["unet2d"])
    ap.add_argument("--main", default="unet2d->unet2d logreg", help="pipeline used as reference in the McNemar tests")
    args = ap.parse_args()
    cnn, seeds = cnn_results()
    seeds.to_csv(OUT / "cnn_seeds.csv", index=False)
    pipes = {**feature_results(args.seg_cfgs), **cnn}
    if "cnn_ensemble" in cnn and not seeds.empty:
        cv, te = cnn["cnn_ensemble"]
        t = fit_temperature(cv[PCOLS].values, cv.group.map(ENC).values)
        pipes["cnn_ensemble+TS"] = (cv.assign(**dict(zip(PCOLS, apply_temperature(cv[PCOLS].values, t).T))),
                                    te.assign(**dict(zip(PCOLS, apply_temperature(te[PCOLS].values, t).T))))
    rows, cov = [], []
    for name, (cv, te) in pipes.items():
        for ev, d in (("cv", cv), ("test", te)):
            d = d.sort_values("pid")
            y = d.group.map(ENC).values
            m = metrics(d[PCOLS].values, y)
            rows.append({"pipeline": name, "eval": ev, **m,
                         "confusion": json.dumps(confusion_matrix(y, d[PCOLS].values.argmax(1), labels=range(5)).tolist())})
            for f, a in coverage_curve(d[PCOLS].values, y):
                cov.append({"pipeline": name, "eval": ev, "coverage": f, "acc": a})
    res = pd.DataFrame(rows)
    if not seeds.empty:  # add mean +- sd over CNN seeds as one row per eval set
        for ev in ("cv", "test"):
            r = {"pipeline": "cnn (mean over seeds)", "eval": ev}
            for k in ("acc", "macro_f1", "nll", "brier", "ece"):
                r[k] = seeds[f"{ev}_{k}"].mean()
                r[f"{k}_sd"] = seeds[f"{ev}_{k}"].std()
            rows.append(r)
        res = pd.DataFrame(rows)
    res.to_csv(OUT / "pipelines.csv", index=False)
    pd.DataFrame(cov).to_csv(OUT / "coverage.csv", index=False)

    # paired tests against the main automatic pipeline
    mc = []
    if args.main in pipes:
        for name, (cv, te) in pipes.items():
            if name == args.main:
                continue
            for ev, a_df, b_df in (("cv", pipes[args.main][0], cv), ("test", pipes[args.main][1], te)):
                a_df, b_df = a_df.sort_values("pid"), b_df.sort_values("pid")
                assert (a_df.pid.values == b_df.pid.values).all()
                y = a_df.group.map(ENC).values
                n01, n10, p = mcnemar(a_df[PCOLS].values, b_df[PCOLS].values, y)
                mc.append({"ref": args.main, "other": name, "eval": ev, "ref_right_other_wrong": n01,
                           "ref_wrong_other_right": n10, "mcnemar_p": p})
    pd.DataFrame(mc).to_csv(OUT / "mcnemar.csv", index=False)

    # figures: coverage curves and confusion matrices for the key pipelines
    key = [k for k in ("image->image logreg", "cnn_ensemble", "gt->gt logreg", args.main,
                       f"gt->{args.main.split('->')[0]} logreg", "classical_lgbm->classical_lgbm logreg") if k in pipes]
    c = pd.DataFrame(cov)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, ev in zip(ax, ("cv", "test")):
        for k in key:
            d = c[(c.pipeline == k) & (c["eval"] == ev)]
            a.plot(d.coverage, d.acc, "o-", label=k, ms=3)
        a.set_title(f"{'CV out-of-fold (n=100)' if ev == 'cv' else 'test (n=50)'}"); a.set_xlabel("coverage (fraction of patients kept, most confident first)")
        a.set_ylabel("accuracy"); a.grid(alpha=0.3); a.set_ylim(0.3, 1.02)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_coverage.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, len(key), figsize=(3.6 * len(key), 3.6))
    for a, k in zip(np.atleast_1d(ax), key):
        te = pipes[k][1].sort_values("pid")
        cm = confusion_matrix(te.group.map(ENC).values, te[PCOLS].values.argmax(1), labels=range(5))
        a.imshow(cm, cmap="Blues", vmin=0, vmax=10)
        for i in range(5):
            for j in range(5):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 5 else "black")
        a.set_xticks(range(5), GROUPS, fontsize=8); a.set_yticks(range(5), GROUPS, fontsize=8)
        a.set_title(f"{k}\nacc {np.trace(cm) / cm.sum():.2f}", fontsize=9); a.set_xlabel("predicted")
    np.atleast_1d(ax)[0].set_ylabel("true")
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion.png", dpi=120); plt.close(fig)
    print(res[["pipeline", "eval", "acc", "acc_lo", "acc_hi", "macro_f1", "nll", "ece"]].round(3).to_string())


if __name__ == "__main__":
    main()
