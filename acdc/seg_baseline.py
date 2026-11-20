"""Classical segmentation baseline (CPU): per-pixel LightGBM on hand-crafted multiscale filter features.

Features per pixel (1.25 mm slices): Gaussian-smoothed intensity, gradient magnitude and the two Hessian eigenvalues
at 6 scales (sigma 0.5-16 px, skimage `multiscale_basic_features`, 24 values) + position: row / column offset from the
image centre (mm) and relative slice position in the stack (base 0 -> apex 1). This is the ilastik-style "random forest
pixel classifier" approach, with gradient-boosted trees instead of a random forest.

Protocol = same as the U-Net: for each of the 5 folds, train on the 80 other training patients (8 of them held out for
early stopping), predict the fold's 20 patients (out-of-fold) and the 50 test patients; test = average of the 5 fold
models. Training pixels are sampled class-balanced per slice; predicted probabilities are corrected back to the true
class priors (p_c * prior_true_c / prior_sampled_c, renormalised). Same post-processing as the U-Net (largest component).

    python -m acdc.seg_baseline --threads 6
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from skimage.feature import multiscale_basic_features

from acdc.common import RESULTS, TARGET_SPACING, WORK
from acdc.predict_seg import finalize, score_and_save

# Setting used for all reported results. tune_seg_baseline.py checked 12 settings by 5-fold CV on the training patients:
# the best (31 leaves, lr 0.1, 200 min samples) was +0.016 sampled Dice, almost all from one fold (within CV noise).
PARAMS = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=63, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)


def volume_features(img):
    """(Z,H,W) -> (Z,H,W,27) float32 feature array."""
    Z, H, W = img.shape
    feats = []
    rr, cc = np.meshgrid((np.arange(H) - H / 2) * TARGET_SPACING, (np.arange(W) - W / 2) * TARGET_SPACING, indexing="ij")
    for z in range(Z):
        f = multiscale_basic_features(img[z], intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16,
                                      num_sigma=6)
        pos = np.stack([rr, cc, np.full((H, W), z / max(Z - 1, 1))], -1)
        feats.append(np.concatenate([f, pos], -1).astype(np.float32))
    return np.stack(feats)


def sample_pixels(F, lab, rng, per_class=150):
    """Class-balanced pixel sample from one volume: up to `per_class` pixels of each class per slice."""
    X, y = [], []
    for z in range(lab.shape[0]):
        for c in range(4):
            idx = np.flatnonzero(lab[z] == c)
            if len(idx):
                pick = rng.choice(idx, min(per_class, len(idx)), replace=False)
                X.append(F[z].reshape(-1, F.shape[-1])[pick]); y.append(np.full(len(pick), c))
    return np.concatenate(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    name = "classical_lgbm"
    (WORK / "preds" / name).mkdir(parents=True, exist_ok=True)
    (RESULTS / "seg").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    rng = np.random.default_rng(42)
    t0 = time.time()

    # 1) features + class-balanced pixel samples for every training volume (cached in memory), true class frequencies
    samples, counts = {}, np.zeros(4)
    for r in meta[meta.split == "train"].itertuples():
        d = np.load(WORK / "cache" / f"{r.pid}.npz")
        for ph in ("ED", "ES"):
            lab = d[f"lab_{ph}"]
            samples[(r.pid, ph)] = sample_pixels(volume_features(d[f"img_{ph}"].astype(np.float32)), lab, rng)
            counts += np.bincount(lab.ravel(), minlength=4)
    print(f"features + samples: {(time.time() - t0) / 60:.1f} min", flush=True)

    # 2) one model per fold
    models, info = [], []
    for f in range(5):
        tr = meta[(meta.split == "train") & (meta.fold != f)].pid.tolist()
        es = rng.choice(tr, 8, replace=False).tolist()  # early-stopping patients (inside the training folds)
        fit = [p for p in tr if p not in es]
        Xa = np.concatenate([samples[(p, ph)][0] for p in fit for ph in ("ED", "ES")])
        ya = np.concatenate([samples[(p, ph)][1] for p in fit for ph in ("ED", "ES")])
        Xe = np.concatenate([samples[(p, ph)][0] for p in es for ph in ("ED", "ES")])
        ye = np.concatenate([samples[(p, ph)][1] for p in es for ph in ("ED", "ES")])
        prior_sampled = np.bincount(ya, minlength=4) / len(ya)
        m = lgb.train({**PARAMS, "num_threads": args.threads}, lgb.Dataset(Xa, ya), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(Xe, ye)], callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append((m, prior_sampled))
        info.append({"fold": f, "n_pixels": int(len(ya)), "best_iter": int(m.best_iteration)})
        print(info[-1], f"{(time.time() - t0) / 60:.1f} min", flush=True)
    prior_true = counts / counts.sum()

    def predict(m_ps, F):
        m, ps = m_ps
        p = m.predict(F.reshape(-1, F.shape[-1]), num_iteration=m.best_iteration, num_threads=args.threads)
        p = p * (prior_true / ps)
        return p / p.sum(1, keepdims=True)

    # 3) predictions: out-of-fold for training patients, 5-model average for test patients
    rows = []
    t1 = time.time()
    for r in meta.itertuples():
        d = np.load(WORK / "cache" / f"{r.pid}.npz")
        orig_xy = tuple(int(v) for v in d["orig_shape"][1:])
        members = [models[r.fold]] if r.split == "train" else models
        out = {}
        for ph in ("ED", "ES"):
            F = volume_features(d[f"img_{ph}"].astype(np.float32))
            per_member = [predict(mm, F).reshape(*F.shape[:3], 4) for mm in members]
            prob = torch.from_numpy(np.mean(per_member, 0)).permute(0, 3, 1, 2).float()
            seg, ent = finalize(prob, orig_xy)
            out[ph] = (seg, ent, None)
        rows += score_and_save(name, r.pid, out, r)
        print(r.pid, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith("dice")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "seg" / f"{name}_metrics.csv", index=False)
    info = {"folds": info, "train_minutes": (t1 - t0) / 60, "predict_minutes": (time.time() - t1) / 60,
            "n_features": 27, "params": PARAMS, "prior_true": prior_true.tolist()}
    (RESULTS / "seg" / f"{name}_info.json").write_text(json.dumps(info, indent=2))
    import joblib
    joblib.dump(models, WORK / "runs" / f"{name}.joblib")
    print(df.groupby(["split", "phase"])[[c for c in df.columns if c.startswith("dice")]].mean().round(4))


if __name__ == "__main__":
    main()
