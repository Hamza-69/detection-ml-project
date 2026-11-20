"""Hyper-parameter tuning of the classical segmentation baseline (per-pixel LightGBM, see seg_baseline.py).

Grid: num_leaves {31, 63, 255} x learning_rate {0.05, 0.1} x min_child_samples {50, 200} = 12 configurations
(the original setting 63 / 0.05 / 50 is one of them). Everything else as in seg_baseline.py.

Protocol (training patients only, the test set is never used):
  the frozen 5 folds; for each fold, train on the other 80 patients (8 of them for early stopping, as in
  seg_baseline.py) and score the fold's 20 patients (40 volumes).
Score: Dice of RV, MYO and LV estimated on a uniform random sample of 30,000 pixels per volume (prior-corrected
  argmax, no post-processing), averaged over structures and volumes. Predicting every pixel of every volume would take
  ~2 h per configuration; the pixel sample keeps the TP/FP/FN proportions, so it ranks configurations the same way.
Note: choosing the configuration by this CV makes the baseline's CV numbers slightly optimistic; its test numbers stay
  clean.

Output: results/seg/classical_lgbm_tuning.csv (one row per configuration and fold) and
        results/seg/classical_lgbm_tuning_best.json (best configuration, read by seg_baseline.py --params).

    python -m acdc.tune_seg_baseline --threads 6
"""
import argparse
import itertools
import json
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from acdc.common import RESULTS, WORK
from acdc.seg_baseline import PARAMS, sample_pixels, volume_features

N_UNIFORM = 30000


def dice_from_sample(pred, lab):
    """Mean Dice over classes 1-3 (RV, MYO, LV) from sampled pixels; a class absent in both counts as 1."""
    d = []
    for c in (1, 2, 3):
        p, t = pred == c, lab == c
        d.append(1.0 if p.sum() + t.sum() == 0 else 2 * (p & t).sum() / (p.sum() + t.sum()))
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    train = meta[meta.split == "train"]
    rng = np.random.default_rng(42)  # same seed and sampling order as seg_baseline.py for the balanced samples
    t0 = time.time()

    # 1) features once per volume: balanced training sample (as in seg_baseline) + uniform scoring sample
    bal, uni, counts = {}, {}, np.zeros(4)
    for r in train.itertuples():
        d = np.load(WORK / "cache" / f"{r.pid}.npz")
        for ph in ("ED", "ES"):
            lab = d[f"lab_{ph}"]
            F = volume_features(d[f"img_{ph}"].astype(np.float32))
            bal[(r.pid, ph)] = sample_pixels(F, lab, rng)
            counts += np.bincount(lab.ravel(), minlength=4)
            idx = np.random.default_rng(zlib.crc32(f"{r.pid}_{ph}".encode())).choice(  # reproducible per volume
                lab.size, min(N_UNIFORM, lab.size), replace=False)
            uni[(r.pid, ph)] = (F.reshape(-1, F.shape[-1])[idx], lab.ravel()[idx])
    prior_true = counts / counts.sum()
    print(f"features: {(time.time() - t0) / 60:.1f} min", flush=True)

    # same early-stopping patients per fold for every configuration
    es_pat = {f: rng.choice(train[train.fold != f].pid.tolist(), 8, replace=False).tolist() for f in range(5)}
    out_csv = RESULTS / "seg" / "classical_lgbm_tuning.csv"
    rows = []
    grid = list(itertools.product([31, 63, 255], [0.05, 0.1], [50, 200]))
    for leaves, lr, mcs in grid:
        params = {**PARAMS, "num_leaves": leaves, "learning_rate": lr, "min_child_samples": mcs, "num_threads": args.threads}
        for f in range(5):
            t = time.time()
            fit = [p for p in train[train.fold != f].pid if p not in es_pat[f]]
            stack = lambda pids: [np.concatenate(z) for z in zip(*[bal[(p, ph)] for p in pids for ph in ("ED", "ES")])]  # noqa: E731
            (Xa, ya), (Xe, ye) = stack(fit), stack(es_pat[f])
            prior_sampled = np.bincount(ya, minlength=4) / len(ya)
            m = lgb.train(params, lgb.Dataset(Xa, ya), num_boost_round=3000, valid_sets=[lgb.Dataset(Xe, ye)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            dices = []
            for p in train[train.fold == f].pid:
                for ph in ("ED", "ES"):
                    X, y = uni[(p, ph)]
                    pr = m.predict(X, num_iteration=m.best_iteration) * (prior_true / prior_sampled)
                    dices.append(dice_from_sample(pr.argmax(1), y))
            rows.append({"num_leaves": leaves, "learning_rate": lr, "min_child_samples": mcs, "fold": f,
                         "dice_sampled": float(np.mean(dices)), "best_iter": int(m.best_iteration),
                         "train_s": round(time.time() - t, 1)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)  # written after every fold, so partial results survive

    df = pd.DataFrame(rows)
    s = df.groupby(["num_leaves", "learning_rate", "min_child_samples"]).agg(
        dice=("dice_sampled", "mean"), dice_sd=("dice_sampled", "std"), best_iter=("best_iter", "mean"),
        train_s=("train_s", "sum")).reset_index().sort_values("dice", ascending=False)
    print(s.round(4).to_string(index=False))
    b = s.iloc[0]
    orig = s[(s.num_leaves == 63) & (s.learning_rate == 0.05) & (s.min_child_samples == 50)].iloc[0]
    best = {"params": {**PARAMS, "num_leaves": int(b.num_leaves), "learning_rate": float(b.learning_rate),
                       "min_child_samples": int(b.min_child_samples)},
            "cv_dice_sampled": float(b.dice), "cv_dice_sampled_sd_over_folds": float(b.dice_sd),
            "original_cv_dice_sampled": float(orig.dice), "n_configs": len(grid),
            "minutes": (time.time() - t0) / 60}
    (RESULTS / "seg" / "classical_lgbm_tuning_best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
