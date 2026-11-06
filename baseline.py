"""Step 3 - classical baseline (CPU): hand-crafted features + one binary model per target.

  nice -n 10 uv run python baseline.py --threads 6

Features (on 256x256 images, cached to data/features_256.npy):
  * HOG: 9 orientations, 32x32-pixel cells, 2x2-cell blocks          -> 1764 values
  * LBP: uniform patterns (P=8, R=1), 10-bin histogram per cell of a 4x4 grid -> 160
  * Intensity: global 32-bin histogram + mean and std per cell of a 4x4 grid -> 64
Models (majority labels, train split only, tuned by 5-fold stratified CV on train):
  * L2 logistic regression (standardised features), C in {1e-4, 1e-3, 1e-2, 1e-1}
  * LightGBM, num_leaves in {15, 31}, learning rate 0.05, number of trees by CV early stopping
Outputs in results/baseline/:
  cv_results.csv, preds_{logreg,lgbm}_{val,test,disagree}.csv, timing.json, logreg.joblib, lgbm.joblib
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features
from prepare import TARGETS

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
LEAVES_GRID = [15, 31]


def _feat_worker(args):
    path, i = args
    return extract_features(np.load(path, mmap_mode="r")[i])


def compute_features(cache_path, out_path, workers):
    n = np.load(cache_path, mmap_mode="r").shape[0]
    t0 = time.time()
    with Pool(workers) as pool:
        feats = np.stack(pool.map(_feat_worker, [(cache_path, i) for i in range(n)], chunksize=64))
    np.save(out_path, feats)
    return feats, time.time() - t0


def tune_logreg(X, y, k, threads):
    """5-fold CV AUROC for each C; returns (best C, best CV AUROC, all scores)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    for C in C_GRID:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        scores[C] = cross_val_score(pipe, X, y[:, k], cv=cv, scoring="roc_auc", n_jobs=1).mean()
    best = max(scores, key=scores.get)
    return k, best, scores[best], scores


def tune_lgbm(X, y, k, threads):
    """lgb.cv with early stopping for each num_leaves; returns best params and number of trees."""
    results = {}
    for leaves in LEAVES_GRID:
        params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": leaves,
                  "min_child_samples": 20, "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "lambda_l2": 1.0, "num_threads": threads, "verbose": -1, "seed": 42}
        cvres = lgb.cv(params, lgb.Dataset(X, y[:, k]), num_boost_round=1000, nfold=5, stratified=True,
                       seed=42, callbacks=[lgb.early_stopping(50, verbose=False)])
        aucs = cvres["valid auc-mean"]
        results[leaves] = (max(aucs), int(np.argmax(aucs)) + 1, params)
    best = max(results, key=lambda l: results[l][0])
    return results[best], {l: r[0] for l, r in results.items()}


def save_preds(path, ids, probs):
    df = pd.DataFrame(probs, columns=TARGETS)
    df.insert(0, "image_id", ids)
    df.to_csv(path, index=False, float_format="%.6f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--data_dir", default="/workspace/data")
    ap.add_argument("--results_dir", default="/workspace/results")
    args = ap.parse_args()
    out = os.path.join(args.results_dir, "baseline")
    os.makedirs(out, exist_ok=True)
    timing = {}

    # ---------- features (cached) ----------
    feat_path = os.path.join(args.data_dir, "features_256.npy")
    if os.path.exists(feat_path):
        X_all = np.load(feat_path)
    else:
        X_all, secs = compute_features(os.path.join(args.data_dir, "images_512.npy"), feat_path, args.threads)
        timing["feature_extraction_sec_total"] = round(secs, 1)
        print(f"features: {X_all.shape} in {secs:.0f}s", flush=True)

    ids = np.loadtxt(os.path.join(args.data_dir, "image_ids.txt"), dtype=str)
    split = pd.read_csv(os.path.join(args.data_dir, "splits.csv")).split.values
    y_all = pd.read_csv(os.path.join(args.data_dir, "labels_majority.csv"))[TARGETS].values
    tr = split == "train"
    X, y = X_all[tr], y_all[tr]

    # ---------- logistic regression: tune C per target (parallel over targets) ----------
    t0 = time.time()
    lr_res = Parallel(n_jobs=args.threads)(delayed(tune_logreg)(X, y, k, 1) for k in range(len(TARGETS)))
    logreg_models, rows = [], []
    for k, C, auc, scores in sorted(lr_res):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y[:, k])
        logreg_models.append(m)
        rows.append({"target": TARGETS[k], "model": "logreg", "best_param": f"C={C}", "cv_auroc": round(auc, 4),
                     "grid": json.dumps({str(c): round(s, 4) for c, s in scores.items()})})
        print(f"logreg {TARGETS[k]:20s} C={C} cv AUROC {auc:.4f}", flush=True)
    timing["logreg_tune_and_fit_sec"] = round(time.time() - t0, 1)

    # ---------- LightGBM: small grid per target ----------
    t0 = time.time()
    lgbm_models = []
    for k in range(len(TARGETS)):
        (auc, n_trees, params), grid = tune_lgbm(X, y, k, args.threads)
        lgbm_models.append(lgb.train(params, lgb.Dataset(X, y[:, k]), num_boost_round=n_trees))
        rows.append({"target": TARGETS[k], "model": "lgbm", "best_param": f"num_leaves={params['num_leaves']},trees={n_trees}",
                     "cv_auroc": round(auc, 4), "grid": json.dumps({str(l): round(s, 4) for l, s in grid.items()})})
        print(f"lgbm   {TARGETS[k]:20s} leaves={params['num_leaves']} trees={n_trees} cv AUROC {auc:.4f}", flush=True)
    timing["lgbm_tune_and_fit_sec"] = round(time.time() - t0, 1)
    pd.DataFrame(rows).to_csv(os.path.join(out, "cv_results.csv"), index=False)

    # ---------- predictions ----------
    for s in ["val", "test", "disagree"]:
        m = split == s
        save_preds(os.path.join(out, f"preds_logreg_{s}.csv"), ids[m],
                   np.stack([lm.predict_proba(X_all[m])[:, 1] for lm in logreg_models], 1))
        save_preds(os.path.join(out, f"preds_lgbm_{s}.csv"), ids[m],
                   np.stack([gm.predict(X_all[m]) for gm in lgbm_models], 1))
    te = split == "test"
    print("test AUROC (logreg, majority):",
          {TARGETS[k]: round(roc_auc_score(y_all[te, k], logreg_models[k].predict_proba(X_all[te])[:, 1]), 3)
           for k in range(len(TARGETS)) if y_all[te, k].sum() > 0})

    # ---------- inference timing (single image, single thread, CPU) ----------
    cache = np.load(os.path.join(args.data_dir, "images_512.npy"), mmap_mode="r")
    sample = [np.array(cache[i]) for i in np.where(te)[0][:50]]
    t0 = time.time()
    feats = np.stack([extract_features(im) for im in sample])
    t_feat = (time.time() - t0) / len(sample)
    t0 = time.time()
    for f in feats:
        [lm.predict_proba(f[None])[:, 1] for lm in logreg_models]
    t_lr = (time.time() - t0) / len(sample)
    t0 = time.time()
    for f in feats:
        [gm.predict(f[None], num_threads=1) for gm in lgbm_models]
    t_gb = (time.time() - t0) / len(sample)
    timing.update({"cpu_ms_per_image_features": round(1000 * t_feat, 2),
                   "cpu_ms_per_image_logreg_total": round(1000 * (t_feat + t_lr), 2),
                   "cpu_ms_per_image_lgbm_total": round(1000 * (t_feat + t_gb), 2),
                   "n_features": int(X_all.shape[1]),
                   "logreg_n_params": int(sum(lm[-1].coef_.size + 1 for lm in logreg_models)),
                   "lgbm_total_trees": int(sum(gm.num_trees() for gm in lgbm_models))})
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=2)
    joblib.dump(logreg_models, os.path.join(out, "logreg.joblib"))
    joblib.dump(lgbm_models, os.path.join(out, "lgbm.joblib"))
    print(timing)


if __name__ == "__main__":
    main()
