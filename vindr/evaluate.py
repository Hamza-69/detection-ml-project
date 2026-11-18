"""Step 5 - evaluate every trained model and write results/summary.md.

  uv run python evaluate.py            # metrics, tables, figures, summary.md
  uv run python evaluate.py --timing   # also measure inference speed (run when the GPU is idle)

Ground truth on the internal test set = MAJORITY labels (main tables).
Metrics:
  * AUROC per finding, macro-AUROC (mean over the 14 findings), abnormal AUROC; 95% bootstrap CIs
    (1,000 resamples of test images). Paired bootstrap for soft - majority differences.
  * ECE (15 equal-width bins) per output, macro-ECE = mean over the 14 findings; Brier score.
  * Temperature scaling: one T per model fit on val (BCE vs majority labels, all 15 outputs pooled).
  * Sensitivity / specificity at threshold 0.5 (macro over findings).
  * Disagreement set: mean probability per agreement level (0/3..3/3), Spearman(prob, votes/3),
    ECE and mean |p - votes/3| against the soft targets.
"""
import argparse
import glob
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

from prepare import FINDINGS, TARGETS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/results"
D = "/workspace/data"
B = 1000          # bootstrap resamples
N_BINS = 15
RNG = np.random.default_rng(0)
COLORS = {"majority": "#1f77b4", "soft": "#2ca02c", "union": "#d62728", "logreg": "#7f7f7f", "lgbm": "#9467bd"}


# ---------------------------------------------------------------- metrics
def auc(y, s):
    """AUROC via the rank formula (ties get average ranks). NaN if y has one class only."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(p, y, bins=N_BINS):
    """Expected calibration error: sum over bins of (bin size / n) * |mean prob - mean label|.
    y may be soft (fraction of readers)."""
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return total


def per_class(fn, P, Y, ks):
    return np.array([fn(Y[:, k], P[:, k]) if fn is auc else fn(P[:, k], Y[:, k]) for k in ks])


def boot_ci(stat_fn, n, idx_sets):
    vals = np.array([stat_fn(ix) for ix in idx_sets])
    vals = vals[~np.isnan(vals)]
    return np.percentile(vals, [2.5, 97.5])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(z_val, y_val):
    """Single T minimising BCE of sigmoid(z / T) on the validation set (all outputs pooled)."""
    def nll(logT):
        p = np.clip(sigmoid(z_val / np.exp(logT)), 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


# ---------------------------------------------------------------- loading
def load_truth():
    ids = np.loadtxt(f"{D}/image_ids.txt", dtype=str)
    split = pd.read_csv(f"{D}/splits.csv").split.values
    lab = {n: pd.read_csv(f"{D}/labels_{n}.csv").set_index("image_id")[TARGETS] for n in ["majority", "soft", "union"]}
    votes = pd.read_csv(f"{D}/votes.csv").set_index("image_id")[TARGETS]
    return ids, split, lab, votes


def load_models():
    """Returns dict name -> {kind, labels, seed, variant, logits per split}."""
    models = {}
    for d in sorted(glob.glob(f"{R}/cnn_*_s*")):
        if d.endswith("_smoke") or not os.path.exists(f"{d}/preds_test.csv"):
            continue
        info = json.load(open(f"{d}/run.json"))
        name = os.path.basename(d)
        variant = "nopw" if name.endswith("_nopw") else "pw"
        m = {"kind": "cnn", "labels": info["labels"], "seed": info["seed"], "variant": variant, "info": info}
        for s in ["val", "test", "disagree"]:
            df = pd.read_csv(f"{d}/preds_{s}.csv").set_index("image_id")
            m[s] = df[[f"logit_{t}" for t in TARGETS]].rename(columns=lambda c: c[6:])
        models[name] = m
    for kind in ["logreg", "lgbm"]:
        if not os.path.exists(f"{R}/baseline/preds_{kind}_test.csv"):
            continue
        m = {"kind": "baseline", "labels": "majority", "seed": 0, "variant": kind}
        for s in ["val", "test", "disagree"]:
            p = pd.read_csv(f"{R}/baseline/preds_{kind}_{s}.csv").set_index("image_id")[TARGETS].clip(1e-6, 1 - 1e-6)
            m[s] = np.log(p / (1 - p))  # store as logits so everything is handled the same way
        models[f"baseline_{kind}"] = m
    return models


# ---------------------------------------------------------------- main evaluation
def evaluate_model(name, m, lab, votes, boot_idx):
    ks = list(range(len(FINDINGS)))
    test_ids = m["test"].index
    Z = m["test"].values
    P = sigmoid(Z)
    Y = lab["majority"].loc[test_ids].values
    zv = m["val"].values
    yv = lab["majority"].loc[m["val"].index].values
    T = fit_temperature(zv, yv)
    PT = sigmoid(Z / T)
    # per-class Platt scaling (p = sigmoid(a*z + b) per output, fit on val): can remove a shift, unlike one temperature
    PP = np.column_stack([LogisticRegression(C=1e4, max_iter=1000).fit(zv[:, k:k + 1], yv[:, k]).predict_proba(Z[:, k:k + 1])[:, 1]
                          for k in range(15)])

    aucs = per_class(auc, P, Y, range(15))
    macro = np.nanmean(aucs[:14])
    # AUROC among abnormal images only: can the model tell WHICH finding is present, not just normal vs abnormal?
    abn = Y[:, 14] == 1
    aucs_abn = np.array([auc(Y[abn, k], P[abn, k]) for k in ks])
    ci_macro = boot_ci(lambda ix: np.nanmean([auc(Y[ix, k], P[ix, k]) for k in ks]), len(Y), boot_idx)
    ci_abn = boot_ci(lambda ix: auc(Y[ix, 14], P[ix, 14]), len(Y), boot_idx)
    eces = per_class(ece, P, Y, range(15))
    eces_T = per_class(ece, PT, Y, range(15))
    eces_P = per_class(ece, PP, Y, range(15))
    briers = ((P - Y) ** 2).mean(0)
    briers_T = ((PT - Y) ** 2).mean(0)
    pred = P >= 0.5
    sens = np.array([pred[Y[:, k] == 1, k].mean() for k in ks])
    spec = np.array([1 - pred[Y[:, k] == 0, k].mean() for k in ks])

    # disagreement set
    dz = m["disagree"]
    dP = sigmoid(dz.values[:, :14])
    dPT = sigmoid(dz.values[:, :14] / T)
    dv = votes.loc[dz.index].values[:, :14]
    level_means = [float(dP[dv == v].mean()) for v in range(4)]
    level_means_T = [float(dPT[dv == v].mean()) for v in range(4)]
    pos = dv >= 1  # pairs where at least one reader marked the finding
    split_vote = (dv == 1) | (dv == 2)
    rho_all = spearmanr(dP.ravel(), dv.ravel()).correlation
    rho_pos = spearmanr(dP[pos], dv[pos]).correlation
    soft_t = dv / 3.0

    row = {
        "model": name, "kind": m["kind"], "labels": m["labels"], "seed": m["seed"], "variant": m["variant"],
        "macro_auroc": macro, "macro_auroc_lo": ci_macro[0], "macro_auroc_hi": ci_macro[1],
        "abnormal_auroc": aucs[14], "abnormal_auroc_lo": ci_abn[0], "abnormal_auroc_hi": ci_abn[1],
        "macro_auroc_within_abnormal": np.nanmean(aucs_abn),
        "macro_ece": eces[:14].mean(), "macro_ece_T": eces_T[:14].mean(), "macro_ece_platt": eces_P[:14].mean(),
        "macro_brier_platt": ((PP - Y) ** 2).mean(0)[:14].mean(),
        "abnormal_ece": eces[14], "abnormal_ece_T": eces_T[14],
        "macro_brier": briers[:14].mean(), "macro_brier_T": briers_T[:14].mean(), "T": T,
        "macro_sens@0.5": np.nanmean(sens), "macro_spec@0.5": np.nanmean(spec),
        "abnormal_sens@0.5": pred[Y[:, 14] == 1, 14].mean(), "abnormal_spec@0.5": 1 - pred[Y[:, 14] == 0, 14].mean(),
        "dis_mean_p_0of3": level_means[0], "dis_mean_p_1of3": level_means[1],
        "dis_mean_p_2of3": level_means[2], "dis_mean_p_3of3": level_means[3],
        "dis_mean_p_T_0of3": level_means_T[0], "dis_mean_p_T_1of3": level_means_T[1],
        "dis_mean_p_T_2of3": level_means_T[2], "dis_mean_p_T_3of3": level_means_T[3],
        "dis_spearman_all": rho_all, "dis_spearman_marked": rho_pos,
        "dis_ece_vs_soft": np.mean([ece(dP[:, k], soft_t[:, k]) for k in ks]),
        "dis_ece_vs_soft_T": np.mean([ece(dPT[:, k], soft_t[:, k]) for k in ks]),
        "dis_mae_vs_soft_splitvote": float(np.abs(dP - soft_t)[split_vote].mean()),
        "dis_mae_vs_soft_splitvote_T": float(np.abs(dPT - soft_t)[split_vote].mean()),
    }
    per = pd.DataFrame({"model": name, "target": TARGETS, "auroc": aucs,
                        "auroc_within_abnormal": list(aucs_abn) + [np.nan], "ece": eces, "ece_T": eces_T,
                        "brier": briers, "sens@0.5": list(sens) + [np.nan], "spec@0.5": list(spec) + [np.nan]})
    return row, per, (P, PT, Y)


def paired_diff(models, a, b, lab, boot_idx):
    """Paired bootstrap of (a - b) for macro-AUROC and macro-ECE on the test set."""
    ids = models[a]["test"].index
    Y = lab["majority"].loc[ids].values
    Pa, Pb = sigmoid(models[a]["test"].values), sigmoid(models[b]["test"].values)
    ks = range(14)
    d_auc = [np.nanmean([auc(Y[ix, k], Pa[ix, k]) for k in ks]) - np.nanmean([auc(Y[ix, k], Pb[ix, k]) for k in ks]) for ix in boot_idx]
    d_ece = [np.mean([ece(Pa[ix, k], Y[ix, k]) for k in ks]) - np.mean([ece(Pb[ix, k], Y[ix, k]) for k in ks]) for ix in boot_idx]
    return {"a": a, "b": b,
            "d_macro_auroc": np.nanmean([auc(Y[:, k], Pa[:, k]) for k in ks]) - np.nanmean([auc(Y[:, k], Pb[:, k]) for k in ks]),
            "d_macro_auroc_lo": np.percentile(d_auc, 2.5), "d_macro_auroc_hi": np.percentile(d_auc, 97.5),
            "d_macro_ece": np.mean([ece(Pa[:, k], Y[:, k]) for k in ks]) - np.mean([ece(Pb[:, k], Y[:, k]) for k in ks]),
            "d_macro_ece_lo": np.percentile(d_ece, 2.5), "d_macro_ece_hi": np.percentile(d_ece, 97.5)}


# ---------------------------------------------------------------- figures
def reliability_curve(P, Y):
    idx = np.clip((P * N_BINS).astype(int), 0, N_BINS - 1)
    xs, ys, ns = [], [], []
    for b in range(N_BINS):
        m = idx == b
        if m.sum() >= 20:
            xs.append(P[m].mean()); ys.append(Y[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def fig_reliability(preds, path, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, use_T, title in [(axes[0], False, "before temperature scaling"), (axes[1], True, "after temperature scaling")]:
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for lbl, (P, PT, Y) in preds.items():
            x, y, n = reliability_curve((PT if use_T else P)[:, :14].ravel(), Y[:, :14].ravel())
            ax.plot(x, y, "o-", color=COLORS[lbl], label=f"CNN {lbl}", ms=4)
        ax.set_title(f"14 findings pooled, {title}", fontsize=10)
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed frequency (majority label)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Reliability diagram, internal test set, CNN seed 42 {tag} (bins with >= 20 image-finding pairs)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_h2(summary, path, suffix="", variant="pw"):
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for lbl in ["majority", "soft", "union"]:
        rows = summary[(summary.kind == "cnn") & (summary.labels == lbl) & (summary.variant == variant)]
        if len(rows) == 0:
            continue
        cols = [f"dis_mean_p{suffix}_{v}of3" for v in range(4)]
        mean = rows[cols].mean().values
        ax.plot(range(4), mean, "o-", color=COLORS[lbl], label=f"CNN {lbl} ({len(rows)} seed{'s' * (len(rows) > 1)})")
        if len(rows) > 1:
            ax.fill_between(range(4), rows[cols].min().values, rows[cols].max().values, color=COLORS[lbl], alpha=0.15)
    ax.plot(range(4), [0, 1 / 3, 2 / 3, 1], "k--", lw=1, label="reader fraction")
    ax.set_xticks(range(4), ["0/3", "1/3", "2/3", "3/3"])
    ax.set_xlabel("readers who marked the finding")
    ax.set_ylabel("mean predicted probability")
    ax.set_title("Disagreement set: probability vs reader agreement\n(14 findings pooled, " +
                 ("with pos_weight" if variant == "pw" else "no pos_weight") + (", temperature-scaled)" if suffix else ")"), fontsize=10)
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_h2_both(summary, path):
    """H2 plot, two panels: with and without pos_weight (mean over seeds, band = range over seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, variant, title in [(axes[0], "pw", "with pos_weight"), (axes[1], "nopw", "without pos_weight")]:
        for lbl in ["majority", "soft", "union"]:
            rows = summary[(summary.kind == "cnn") & (summary.labels == lbl) & (summary.variant == variant)]
            if len(rows) == 0:
                continue
            cols = [f"dis_mean_p_{v}of3" for v in range(4)]
            ax.plot(range(4), rows[cols].mean().values, "o-", color=COLORS[lbl], label=f"CNN {lbl}")
            ax.fill_between(range(4), rows[cols].min().values, rows[cols].max().values, color=COLORS[lbl], alpha=0.15)
        ax.plot(range(4), [0, 1 / 3, 2 / 3, 1], "k--", lw=1, label="reader fraction")
        ax.set_xticks(range(4), ["0/3", "1/3", "2/3", "3/3"])
        ax.set_xlabel("readers who marked the finding")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean predicted probability")
    axes[0].legend(fontsize=8)
    fig.suptitle("Disagreement set: predicted probability vs reader agreement (14 findings pooled, 3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_ladder(per, cnn_names, base_name, counts, path):
    cnn = per[per.model.isin(cnn_names)].groupby("target").auroc.mean()
    base = per[per.model == base_name].set_index("target").auroc
    order = sorted(FINDINGS, key=lambda t: -counts[t]) + ["Abnormal"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, [base[t] for t in order], 0.4, color=COLORS["logreg"], label=base_name.replace("baseline_", "baseline "))
    ax.bar(x + 0.2, [cnn[t] for t in order], 0.4, color=COLORS["majority"], label=f"CNN majority (mean of {len(cnn_names)} seed(s))")
    ax.set_xticks(x, [f"{t}\n(n={counts[t]})" if t != "Abnormal" else t for t in order], rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1); ax.set_ylabel("AUROC (internal test, majority labels)")
    ax.set_title("Model ladder: per-finding AUROC (n = majority positives in train)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- timing
def measure_timing(n_threads_cpu=4):
    import torch
    from train_cnn import build_model, to_input
    out = {}
    model = build_model().eval()
    out["cnn_n_params"] = sum(p.numel() for p in model.parameters())
    ckpts = glob.glob(f"{R}/cnn_majority_s42/best.pt")
    out["cnn_checkpoint_MB"] = round(os.path.getsize(ckpts[0]) / 1e6, 1) if ckpts else None
    x = torch.randint(0, 255, (64, 512, 512), dtype=torch.uint8)
    if torch.cuda.is_available():
        m = model.cuda().to(memory_format=torch.channels_last)
        xg = x.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for bs, key in [(64, "cnn_gpu_ms_per_image_batch64"), (1, "cnn_gpu_ms_latency_batch1")]:
                for _ in range(5):
                    m(to_input(xg[:bs]))
                torch.cuda.synchronize(); t = time.time()
                for _ in range(20):
                    m(to_input(xg[:bs]))
                torch.cuda.synchronize()
                out[key] = round(1000 * (time.time() - t) / (20 * bs), 3)
        out["gpu"] = torch.cuda.get_device_name()
        model = model.cpu()
    torch.set_num_threads(n_threads_cpu)
    model = model.to(memory_format=torch.contiguous_format)
    with torch.no_grad():
        model(to_input(x[:1]))
        t = time.time()
        for _ in range(10):
            model(to_input(x[:1]))
    out["cnn_cpu_ms_per_image_batch1"] = round(100 * (time.time() - t), 1)
    out["cpu_threads"] = n_threads_cpu
    yw = f"{R}/yolo/yolov8s_majority/weights/best.pt"
    if os.path.exists(yw):
        from ultralytics import YOLO
        y = YOLO(yw)
        out["yolo_n_params"] = sum(p.numel() for p in y.model.parameters())
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        y.predict(img, imgsz=640, device="cpu", verbose=False)
        t = time.time()
        for _ in range(10):
            y.predict(img, imgsz=640, device="cpu", verbose=False)
        out["yolo_cpu_ms_per_image"] = round(100 * (time.time() - t), 1)
    json.dump(out, open(f"{R}/timing_cnn.json", "w"), indent=2)
    return out


# ---------------------------------------------------------------- summary writer
def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(df, cols, headers=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    os.makedirs(f"{R}/tables", exist_ok=True)

    ids, split, lab, votes = load_truth()
    models = load_models()
    print("models:", list(models))
    n_test = (split == "test").sum()
    boot_idx = [RNG.integers(0, n_test, n_test) for _ in range(B)]

    rows, pers, preds_main = [], [], {}
    for name, m in models.items():
        t0 = time.time()
        row, per, pr = evaluate_model(name, m, lab, votes, boot_idx)
        rows.append(row); pers.append(per)
        if m["kind"] == "cnn" and m["seed"] == 42:
            preds_main.setdefault(m["variant"], {})[m["labels"]] = pr
        print(f"  {name}: macroAUROC {row['macro_auroc']:.4f} ECE {row['macro_ece']:.4f} -> {row['macro_ece_T']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    S = pd.DataFrame(rows)
    PER = pd.concat(pers)
    S.to_csv(f"{R}/tables/all_models.csv", index=False)
    PER.to_csv(f"{R}/tables/per_class.csv", index=False)

    # paired differences (same seed), soft - majority and union - majority
    diffs = []
    for variant in ["pw", "nopw"]:
        sfx = "_nopw" if variant == "nopw" else ""
        for seed in sorted(S[S.kind == "cnn"].seed.unique()):
            for other in ["soft", "union"]:
                a, b = f"cnn_{other}_s{seed}{sfx}", f"cnn_majority_s{seed}{sfx}"
                if a in models and b in models:
                    diffs.append({**paired_diff(models, a, b, lab, boot_idx), "seed": seed, "variant": variant})
    DIFF = pd.DataFrame(diffs)
    DIFF.to_csv(f"{R}/tables/paired_differences.csv", index=False)

    # seed spread per label type
    cnn = S[S.kind == "cnn"]
    agg = cnn.groupby(["variant", "labels"]).agg(
        n_seeds=("seed", "count"), macro_auroc_mean=("macro_auroc", "mean"), macro_auroc_sd=("macro_auroc", "std"),
        macro_ece_mean=("macro_ece", "mean"), macro_ece_sd=("macro_ece", "std"),
        macro_ece_T_mean=("macro_ece_T", "mean"), macro_ece_T_sd=("macro_ece_T", "std"),
        macro_brier_mean=("macro_brier", "mean"), sens_mean=("macro_sens@0.5", "mean"), spec_mean=("macro_spec@0.5", "mean"),
        dis_mae_mean=("dis_mae_vs_soft_splitvote", "mean"), dis_rho_marked_mean=("dis_spearman_marked", "mean"),
        dis_ece_soft_mean=("dis_ece_vs_soft", "mean")).reset_index()
    agg.to_csv(f"{R}/tables/seed_aggregate.csv", index=False)

    # figures
    for variant, sfx in [("pw", ""), ("nopw", "_nopw")]:
        if variant in preds_main:
            fig_reliability(preds_main[variant], f"{R}/fig_reliability{sfx}.png",
                            "with pos_weight" if variant == "pw" else "no pos_weight")
        if ((S.kind == "cnn") & (S.variant == variant)).any():
            fig_h2(S, f"{R}/fig_h2_disagreement{sfx}.png", variant=variant)
            fig_h2(S, f"{R}/fig_h2_disagreement{sfx}_T.png", suffix="_T", variant=variant)
    fig_h2_both(S, f"{R}/fig_h2_both.png")
    counts = pd.read_csv(f"{R}/split_counts.csv", index_col=0)["train"].to_dict()
    maj_names = [n for n, m in models.items() if m["kind"] == "cnn" and m["labels"] == "majority" and m["variant"] == "pw"]
    if maj_names and "baseline_logreg" in models:
        fig_ladder(PER, maj_names, "baseline_logreg", counts, f"{R}/fig_ladder_auroc.png")

    # model ladder table (per class AUROC): baselines vs CNN majority (mean over seeds)
    ladder = PER[PER.model.isin(["baseline_logreg", "baseline_lgbm"])].pivot(index="target", columns="model", values="auroc")
    if maj_names:
        ladder["cnn_majority"] = PER[PER.model.isin(maj_names)].groupby("target").auroc.mean()
        ladder["gain_vs_best_baseline"] = ladder.cnn_majority - ladder[[c for c in ladder.columns if c.startswith("baseline")]].max(1)
    # same, restricted to abnormal test images (which finding, not whether abnormal)
    lad_abn = PER[PER.model.isin(["baseline_logreg", "baseline_lgbm"])].pivot(index="target", columns="model", values="auroc_within_abnormal")
    if maj_names:
        lad_abn["cnn_majority"] = PER[PER.model.isin(maj_names)].groupby("target").auroc_within_abnormal.mean()
    for c in lad_abn.columns:
        ladder[f"{c}_within_abn"] = lad_abn[c]
    ladder["train_pos"] = [counts.get(t, np.nan) for t in ladder.index]
    ladder = ladder.loc[TARGETS]
    ladder.to_csv(f"{R}/tables/ladder_per_class.csv")
    # H4: is the CNN's gain smaller for rare findings? (14 findings; rare = < 150 majority train positives)
    h4 = {}
    if "gain_vs_best_baseline" in ladder:
        f = ladder.loc[FINDINGS].copy()
        best_b_abn = f[[c for c in f.columns if c.startswith("baseline") and c.endswith("_within_abn")]].max(1)
        f["gain_within_abn"] = f["cnn_majority_within_abn"] - best_b_abn
        rare = f.train_pos < 150
        h4 = {"spearman_gain_vs_trainpos": spearmanr(f.gain_vs_best_baseline, f.train_pos).correlation,
              "spearman_gain_within_abn_vs_trainpos": spearmanr(f.gain_within_abn, f.train_pos).correlation,
              "mean_gain_rare": f.gain_vs_best_baseline[rare].mean(), "mean_gain_common": f.gain_vs_best_baseline[~rare].mean(),
              "mean_gain_within_abn_rare": f.gain_within_abn[rare].mean(), "mean_gain_within_abn_common": f.gain_within_abn[~rare].mean(),
              "rare_classes": ", ".join(f.index[rare]), "n_rare": int(rare.sum())}
        json.dump(h4, open(f"{R}/tables/h4_gain_vs_frequency.json", "w"), indent=2)

    # per-finding disagreement-set behaviour (seed 42 models): Spearman(prob, votes) on pairs with >= 1 vote
    rows_pf = []
    for name, m in models.items():
        if not ((m["kind"] == "cnn" and m["seed"] == 42) or m["kind"] == "baseline"):
            continue
        dv = votes.loc[m["disagree"].index].values
        dP = sigmoid(m["disagree"].values)
        for k, t in enumerate(FINDINGS):
            sel = dv[:, k] >= 1
            rows_pf.append({"model": name, "finding": t, "n_marked": int(sel.sum()),
                            "spearman_marked": spearmanr(dP[sel, k], dv[sel, k]).correlation if sel.sum() > 5 else np.nan,
                            **{f"mean_p_{v}of3": dP[dv[:, k] == v, k].mean() if (dv[:, k] == v).any() else np.nan for v in range(4)}})
    pd.DataFrame(rows_pf).to_csv(f"{R}/tables/disagreement_per_finding.csv", index=False)

    timing = measure_timing() if args.timing else (json.load(open(f"{R}/timing_cnn.json")) if os.path.exists(f"{R}/timing_cnn.json") else {})
    btime = json.load(open(f"{R}/baseline/timing.json")) if os.path.exists(f"{R}/baseline/timing.json") else {}
    yolo = pd.read_csv(f"{R}/yolo/map40_test.csv") if os.path.exists(f"{R}/yolo/map40_test.csv") else None

    write_summary(S, PER, DIFF, agg, ladder, timing, btime, yolo, models, h4)
    print("wrote", f"{R}/summary.md")


def write_summary(S, PER, DIFF, agg, ladder, timing, btime, yolo, models, h4):
    L = ["# Preliminary results (auto-generated by evaluate.py)", "",
         "**All numbers are preliminary**: internal test split (2,250 images held out from the 15,000 "
         "VinDr-CXR training images; official test labels not available), 512 px, at most 12 epochs, few seeds. "
         "Ground truth for test metrics = majority vote (>= 2 of 3 readers). Macro = mean over the 14 findings "
         "(Abnormal reported separately). CIs = 95% bootstrap over test images (1,000 resamples).", ""]

    # Table 1 ladder
    L += ["## Table 1 - Model ladder: per-finding AUROC (internal test)", "",
          "Columns ending `_within_abn` = AUROC computed only on abnormal test images (majority label), i.e. "
          "telling findings apart rather than normal vs abnormal.", ""]
    t1 = ladder.reset_index().rename(columns={"index": "target"})
    for c in t1.columns[1:]:
        t1[c] = t1[c].map(lambda v: fmt(v, 3) if c != "train_pos" else str(int(v)) if not np.isnan(v) else "–")
    L += [md_table(t1, list(t1.columns)), ""]
    base_rows = S[S.kind == "baseline"].copy()
    if len(base_rows):
        L += ["Baseline summary:", ""]
        base_rows["macro AUROC [95% CI]"] = base_rows.apply(lambda r: f"{r.macro_auroc:.3f} [{r.macro_auroc_lo:.3f}, {r.macro_auroc_hi:.3f}]", axis=1)
        base_rows["abnormal AUROC [95% CI]"] = base_rows.apply(lambda r: f"{r.abnormal_auroc:.3f} [{r.abnormal_auroc_lo:.3f}, {r.abnormal_auroc_hi:.3f}]", axis=1)
        base_rows["macro ECE"] = base_rows.macro_ece.map(fmt)
        base_rows["macro AUROC within abnormal"] = base_rows.macro_auroc_within_abnormal.map(fmt)
        L += [md_table(base_rows, ["model", "macro AUROC [95% CI]", "macro AUROC within abnormal", "abnormal AUROC [95% CI]", "macro ECE"]), ""]
    if h4:
        L += [f"H4 check (14 findings; rare = < 150 majority train positives: {h4['rare_classes']}): "
              f"mean AUROC gain of CNN over best baseline = {h4['mean_gain_rare']:.3f} (rare) vs {h4['mean_gain_common']:.3f} (common); "
              f"within abnormal images {h4['mean_gain_within_abn_rare']:.3f} (rare) vs {h4['mean_gain_within_abn_common']:.3f} (common). "
              f"Spearman(gain, train positives) = {h4['spearman_gain_vs_trainpos']:.2f} "
              f"(within abnormal: {h4['spearman_gain_within_abn_vs_trainpos']:.2f}). "
              "`results/tables/h4_gain_vs_frequency.json`", ""]
    L += ["Figure: `results/fig_ladder_auroc.png`", ""]

    # Table 2 label strategies, each run
    for variant, title in [("pw", "with pos_weight (main setting, as specified)"), ("nopw", "without pos_weight (calibration ablation)")]:
        rows = S[(S.kind == "cnn") & (S.variant == variant)].sort_values(["labels", "seed"]).copy()
        if not len(rows):
            continue
        rows["macro AUROC [95% CI]"] = rows.apply(lambda r: f"{r.macro_auroc:.3f} [{r.macro_auroc_lo:.3f}, {r.macro_auroc_hi:.3f}]", axis=1)
        rows["abnormal AUROC"] = rows.abnormal_auroc.map(fmt)
        rows["within-abn AUROC"] = rows.macro_auroc_within_abnormal.map(fmt)
        rows["macro ECE (before -> after T)"] = rows.apply(lambda r: f"{r.macro_ece:.4f} -> {r.macro_ece_T:.4f}", axis=1)
        rows["T"] = rows["T"].map(lambda v: fmt(v, 2))
        rows["macro Brier"] = rows.macro_brier.map(lambda v: fmt(v, 4))
        rows["macro sens / spec @0.5"] = rows.apply(lambda r: f"{r['macro_sens@0.5']:.3f} / {r['macro_spec@0.5']:.3f}", axis=1)
        L += [f"## Table 2{'a' if variant == 'pw' else 'b'} - Label strategies, CNN {title}", "",
              md_table(rows, ["labels", "seed", "macro AUROC [95% CI]", "within-abn AUROC", "abnormal AUROC", "macro ECE (before -> after T)", "T",
                              "macro Brier", "macro sens / spec @0.5"]), ""]
        a = agg[agg.variant == variant].copy()
        for c in ["macro_auroc_mean", "macro_auroc_sd", "macro_ece_mean", "macro_ece_sd", "macro_ece_T_mean", "macro_ece_T_sd"]:
            a[c] = a[c].map(lambda v: fmt(v, 4))
        L += ["Mean and SD across seeds:", "",
              md_table(a, ["labels", "n_seeds", "macro_auroc_mean", "macro_auroc_sd", "macro_ece_mean", "macro_ece_sd",
                           "macro_ece_T_mean", "macro_ece_T_sd"]), ""]
        d = DIFF[DIFF.variant == variant].copy() if len(DIFF) else pd.DataFrame()
        if len(d):
            d["d macro AUROC [95% CI]"] = d.apply(lambda r: f"{r.d_macro_auroc:+.4f} [{r.d_macro_auroc_lo:+.4f}, {r.d_macro_auroc_hi:+.4f}]", axis=1)
            d["d macro ECE [95% CI]"] = d.apply(lambda r: f"{r.d_macro_ece:+.4f} [{r.d_macro_ece_lo:+.4f}, {r.d_macro_ece_hi:+.4f}]", axis=1)
            d["comparison"] = d.a.str.replace("cnn_", "") + " - " + d.b.str.replace("cnn_", "")
            L += ["Paired bootstrap differences on the test set (same seed):", "",
                  md_table(d, ["comparison", "d macro AUROC [95% CI]", "d macro ECE [95% CI]"]), ""]

    # disagreement set
    rows = S[S.kind == "cnn"].sort_values(["variant", "labels", "seed"]).copy()
    if len(rows):
        for c in ["dis_mean_p_0of3", "dis_mean_p_1of3", "dis_mean_p_2of3", "dis_mean_p_3of3", "dis_spearman_all",
                  "dis_spearman_marked", "dis_ece_vs_soft", "dis_mae_vs_soft_splitvote", "dis_mae_vs_soft_splitvote_T"]:
            rows[c] = rows[c].map(fmt)
        L += ["## Table 3 - Disagreement set (2,250 held-out images, all 3 votes kept; 14 findings pooled)", "",
              "mean p at k/3 = mean predicted probability over image-finding pairs that k of 3 readers marked. "
              "Spearman 'marked' = only pairs with >= 1 vote. MAE split-vote = mean |p - k/3| over pairs with 1 or 2 votes.", "",
              md_table(rows, ["model", "dis_mean_p_0of3", "dis_mean_p_1of3", "dis_mean_p_2of3", "dis_mean_p_3of3",
                              "dis_spearman_all", "dis_spearman_marked", "dis_ece_vs_soft", "dis_mae_vs_soft_splitvote",
                              "dis_mae_vs_soft_splitvote_T"],
                       ["model", "p@0/3", "p@1/3", "p@2/3", "p@3/3", "Spearman all", "Spearman marked", "ECE vs soft",
                        "MAE split-vote", "MAE split-vote (T)"]), "",
              "Figures: `results/fig_h2_disagreement.png` (raw), `results/fig_h2_disagreement_T.png` (temperature-scaled)", ""]

    L += ["## Reliability diagram", "", "`results/fig_reliability.png` (seed 42, with pos_weight, 14 findings pooled).", ""]

    if yolo is not None:
        L += ["## YOLOv8s detection, internal test, mAP@0.4 (majority-fused boxes)", "", md_table(yolo, list(yolo.columns)), ""]

    # complexity
    cnn_runs = [m["info"] for m in models.values() if m["kind"] == "cnn"]
    L += ["## Complexity", "",
          "| | parameters | training time | inference CPU (ms/image) | inference GPU (ms/image) |", "|---|---|---|---|---|"]
    if btime:
        L.append(f"| LogReg (HOG+LBP+hist) | {btime.get('logreg_n_params', '–')} | "
                 f"{(btime.get('feature_extraction_sec_total', 0) + btime.get('logreg_tune_and_fit_sec', 0)) / 60:.1f} CPU-min (6 cores, incl. 5-fold CV + features) | "
                 f"{btime.get('cpu_ms_per_image_logreg_total', '–')} (1 thread, incl. features) | n/a |")
        L.append(f"| LightGBM | {btime.get('lgbm_total_trees', '–')} trees | {btime.get('lgbm_tune_and_fit_sec', 0) / 60:.1f} CPU-min (6 cores, incl. CV) | "
                 f"{btime.get('cpu_ms_per_image_lgbm_total', '–')} (1 thread, incl. features) | n/a |")
    if cnn_runs:
        gm = np.mean([r["train_gpu_minutes"] for r in cnn_runs])
        L.append(f"| DenseNet-121 | {cnn_runs[0]['n_params']:,} | {gm:.1f} GPU-min per run (mean of {len(cnn_runs)}; "
                 f"{cnn_runs[0]['gpu']}, 2 runs sharing the GPU) | {timing.get('cnn_cpu_ms_per_image_batch1', '–')} "
                 f"({timing.get('cpu_threads', '–')} threads) | {timing.get('cnn_gpu_ms_per_image_batch64', '–')} (batch 64), "
                 f"{timing.get('cnn_gpu_ms_latency_batch1', '–')} (batch 1) |")
    if os.path.exists(f"{R}/yolo/run.json"):
        y = json.load(open(f"{R}/yolo/run.json"))
        L.append(f"| YOLOv8s (detector, separate task) | {timing.get('yolo_n_params', '–')} | {y['train_gpu_minutes']} GPU-min "
                 f"({y['epochs']} epochs @ {y['imgsz']} px, GPU shared with CNN runs) | {timing.get('yolo_cpu_ms_per_image', '–')} | "
                 f"{y['gpu_ms_per_image_predict']} (batch 64 incl. image loading, GPU shared) |")
    L += ["", "## Notes on hypotheses", "", "[filled in by hand after reading the tables - see report]", ""]
    open(f"{R}/summary.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
