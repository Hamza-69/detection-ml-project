"""Step 6 - segmentation evaluation across configurations (tables + figures in results/seg/).

    python -m acdc.evaluate_seg --cfgs classical_lgbm unet2d unet25d unet2d_noaug unet2d_ce unet2d_notta unet2d_nopp

Uses results/seg/<cfg>_metrics.csv (from predict_seg / seg_baseline) and the saved predictions.
  seg_summary.csv        mean (95% bootstrap CI over patients) Dice / HD95 / ASSD per structure, phase, eval set
  seg_paired_tests.csv   Wilcoxon signed-rank test, reference config vs every other (per-patient mean Dice / HD95)
  clinical_agreement.csv Pearson r, bias, 95% limits of agreement, mean absolute error of clinical indices vs ground truth
  slice_position.csv     Dice per structure for basal / mid / apical thirds of the heart (slice 0 = base in ACDC)
  dice_by_group.csv      mean Dice per diagnostic group
  uncertainty.csv        how well volume-level uncertainty flags poor segmentations (Spearman, AUROC, Dice after
                         discarding the most uncertain 10/20/30%)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from acdc.common import CLASSES, GROUPS, RESULTS, WORK, dice, load_phase
from acdc.features import feature_table

OUT = RESULTS / "seg"
STRUCT = list(CLASSES.values())
INDICES = ["LV_EDV", "LV_ESV", "LV_EF", "RV_EDV", "RV_ESV", "RV_EF", "MYO_mass"]
UNITS = {"LV_EDV": "mL", "LV_ESV": "mL", "LV_EF": "%", "RV_EDV": "mL", "RV_ESV": "mL", "RV_EF": "%", "MYO_mass": "g"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def summary(dfs):
    rows = []
    for cfg, df in dfs.items():
        for ev, de in (("cv", df[df.split == "train"]), ("test", df[df.split == "test"])):
            for ph in ("ED", "ES"):
                d = de[de.phase == ph]
                for s in STRUCT:
                    r = {"cfg": cfg, "eval": ev, "phase": ph, "structure": s}
                    for met in ("dice", "hd95", "hd", "assd"):
                        m, lo, hi = boot_ci(d[f"{met}_{s}"])
                        r.update({met: m, f"{met}_lo": lo, f"{met}_hi": hi})
                    r["n_failed"] = int(d[f"hd_{s}"].isna().sum())  # empty prediction for this structure
                    rows.append(r)
    return pd.DataFrame(rows)


def per_patient(df, met):
    cols = [f"{met}_{s}" for s in STRUCT]
    return df.groupby("pid")[cols].mean().mean(axis=1)


def paired_tests(dfs, ref):
    rows = []
    for cfg, df in dfs.items():
        if cfg == ref:
            continue
        for ev in ("train", "test"):
            a, b = dfs[ref][dfs[ref].split == ev], df[df.split == ev]
            for met in ("dice", "hd95"):
                x, y = per_patient(a, met), per_patient(b, met).reindex(per_patient(a, met).index)
                ok = np.isfinite(x) & np.isfinite(y)
                w = stats.wilcoxon(x[ok], y[ok])
                rows.append({"ref": ref, "other": cfg, "eval": "cv" if ev == "train" else "test", "metric": met,
                             "ref_mean": x[ok].mean(), "other_mean": y[ok].mean(), "diff": (x[ok] - y[ok]).mean(),
                             "n": int(ok.sum()), "wilcoxon_p": w.pvalue})
    return pd.DataFrame(rows)


def clinical(cfgs):
    gt = feature_table("gt").set_index("pid")
    rows = []
    for cfg in cfgs:
        pr = feature_table(cfg).set_index("pid").loc[gt.index]
        for ev in ("train", "test"):
            m = gt.split == ev
            for k in INDICES:
                a, b = gt.loc[m, k], pr.loc[m, k]
                d = b - a
                rows.append({"cfg": cfg, "eval": "cv" if ev == "train" else "test", "index": k, "unit": UNITS[k],
                             "pearson_r": stats.pearsonr(a, b)[0], "bias": d.mean(), "loa_lo": d.mean() - 1.96 * d.std(),
                             "loa_hi": d.mean() + 1.96 * d.std(), "mae": d.abs().mean()})
    return pd.DataFrame(rows)


def slice_position(cfgs):
    """Dice per slice, grouped by position within the ground-truth heart (thirds from base to apex)."""
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    rows = []
    for cfg in cfgs:
        for r in meta.itertuples():
            d = np.load(WORK / "preds" / cfg / f"{r.pid}.npz")
            for ph in ("ED", "ES"):
                _, gt, _ = load_phase(r.pid, ph)
                pred = d[f"seg_{ph}"]
                z = np.nonzero((gt > 0).any((1, 2)))[0]
                for zz in z:
                    rel = (zz - z[0]) / max(z[-1] - z[0], 1)
                    pos = "base" if rel < 1 / 3 else ("mid" if rel <= 2 / 3 else "apex")
                    for c, s in CLASSES.items():
                        if (gt[zz] == c).any() or (pred[zz] == c).any():
                            rows.append({"cfg": cfg, "split": r.split, "pid": r.pid, "phase": ph, "position": pos,
                                         "structure": s, "dice": dice(pred[zz] == c, gt[zz] == c)})
    df = pd.DataFrame(rows)
    df["eval"] = df.split.map({"train": "cv", "test": "test"})
    return df.groupby(["cfg", "eval", "position", "structure"]).dice.agg(["mean", "std", "count"]).reset_index()


def uncertainty(dfs):
    rows = []
    for cfg, df in dfs.items():
        if "ent_fg" not in df or df.ent_fg.isna().all():
            continue
        for ev in ("train", "test"):
            d = df[df.split == ev].copy()
            d["mdice"] = d[[f"dice_{s}" for s in STRUCT]].mean(axis=1)
            d["fail"] = (d[[f"dice_{s}" for s in STRUCT]].min(axis=1) < 0.8).astype(int)  # any structure below 0.80
            for score in ("ent_fg", "disagree"):
                if score not in d or d[score].isna().all():
                    continue
                r = {"cfg": cfg, "eval": "cv" if ev == "train" else "test", "score": score, "n_volumes": len(d),
                     "n_fail": int(d.fail.sum()), "spearman_vs_dice": stats.spearmanr(d[score], d.mdice)[0],
                     "auroc_fail": roc_auc_score(d.fail, d[score]) if 0 < d.fail.sum() < len(d) else np.nan,
                     "mean_dice_all": d.mdice.mean()}
                for frac in (0.1, 0.2, 0.3):
                    keep = d[score] <= d[score].quantile(1 - frac)
                    r[f"mean_dice_drop{int(frac * 100)}"] = d.mdice[keep].mean()
                    r[f"fails_drop{int(frac * 100)}"] = int(d.fail[keep].sum())
                rows.append(r)
    return pd.DataFrame(rows)


def figures(dfs, clin, ref):
    # 1) Dice box plots, test set
    cfgs = list(dfs)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a, s in zip(ax, STRUCT):
        data = [dfs[c][dfs[c].split == "test"][f"dice_{s}"].values for c in cfgs]
        a.boxplot(data, showfliers=True)
        a.set_xticks(range(1, len(cfgs) + 1), cfgs, rotation=35, ha="right")
        a.set_title(f"{s} Dice (test, ED+ES volumes)")
        a.grid(alpha=0.3)
    ax[0].set_ylim(0, 1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig_dice_box.png", dpi=130); plt.close(fig)

    # 2) Bland-Altman of the reference config on the test set + CV
    gt = feature_table("gt").set_index("pid")
    pr = feature_table(ref).set_index("pid").loc[gt.index]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
    for a, k in zip(ax.ravel(), INDICES):
        for ev, col in (("train", "tab:blue"), ("test", "tab:red")):
            m = gt.split == ev
            mean, diff = (gt.loc[m, k] + pr.loc[m, k]) / 2, pr.loc[m, k] - gt.loc[m, k]
            a.scatter(mean, diff, s=12, alpha=0.7, c=col, label="CV (out-of-fold)" if ev == "train" else "test")
        d = pr[k] - gt[k]
        for v, ls in ((d.mean(), "-"), (d.mean() + 1.96 * d.std(), "--"), (d.mean() - 1.96 * d.std(), "--")):
            a.axhline(v, color="k", ls=ls, lw=0.8)
        a.set_title(f"{k} ({UNITS[k]})"); a.set_xlabel("mean of GT and predicted"); a.set_ylabel("predicted - GT")
    ax[0, 0].legend(fontsize=8)
    ax.ravel()[-1].axis("off")
    fig.suptitle(f"Bland-Altman: clinical indices from {ref} vs ground-truth segmentation (all 150 patients)")
    fig.tight_layout(); fig.savefig(OUT / "fig_bland_altman.png", dpi=120); plt.close(fig)


def example_figure(ref):
    """Test patients with the best, median and worst mean Dice for the reference config: mid slice ED, GT vs prediction."""
    df = pd.read_csv(OUT / f"{ref}_metrics.csv")
    d = df[(df.split == "test") & (df.phase == "ED")].copy()
    d["mdice"] = d[[f"dice_{s}" for s in STRUCT]].mean(axis=1)
    d = d.sort_values("mdice")
    picks = [d.iloc[-1], d.iloc[len(d) // 2], d.iloc[1], d.iloc[0]]
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), (0.2, 0.5, 1, 0.5), (0.1, 0.8, 0.3, 0.5), (1, 0.3, 0.2, 0.5)])
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for j, r in enumerate(picks):
        img, gt, _ = load_phase(r.pid, "ED")
        pred = np.load(WORK / "preds" / ref / f"{r.pid}.npz")["seg_ED"]
        z = int(np.argmax((gt > 0).sum((1, 2))))  # largest heart slice for the good cases
        if j >= 2:  # worst slice for the poor cases
            ds = [dice(pred[k] > 0, gt[k] > 0) if (gt[k] > 0).any() else 1 for k in range(len(gt))]
            z = int(np.argmin(ds))
        for i, (lab, t) in enumerate(((gt, "ground truth"), (pred, ref))):
            ax[i, j].imshow(img[z].T, cmap="gray")
            ax[i, j].imshow(lab[z].T, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax[i, j].set_title(f"{r.pid} ({r.group}) z={z} - {t}" + (f"\nmean Dice {r.mdice:.3f}" if i == 1 else ""), fontsize=9)
            ax[i, j].axis("off")
    fig.suptitle("Test set ED: best, median, second-worst and worst patient (mean 3D Dice)")
    fig.tight_layout(); fig.savefig(OUT / "fig_examples_best_worst.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--ref", default="unet2d")
    args = ap.parse_args()
    cfgs = [c for c in args.cfgs if (OUT / f"{c}_metrics.csv").exists()]
    print("configs found:", cfgs)
    dfs = {c: pd.read_csv(OUT / f"{c}_metrics.csv") for c in cfgs}
    s = summary(dfs)
    s.to_csv(OUT / "seg_summary.csv", index=False)
    # paired tests against the main model, and against the main model without TTA (fair reference for models that
    # are predicted without TTA, e.g. the no-augmentation ablation)
    refs = [r for r in (args.ref, f"{args.ref}_notta") if r in dfs]
    if refs:
        pd.concat([paired_tests(dfs, r) for r in refs]).to_csv(OUT / "seg_paired_tests.csv", index=False)
    clin = clinical(cfgs)
    clin.to_csv(OUT / "clinical_agreement.csv", index=False)
    slice_position(cfgs).to_csv(OUT / "slice_position.csv", index=False)
    by_group = []
    for c, df in dfs.items():
        g = df.groupby(["split", "group"])[[f"dice_{x}" for x in STRUCT]].mean().reset_index()
        g.insert(0, "cfg", c)
        by_group.append(g)
    pd.concat(by_group).to_csv(OUT / "dice_by_group.csv", index=False)
    uncertainty(dfs).to_csv(OUT / "uncertainty.csv", index=False)
    if args.ref in dfs:
        figures(dfs, clin, args.ref)
        example_figure(args.ref)
    # compact console table (test set)
    t = s[s["eval"] == "test"].pivot_table(index="cfg", columns=["phase", "structure"], values="dice").round(3)
    print(t.to_string())
    (OUT / "seg_overview.json").write_text(json.dumps({c: {
        "test_mean_dice": float(s[(s.cfg == c) & (s["eval"] == "test")].dice.mean()),
        "cv_mean_dice": float(s[(s.cfg == c) & (s["eval"] == "cv")].dice.mean()),
        "test_mean_hd95": float(s[(s.cfg == c) & (s["eval"] == "test")].hd95.mean())} for c in cfgs}, indent=2))


if __name__ == "__main__":
    main()
