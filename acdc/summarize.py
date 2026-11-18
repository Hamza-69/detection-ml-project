"""Step 11 - collect all results into results/summary.md (every number is read from a file in results/).

    python -m acdc.summarize
"""
import json

import pandas as pd

from acdc.common import RESULTS

SEG_NAMES = {"classical_lgbm": "Classical (LightGBM pixel classifier)", "unet2d": "U-Net 2D (main)",
             "unet25d": "U-Net 2.5D (3 slices)", "unet2d_noaug": "U-Net 2D, no augmentation",
             "unet2d_ce": "U-Net 2D, CE loss only", "unet2d_notta": "U-Net 2D, no TTA", "unet2d_nopp": "U-Net 2D, no post-processing"}


def f(x, d=3):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def md(df):
    return df.to_markdown(index=False)


def seg_section():
    out = ["## 1. Segmentation\n"]
    s = pd.read_csv(RESULTS / "seg" / "seg_summary.csv")
    for ev, title in (("test", "Test set (50 patients, 5-fold ensemble)"), ("cv", "Cross-validation (100 training patients, out-of-fold)")):
        rows = []
        for cfg in [c for c in SEG_NAMES if c in s.cfg.unique()]:
            r = {"model": SEG_NAMES[cfg]}
            for ph in ("ED", "ES"):
                for st in ("LV", "RV", "MYO"):
                    d = s[(s.cfg == cfg) & (s["eval"] == ev) & (s.phase == ph) & (s.structure == st)].iloc[0]
                    r[f"{st} {ph}"] = f"{f(d.dice)} / {f(d.hd95, 1)}"
            d = s[(s.cfg == cfg) & (s["eval"] == ev)]
            r["mean Dice"] = f(d.dice.mean())
            rows.append(r)
        out += [f"### {title}: Dice / HD95 (mm)\n", md(pd.DataFrame(rows)), ""]
    p = RESULTS / "seg" / "seg_paired_tests.csv"
    if p.exists():
        t = pd.read_csv(p)
        t = t.assign(**{c: t[c].map(f) for c in ("ref_mean", "other_mean", "diff")}, wilcoxon_p=t.wilcoxon_p.map(lambda x: f"{x:.2g}"))
        out += ["### Paired Wilcoxon tests (per-patient mean over structures and phases)\n", md(t), ""]
    c = pd.read_csv(RESULTS / "seg" / "clinical_agreement.csv")
    c = c[c.cfg.isin(["unet2d", "classical_lgbm", "unet25d"])]
    c = c.assign(**{k: c[k].map(lambda x: f(x, 2)) for k in ("pearson_r", "bias", "loa_lo", "loa_hi", "mae")})
    out += ["### Clinical indices vs expert segmentation (Pearson r, bias, 95% limits of agreement, MAE)\n", md(c), ""]
    sp = pd.read_csv(RESULTS / "seg" / "slice_position.csv")
    sp = sp[sp.cfg == "unet2d"].pivot_table(index=["eval", "structure"], columns="position", values="mean").reset_index()
    out += ["### U-Net 2D: slice-wise Dice by position (base / mid / apex thirds)\n", md(sp.round(3)), ""]
    g = pd.read_csv(RESULTS / "seg" / "dice_by_group.csv")
    out += ["### Dice by diagnostic group\n", md(g.round(3)), ""]
    u = RESULTS / "seg" / "uncertainty.csv"
    if u.exists():
        out += ["### Uncertainty as a failure detector (failure = any structure Dice < 0.80)\n", md(pd.read_csv(u).round(3)), ""]
    out += ["Figures: `seg/fig_dice_box.png`, `seg/fig_bland_altman.png`, `seg/fig_examples_best_worst.png`, `fig_examples.png`\n"]
    return out


def cls_section():
    out = ["## 2. Diagnosis (5 classes)\n"]
    p = pd.read_csv(RESULTS / "cls" / "pipelines.csv")
    cols = ["pipeline", "eval", "acc", "acc_lo", "acc_hi", "macro_f1", "nll", "brier", "ece"]
    p = p[[c for c in cols + ["acc_sd", "nll_sd"] if c in p.columns]].copy()
    for c in p.columns[2:]:
        p[c] = p[c].map(f)
    for ev in ("test", "cv"):
        out += [f"### {'Test (50 patients)' if ev == 'test' else 'Nested CV, out-of-fold (100 training patients)'}\n",
                md(p[p["eval"] == ev]), ""]
    for name, title in (("mcnemar.csv", "McNemar tests vs the main automatic pipeline"), ("cnn_seeds.csv", "End-to-end CNN per seed (with temperature scaling fitted on out-of-fold predictions)")):
        q = RESULTS / "cls" / name
        if q.exists():
            out += [f"### {title}\n", md(pd.read_csv(q).round(3)), ""]
    g = RESULTS / "cls" / "gradcam_summary.json"
    if g.exists():
        out += ["### Grad-CAM of the end-to-end CNN vs expert heart contours (test)\n", "```json", g.read_text(), "```", ""]
    imp = RESULTS / "cls" / "rf_importance_unet2d.csv"
    if imp.exists():
        out += ["### Most important features (random forest, U-Net features)\n", md(pd.read_csv(imp).head(10).round(3)), ""]
    out += ["Figures: `cls/fig_confusion.png`, `cls/fig_coverage.png`, `cls/fig_gradcam.png`\n"]
    return out


def other_section():
    out = []
    r = RESULTS / "robustness" / "robustness_cls.csv"
    if r.exists():
        d = pd.read_csv(r)
        t = d.pivot_table(index=["degradation", "level"], columns="cfg", values=["mean_dice", "diag_acc"]).round(3)
        t.columns = [f"{a} {b}" for a, b in t.columns]
        out += ["## 3. Robustness to image degradations (test set)\n", t.reset_index().to_markdown(index=False), "",
                "Figures: `robustness/fig_robustness.png`, `robustness/fig_degradations.png`\n"]
    c = RESULTS / "complexity.csv"
    if c.exists():
        out += ["## 4. Complexity\n", md(pd.read_csv(c).round(3)), ""]
    return out


def main():
    data = json.loads((RESULTS / "data_summary.json").read_text())
    lines = ["# Results summary (generated by `python -m acdc.summarize`; all numbers read from files in `results/`)\n",
             f"Data: {data['n_train']} training patients (5-fold CV) and {data['n_test']} test patients; "
             f"slices per volume {data['slices_min_median_max']}, in-plane spacing {data['inplane_spacing_min_median_max']} mm, "
             f"slice thickness {data['slice_thickness_min_median_max']} mm.\n"]
    for sec in (seg_section, cls_section, other_section):
        try:
            lines += sec()
        except FileNotFoundError as e:
            lines += [f"*(section {sec.__name__} not available yet: {e.filename})*\n"]
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("wrote", RESULTS / "summary.md")


if __name__ == "__main__":
    main()
