"""Step 7 - assemble the progress report from report/template.md and the result files.

  uv run python build_report.py      # -> /workspace/report/progress_report.md and .docx

Every number in the report comes from a file in /workspace/results: tables are generated here, and numbers in the
text are written in the template as {{ expression }} placeholders that are evaluated against the result tables
(so nothing is typed by hand). Placeholders use the objects defined in `context()` below.
"""
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

from prepare import FINDINGS, TARGETS

R = "/workspace/results"
OUT = "/workspace/report"
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report", "template.md")
LABELS = ["majority", "soft", "union"]


def f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pm(series, d=3):
    """mean ± SD over seeds (SD omitted for a single seed)."""
    s = series.dropna()
    if len(s) == 0:
        return "–"
    return f"{s.mean():.{d}f}" + (f" ± {s.std():.{d}f}" if len(s) > 1 else "")


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join(out)


def load():
    S = pd.read_csv(f"{R}/tables/all_models.csv").set_index("model")
    D = pd.read_csv(f"{R}/tables/paired_differences.csv")
    L = pd.read_csv(f"{R}/tables/ladder_per_class.csv").set_index("target")
    H4 = json.load(open(f"{R}/tables/h4_gain_vs_frequency.json"))
    Y = pd.read_csv(f"{R}/yolo/map40_test.csv") if os.path.exists(f"{R}/yolo/map40_test.csv") else None
    YR = json.load(open(f"{R}/yolo/run.json")) if os.path.exists(f"{R}/yolo/run.json") else {}
    BT = json.load(open(f"{R}/baseline/timing.json"))
    CT = json.load(open(f"{R}/timing_cnn.json")) if os.path.exists(f"{R}/timing_cnn.json") else {}
    RUNS = {n: json.load(open(f"{R}/{n}/run.json")) for n in S.index if n.startswith("cnn_")}
    return S, D, L, H4, Y, YR, BT, CT, RUNS


def cnn(S, variant, labels=None):
    m = (S.kind == "cnn") & (S.variant == variant)
    if labels:
        m &= S.labels == labels
    return S[m]


# ------------------------------------------------------------------ tables
def table_ladder(S, L):
    rows = []
    for t in TARGETS:
        r = L.loc[t]
        rows.append([t, "–" if np.isnan(r.train_pos) else int(r.train_pos), f(r.baseline_logreg), f(r.baseline_lgbm),
                     f(r.cnn_majority), f"{r.gain_vs_best_baseline:+.3f}",
                     f(r.get("baseline_logreg_within_abn")), f(r.get("baseline_lgbm_within_abn")), f(r.get("cnn_majority_within_abn"))])
    F = L.loc[FINDINGS]
    rows.append(["**Macro (14 findings)**", "", f(F.baseline_logreg.mean()), f(F.baseline_lgbm.mean()), f(F.cnn_majority.mean()),
                 f"{F.gain_vs_best_baseline.mean():+.3f}", f(F.baseline_logreg_within_abn.mean()),
                 f(F.baseline_lgbm_within_abn.mean()), f(F.cnn_majority_within_abn.mean())])
    return md(pd.DataFrame(rows, columns=["Finding", "Train pos.", "LogReg", "LightGBM", "CNN (maj.)", "Gain",
                                          "LogReg (abn. only)", "LightGBM (abn. only)", "CNN (abn. only)"]))


def table_ci(S):
    rows = []
    for name in ["baseline_logreg", "baseline_lgbm"] + list(cnn(S, "pw", "majority").index):
        r = S.loc[name]
        rows.append([name.replace("baseline_", "").replace("cnn_", "CNN "), f"{r.macro_auroc:.3f} [{r.macro_auroc_lo:.3f}, {r.macro_auroc_hi:.3f}]",
                     f(r.macro_auroc_within_abnormal), f"{r.abnormal_auroc:.3f} [{r.abnormal_auroc_lo:.3f}, {r.abnormal_auroc_hi:.3f}]"])
    return md(pd.DataFrame(rows, columns=["Model", "Macro-AUROC [95% CI]", "Macro-AUROC, abnormal images only", "Abnormal-vs-normal AUROC [95% CI]"]))


def table_labels(S, variant):
    rows = []
    for lab in LABELS:
        g = cnn(S, variant, lab)
        if not len(g):
            continue
        rows.append([lab, len(g), pm(g.macro_auroc), pm(g.macro_auroc_within_abnormal), pm(g.macro_ece), pm(g.macro_ece_T), pm(g.macro_ece_platt),
                     f"{g['T'].min():.2f}–{g['T'].max():.2f}" if len(g) > 1 else f(g['T'].iloc[0], 2),
                     pm(g.macro_brier, 4), pm(g["macro_sens@0.5"]), pm(g["macro_spec@0.5"])])
    return md(pd.DataFrame(rows, columns=["Labels", "Seeds", "Macro-AUROC", "Macro-AUROC (abn. only)", "Macro-ECE",
                                          "Macro-ECE after T", "Macro-ECE after Platt", "T", "Macro-Brier", "Sensitivity @0.5", "Specificity @0.5"]))


def table_diffs(D, variant):
    d = D[D.variant == variant]
    if not len(d):
        return "_(not available yet)_"
    rows = [[f"{r.a.split('_')[1]} − majority", int(r.seed),
             f"{r.d_macro_auroc:+.4f} [{r.d_macro_auroc_lo:+.4f}, {r.d_macro_auroc_hi:+.4f}]",
             f"{r.d_macro_ece:+.4f} [{r.d_macro_ece_lo:+.4f}, {r.d_macro_ece_hi:+.4f}]"] for r in d.itertuples()]
    rows.sort(key=lambda x: (x[0], x[1]))
    return md(pd.DataFrame(rows, columns=["Comparison", "Seed", "Δ macro-AUROC [95% CI]", "Δ macro-ECE [95% CI]"]))


def table_disagreement(S):
    rows = []
    for variant in ["pw", "nopw"]:
        for lab in LABELS:
            g = cnn(S, variant, lab)
            if not len(g):
                continue
            rows.append([f"CNN {lab}" + (" (no pos_weight)" if variant == "nopw" else ""), len(g),
                         *[f(g[f"dis_mean_p_{v}of3"].mean()) for v in range(4)],
                         pm(g.dis_spearman_marked), pm(g.dis_mae_vs_soft_splitvote), pm(g.dis_mae_vs_soft_splitvote_T), pm(g.dis_ece_vs_soft)])
    for name in ["baseline_logreg", "baseline_lgbm"]:
        r = S.loc[name]
        rows.append([name.replace("baseline_", "") + " (baseline)", 1, *[f(r[f"dis_mean_p_{v}of3"]) for v in range(4)],
                     f(r.dis_spearman_marked), f(r.dis_mae_vs_soft_splitvote), f(r.dis_mae_vs_soft_splitvote_T), f(r.dis_ece_vs_soft)])
    return md(pd.DataFrame(rows, columns=["Model", "Seeds", "p̄ @0/3", "p̄ @1/3", "p̄ @2/3", "p̄ @3/3", "Spearman (marked)",
                                          "|p − k/3| split votes", "same after T", "ECE vs soft"]))


def table_yolo(Y):
    if Y is None:
        return "_(YOLO not available)_"
    y = Y.copy()
    y["AP@0.4"] = y["AP@0.4"].map(lambda v: f(v))
    return md(y.rename(columns={"class": "Class", "n_gt_boxes": "Test boxes (≥2 readers)"}))


def table_complexity(S, BT, CT, YR, RUNS):
    cnn_min = [r["train_gpu_minutes"] for r in RUNS.values()]
    rows = [
        ["Logistic regression (15 models)", f"{BT['logreg_n_params']:,}",
         f"{(BT['feature_extraction_sec_total'] + BT['logreg_tune_and_fit_sec']) / 60:.1f} CPU-min (6 cores; features + 5-fold CV + fit)",
         f(BT["cpu_ms_per_image_logreg_total"], 1), "not needed", f(S.loc["baseline_logreg", "macro_auroc"])],
        ["LightGBM (15 models)", f"{BT['lgbm_total_trees']:,} trees",
         f"{(BT['feature_extraction_sec_total'] + BT['lgbm_tune_and_fit_sec']) / 60:.1f} CPU-min (6 cores; features + CV + fit)",
         f(BT["cpu_ms_per_image_lgbm_total"], 1), "not needed", f(S.loc["baseline_lgbm", "macro_auroc"])],
        ["DenseNet-121 (CNN)", f"{next(iter(RUNS.values()))['n_params']:,}",
         f"{np.mean(cnn_min):.1f} GPU-min per run (mean of {len(cnn_min)} runs; 2 runs sharing one GPU)",
         f(CT.get("cnn_cpu_ms_per_image_batch1"), 1) + (f" ({CT['cpu_threads']} threads)" if CT else ""),
         (f"{CT['cnn_gpu_ms_per_image_batch64']} (batch 64) / {CT['cnn_gpu_ms_latency_batch1']} (batch 1)" if CT else "–"),
         pm(cnn(S, "pw", "majority").macro_auroc)],
    ]
    if YR:
        rows.append(["YOLOv8s (detector)", f"{CT['yolo_n_params']:,}" if "yolo_n_params" in CT else "–",
                     f"{YR['train_gpu_minutes']} GPU-min ({YR['epochs']} epochs, GPU shared)",
                     f(CT.get("yolo_cpu_ms_per_image"), 1), f"{YR['gpu_ms_per_image_predict']} (batch 64, incl. loading)", "mAP@0.4 " + f(YR.get("map40", np.nan))])
    return md(pd.DataFrame(rows, columns=["Model", "Parameters", "Training cost", "CPU ms / image", "GPU ms / image", "Macro-AUROC"]))


def data_summary():
    """Counts quoted in sections 2-3, computed from the data files and saved to results/data_summary.json."""
    D = "/workspace/data"
    raw = pd.read_csv(f"{D}/kaggle/train.csv")
    sp = pd.read_csv(f"{D}/splits.csv").split.value_counts().to_dict()
    box = pd.read_csv(f"{D}/boxes_fused.csv")
    maj = box[box.support >= 2]
    ver = pd.read_csv(f"{R}/image_verification.csv")
    st = pd.read_csv(f"{R}/disagreement_stats.csv").set_index("finding")
    fs = st.loc[FINDINGS]
    out = {"n_rows": len(raw), "n_images": raw.image_id.nunique(), "n_readers": raw.rad_id.nunique(),
           "readers_per_image": sorted(raw.groupby("image_id").rad_id.nunique().unique().tolist()),
           "split": sp, "n_verified": len(ver), "n_bit_identical": int((ver.identical_pixels_frac == 1).sum()),
           "max_gray_diff": int(ver.max_abs_diff_gray.max()), "sizes_match": bool(ver.size_match.all()),
           "n_fused": len(box), "n_fused_maj": len(maj), "maj_boxes_by_split": maj.split.value_counts().to_dict(),
           "maj_boxes_atelectasis": int((maj.class_name == "Atelectasis").sum()),
           "n_findings_over75_split": int((fs.pct_split_vote > 75).sum()),
           "unanimous_share": (fs.n_3of3 / fs.union_pos).round(3).to_dict()}
    json.dump(out, open(f"{R}/data_summary.json", "w"), indent=2)
    return out, st


def table_stats(st):
    t = st.reset_index()[["finding", "n_1of3", "n_2of3", "n_3of3", "union_pos", "majority_pos", "pct_split_vote"]]
    t.columns = ["Finding", "1/3", "2/3", "3/3", "Union", "Majority", "% split vote"]
    t["% split vote"] = t["% split vote"].map(lambda v: f"{v:.1f}")
    t = pd.concat([t[t.Finding != "Abnormal"].sort_values("Union", ascending=False), t[t.Finding == "Abnormal"]])
    t["Finding"] = t.Finding.map(lambda x: f"**{x}**" if x == "Abnormal" else x)
    return md(t)


# ------------------------------------------------------------------ assemble
def context():
    S, D, L, H4, Y, YR, BT, CT, RUNS = load()
    if Y is not None:
        YR["map40"] = float(Y[Y["class"].str.startswith("mean")]["AP@0.4"].iloc[0])
    agg = lambda variant, lab, col: cnn(S, variant, lab)[col]
    DS, st = data_summary()
    SC = pd.read_csv(f"{R}/split_counts.csv", index_col=0)
    CV = pd.read_csv(f"{R}/baseline/cv_results.csv")
    return {"DS": DS, "ST": st, "SC": SC, "CV": CV, "T_STATS": table_stats(st),
        "S": S, "D": D, "L": L, "H4": H4, "Y": Y, "YR": YR, "BT": BT, "CT": CT, "RUNS": RUNS, "np": np,
        "f": f, "pm": pm, "agg": agg, "cnn": cnn,
        "T_LADDER": table_ladder(S, L), "T_CI": table_ci(S),
        "T_LABELS_PW": table_labels(S, "pw"), "T_LABELS_NOPW": table_labels(S, "nopw"),
        "T_DIFF_PW": table_diffs(D, "pw"), "T_DIFF_NOPW": table_diffs(D, "nopw"),
        "T_DIS": table_disagreement(S), "T_YOLO": table_yolo(Y), "T_COMPLEX": table_complexity(S, BT, CT, YR, RUNS),
    }


def main():
    ctx = context()
    text = open(TEMPLATE).read()
    text = re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1).strip(), ctx)), text, flags=re.S)
    os.makedirs(OUT, exist_ok=True)
    md_path = f"{OUT}/progress_report.md"
    open(md_path, "w").write(text)
    subprocess.run(["pandoc", md_path, "-o", f"{OUT}/progress_report.docx", f"--resource-path={OUT}"], check=True)
    print("wrote", md_path, "and progress_report.docx")


if __name__ == "__main__":
    main()
