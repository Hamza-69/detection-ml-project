"""Step 7 - assemble the progress report from report/template.md and the result files.

  uv run python build_report.py      # -> <report dir>/progress_report.md and .docx (needs pandoc)

Every number in the report comes from a file in the results folder: tables are generated here, and numbers in the
text are written in the template as {{ expression }} placeholders that are evaluated against the result tables
(so nothing is typed by hand). Placeholders use the objects defined in `context()` below.

Folders: on the pod, /workspace/results and /workspace/report. On a laptop (no /workspace), the downloaded copy in
local_results/results and local_results/report. Override with RESULTS_DIR / REPORT_DIR.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
import pandas as pd

from prepare import FINDINGS, TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
ON_POD = os.path.isdir("/workspace/results")
R = os.environ.get("RESULTS_DIR", "/workspace/results" if ON_POD else os.path.join(HERE, "local_results", "results"))
OUT = os.environ.get("REPORT_DIR", "/workspace/report" if ON_POD else os.path.join(HERE, "local_results", "report"))
DATA = "/workspace/data"
TEMPLATE = os.path.join(HERE, "report", "template.md")
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


def table_baseline(S, CV, RUNS):
    """Main-text Table 1: tuned baselines (validation = 5-fold CV on train) next to the CNN (validation = held-out
    val split at the selected epoch), then the internal-test numbers."""
    rows = []
    for name, lab in [("baseline_logreg", "logreg"), ("baseline_lgbm", "lgbm")]:
        r = S.loc[name]
        cv = CV[(CV.model == lab) & CV.target.isin(FINDINGS)].cv_auroc.mean()
        rows.append([{"logreg": "Logistic regression", "lgbm": "LightGBM"}[lab], f(cv),
                     f"{r.macro_auroc:.3f} [{r.macro_auroc_lo:.3f}, {r.macro_auroc_hi:.3f}]",
                     f(r.macro_auroc_within_abnormal), f(r.abnormal_auroc), f(r.macro_ece)])
    g = cnn(S, "pw", "majority")
    val = pd.Series([RUNS[n]["best_val_macro_auroc"] for n in g.index])
    rows.append(["DenseNet-121", pm(val), pm(g.macro_auroc),
                 pm(g.macro_auroc_within_abnormal), pm(g.abnormal_auroc), pm(g.macro_ece)])
    return md(pd.DataFrame(rows, columns=["Model", "Val. macro-AUROC", "Test macro-AUROC",
                                          "Test, abnormal images only", "Abnormal vs normal", "Macro-ECE"]))


def table_labels_main(S):
    """Main-text Table 2: the six CNN configurations, mean ± SD over seeds, 3 decimals."""
    rows = []
    for variant, vname in [("pw", "yes"), ("nopw", "no")]:
        for lab in LABELS:
            g = cnn(S, variant, lab)
            if len(g):
                rows.append([lab, vname, pm(g.macro_auroc), pm(g.macro_ece), pm(g.macro_ece_platt), pm(g.macro_brier),
                             pm(g["macro_sens@0.5"], 2), pm(g["macro_spec@0.5"])])
    return md(pd.DataFrame(rows, columns=["Labels", "pos_weight", "AUROC", "ECE", "ECE after Platt",
                                          "Brier", "Sens. @0.5", "Spec. @0.5"]))


def table_ci(S):
    rows = []
    for name in ["baseline_logreg", "baseline_lgbm"] + list(cnn(S, "pw", "majority").index):
        r = S.loc[name]
        rows.append([{"baseline_logreg": "LogReg", "baseline_lgbm": "LightGBM"}.get(name, name.replace("cnn_", "CNN ")), f"{r.macro_auroc:.3f} [{r.macro_auroc_lo:.3f}, {r.macro_auroc_hi:.3f}]",
                     f(r.macro_auroc_within_abnormal), f"{r.abnormal_auroc:.3f} [{r.abnormal_auroc_lo:.3f}, {r.abnormal_auroc_hi:.3f}]"])
    return md(pd.DataFrame(rows, columns=["Model", "Macro-AUROC [95% CI]", "Macro-AUROC, abnormal images only", "Abnormal-vs-normal AUROC [95% CI]"]))


def table_labels(S, variant):
    rows = []
    for lab in LABELS:
        g = cnn(S, variant, lab)
        if not len(g):
            continue
        rows.append([lab, len(g), pm(g.macro_auroc), pm(g.macro_auroc_within_abnormal), pm(g.macro_ece, 4), pm(g.macro_ece_T, 4), pm(g.macro_ece_platt, 4),
                     f"{g['T'].min():.2f}–{g['T'].max():.2f}" if len(g) > 1 else f(g['T'].iloc[0], 2),
                     pm(g.macro_brier, 4), pm(g["macro_sens@0.5"]), pm(g["macro_spec@0.5"])])
    return md(pd.DataFrame(rows, columns=["Labels", "Seeds", "Macro-AUROC", "Macro-AUROC (abn. only)", "Macro-ECE",
                                          "Macro-ECE after T", "Macro-ECE after Platt", "T", "Macro-Brier", "Sensitivity @0.5", "Specificity @0.5"]))


def table_labels_all(S):
    """Compact main-text table: both pos_weight settings, mean ± SD over seeds."""
    rows = []
    for variant, vname in [("pw", "yes"), ("nopw", "no")]:
        for lab in LABELS:
            g = cnn(S, variant, lab)
            if not len(g):
                continue
            rows.append([lab, vname, pm(g.macro_auroc), pm(g.macro_auroc_within_abnormal), pm(g.macro_ece, 4),
                         pm(g.macro_ece_T, 4), pm(g.macro_ece_platt, 4), pm(g.macro_brier, 4),
                         pm(g["macro_sens@0.5"], 2), pm(g["macro_spec@0.5"], 3)])
    return md(pd.DataFrame(rows, columns=["Labels", "pos_weight", "Macro-AUROC", "AUROC abn. only", "ECE", "ECE after T",
                                          "ECE after Platt", "Brier", "Sens. @0.5", "Spec. @0.5"]))


def table_dis_compact(S):
    rows = []
    for variant, vname in [("pw", "yes"), ("nopw", "no")]:
        for lab in LABELS:
            g = cnn(S, variant, lab)
            if not len(g):
                continue
            rows.append([lab, vname, *[f(g[f"dis_mean_p_{v}of3"].mean()) for v in (1, 2, 3)],
                         pm(g.dis_mae_vs_soft_splitvote), pm(g.dis_ece_vs_soft, 4), pm(g.dis_spearman_marked)])
    return md(pd.DataFrame(rows, columns=["Labels", "pos_weight", "p̄ @1/3", "p̄ @2/3", "p̄ @3/3",
                                          "mean abs(p − k/3), split votes", "ECE vs reader fraction", "Spearman (marked)"]))


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
                         pm(g.dis_spearman_marked), pm(g.dis_mae_vs_soft_splitvote), pm(g.dis_mae_vs_soft_splitvote_T), pm(g.dis_ece_vs_soft, 4)])
    for name in ["baseline_logreg", "baseline_lgbm"]:
        r = S.loc[name]
        rows.append([{"baseline_logreg": "LogReg", "baseline_lgbm": "LightGBM"}[name] + " (baseline)", 1, *[f(r[f"dis_mean_p_{v}of3"]) for v in range(4)],
                     f(r.dis_spearman_marked), f(r.dis_mae_vs_soft_splitvote), f(r.dis_mae_vs_soft_splitvote_T), f(r.dis_ece_vs_soft, 4)])
    return md(pd.DataFrame(rows, columns=["Model", "Seeds", "p̄ @0/3", "p̄ @1/3", "p̄ @2/3", "p̄ @3/3", "Spearman (marked)",
                                          "mean abs(p − k/3), split votes", "same after T", "ECE vs soft"]))


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
                     f(CT.get("yolo_cpu_ms_per_image"), 1), f"{YR['gpu_ms_per_image_predict']} (batch 64, incl. loading, GPU shared)", "mAP@0.4 " + f(YR.get("map40", np.nan))])
    return md(pd.DataFrame(rows, columns=["Model", "Parameters", "Training cost", "CPU ms / image", "GPU ms / image", "Macro-AUROC"]))


def data_summary():
    """Counts quoted in sections 1-2, computed from the data files and saved to results/data_summary.json.
    Without the data folder (laptop), the saved data_summary.json from the pod run is used instead."""
    D = DATA
    st = pd.read_csv(f"{R}/disagreement_stats.csv").set_index("finding")
    if not os.path.exists(f"{D}/kaggle/train.csv"):
        return json.load(open(f"{R}/data_summary.json")), st
    raw = pd.read_csv(f"{D}/kaggle/train.csv")
    sp = pd.read_csv(f"{D}/splits.csv").split.value_counts().to_dict()
    box = pd.read_csv(f"{D}/boxes_fused.csv")
    maj = box[box.support >= 2]
    ver = pd.read_csv(f"{R}/image_verification.csv")
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
        "f": f, "pm": pm, "agg": agg, "cnn": cnn, "FINDINGS": FINDINGS,
        "T_LADDER": table_ladder(S, L), "T_CI": table_ci(S),
        "T_LABELS_PW": table_labels(S, "pw"), "T_LABELS_NOPW": table_labels(S, "nopw"),
        "T_DIFF_PW": table_diffs(D, "pw"), "T_DIFF_NOPW": table_diffs(D, "nopw"),
        "T_DIS": table_disagreement(S), "T_LABELS_ALL": table_labels_all(S), "T_DIS_COMPACT": table_dis_compact(S), "T_YOLO": table_yolo(Y), "T_COMPLEX": table_complexity(S, BT, CT, YR, RUNS),
        "T_BASELINE": table_baseline(S, CV, RUNS), "T_LABELS_MAIN": table_labels_main(S),
    }


def _restyle(xml, style_id, old, new):
    """Replace `old` with `new` inside one <w:style> block of styles.xml (fails loudly if it isn't there)."""
    m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % style_id, xml, re.S)
    assert m and old in m.group(0), (style_id, old)
    return xml[:m.start()] + m.group(0).replace(old, new) + xml[m.end():]


def reference_docx(path):
    """Word styles for the report: pandoc's default reference.docx with 11 pt text, 9 pt tables, 10 pt captions,
    tighter spacing, smaller headings, and US Letter pages with 2 cm margins (the report has a 2-page limit)."""
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], check=True, capture_output=True).stdout
    src = path + ".default"
    open(src, "wb").write(default)
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist()}
    s = files["word/styles.xml"].decode()
    s = re.sub(r"(<w:rPrDefault>.*?)w:val=\"24\"(.*?)w:val=\"24\"", r'\1w:val="22"\2w:val="22"', s, flags=re.S)  # 11 pt
    s = _restyle(s, "BodyText", 'w:before="180" w:after="180"', 'w:before="0" w:after="100"')
    s = _restyle(s, "Compact", 'w:before="36" w:after="36"', 'w:before="0" w:after="20"')
    # font sizes in half-points: title 28 -> 16 pt, heading 1 20 -> 13 pt, heading 2 16 -> 11 pt
    for sid, old, new in [("Title", "56", "32"), ("Heading1", "40", "26"), ("Heading2", "32", "22")]:
        s = re.sub(r'(w:styleId="%s".*?)<w:sz w:val="%s" />\s*<w:szCs w:val="%s" />' % (sid, old, old),
                   r'\1<w:sz w:val="%s" /><w:szCs w:val="%s" />' % (new, new), s, count=1, flags=re.S)
    s = _restyle(s, "Heading1", 'w:before="360" w:after="80"', 'w:before="140" w:after="40"')
    s = _restyle(s, "Caption", "<w:i />", '<w:i /><w:sz w:val="20" /><w:szCs w:val="20" />')
    s = _restyle(s, "Table", "<w:tblPr>", '<w:rPr><w:sz w:val="18" /><w:szCs w:val="18" /></w:rPr><w:tblPr>')
    files["word/styles.xml"] = s.encode()
    # theme fonts Aptos -> Calibri (narrower, so more text fits per page)
    files["word/theme/theme1.xml"] = files["word/theme/theme1.xml"].decode() \
        .replace('typeface="Aptos Display"', 'typeface="Calibri"').replace('typeface="Aptos"', 'typeface="Calibri"').encode()
    d = files["word/document.xml"].decode()
    page = ('<w:pgSz w:w="12240" w:h="15840" /><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
            'w:header="567" w:footer="567" w:gutter="0" />')
    d = re.sub(r"(<w:sectPr>.*?</w:footnotePr>)", r"\1" + page, d, count=1, flags=re.S)
    files["word/document.xml"] = d.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    os.remove(src)


def main():
    ctx = context()
    text = open(TEMPLATE).read()
    text = re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1).strip(), ctx)), text, flags=re.S)
    os.makedirs(OUT, exist_ok=True)
    md_path = f"{OUT}/progress_report.md"
    open(md_path, "w").write(text)
    ref = f"{OUT}/reference.docx"
    reference_docx(ref)
    subprocess.run(["pandoc", md_path, "-o", f"{OUT}/progress_report.docx", f"--resource-path={OUT}",
                    f"--reference-doc={ref}"], check=True)
    print("wrote", md_path, "and progress_report.docx")


if __name__ == "__main__":
    main()
