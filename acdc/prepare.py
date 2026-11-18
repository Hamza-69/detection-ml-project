"""Step 1 - read all 150 ACDC patients once and build everything later steps need.

Writes
  WORK/cache/<pid>.npz        ED and ES images resampled to 1.25 mm in-plane and z-scored (float16), their label maps
                              resampled the same way (uint8), plus the original spacing / shape (used to map back).
  results/acdc_meta.csv       one row per patient: split (train/test), CV fold, group, height, weight, frames, spacing,
                              ground-truth clinical indices (from the original-resolution masks).
  results/data_summary.json   counts used in the report.
  results/fig_examples.png    one mid-ventricular ED slice with its ground truth per diagnostic group.

Splits: the official ACDC split is kept - patients 001-100 = training (5-fold CV, stratified by group, seed 42),
patients 101-150 = held-out test set (10 per group), never used for training, tuning or model selection.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from acdc.common import (GROUPS, RESULTS, WORK, body_surface_area, clinical_indices, load_phase, normalize,
                         read_info, resample_image, resample_labels)


def main():
    (WORK / "cache").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rows, examples = [], {}
    for n in range(1, 151):
        pid = f"patient{n:03d}"
        info = read_info(pid)
        arrays = {}
        for phase in ("ED", "ES"):
            img, gt, sp = load_phase(pid, phase, info)
            img_r = resample_image(normalize(img), sp[:2])
            arrays[f"img_{phase}"] = img_r.astype(np.float16)
            arrays[f"lab_{phase}"] = resample_labels(gt, img_r.shape[1:])
            if phase == "ED":
                gt_ed = gt
            else:
                gt_es = gt
        np.savez_compressed(WORK / "cache" / f"{pid}.npz", spacing=np.array(sp), orig_shape=np.array(img.shape), **arrays)
        row = {"pid": pid, "split": "train" if n <= 100 else "test", **info,
               "sx": sp[0], "sy": sp[1], "sz": sp[2], "n_slices": img.shape[0], "nx": img.shape[1], "ny": img.shape[2],
               "bsa": body_surface_area(info["height"], info["weight"])}
        row.update({f"gt_{k}": v for k, v in clinical_indices(gt_ed, gt_es, sp).items()})
        rows.append(row)
        if n <= 100 and info["group"] not in examples:  # first training patient of each group for the figure
            z = int(np.argmax((arrays["lab_ED"] == 3).sum((1, 2))))  # slice with the largest LV cavity
            examples[info["group"]] = (arrays["img_ED"][z].astype(np.float32), arrays["lab_ED"][z], pid)
        print(pid, info["group"], img.shape, sp, flush=True)

    meta = pd.DataFrame(rows)
    # 5-fold CV on the 100 training patients, stratified by diagnostic group (20 per group -> 4 per group per fold).
    meta["fold"] = -1
    tr = meta.index[meta.split == "train"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (_, va) in enumerate(skf.split(tr, meta.loc[tr, "group"])):
        meta.loc[tr[va], "fold"] = f
    meta.to_csv(RESULTS / "acdc_meta.csv", index=False)

    summary = {
        "n_patients": len(meta), "n_train": int((meta.split == "train").sum()), "n_test": int((meta.split == "test").sum()),
        "per_group": {s: meta[meta.split == s].group.value_counts().to_dict() for s in ("train", "test")},
        "per_fold_group": meta[meta.split == "train"].groupby(["fold", "group"]).size().unstack().to_dict(orient="index"),
        "n_slices_total_per_phase": int(meta.n_slices.sum()),
        "slices_min_median_max": [int(meta.n_slices.min()), float(meta.n_slices.median()), int(meta.n_slices.max())],
        "inplane_spacing_min_median_max": [float(meta.sx.min()), float(meta.sx.median()), float(meta.sx.max())],
        "slice_thickness_min_median_max": [float(meta.sz.min()), float(meta.sz.median()), float(meta.sz.max())],
        "gt_indices_mean_by_group": meta.groupby("group")[[c for c in meta.columns if c.startswith("gt_")]].mean().round(1).to_dict(orient="index"),
    }
    (RESULTS / "data_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 5, figsize=(15, 3.4))
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), (0.2, 0.5, 1, 0.45), (0.1, 0.8, 0.3, 0.45), (1, 0.3, 0.2, 0.45)])
    for a, g in zip(ax, GROUPS):
        im, lab, pid = examples[g]
        a.imshow(im.T, cmap="gray"); a.imshow(lab.T, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        a.set_title(f"{g} ({pid})"); a.axis("off")
    fig.suptitle("ACDC: mid-ventricular ED slice per group (blue RV, green myocardium, red LV)")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_examples.png", dpi=130)
    print(json.dumps({k: summary[k] for k in ("n_train", "n_test", "per_group")}, indent=1))


if __name__ == "__main__":
    main()
