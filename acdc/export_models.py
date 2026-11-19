"""Collect the final models for the demo / deployment into one folder (default deploy/models).

    python -m acdc.export_models --seg_cfg unet2d --out deploy/models

  unet_f{0..4}.pt          segmentation U-Nets (5 folds), weights stored in float16 to halve the size
  cnn_f{0..4}.pt           end-to-end ResNet-18 diagnosis CNNs (seed 0, 5 folds), float16
  feature_clf.joblib       logistic regression on cardiac features, trained on all 100 training patients with the
                           OUT-OF-FOLD predicted features of the same U-Net (the automatic pipeline, as evaluated)
  config.json              in_ch, feature names, uncertainty threshold (90th percentile of the out-of-fold entropy
                           score on the training patients), CNN temperature
  examples/                a few ACDC test patients (ED/ES frames + Info.cfg) for the demo, CC BY-NC-SA 4.0, with citation
"""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from acdc.classify import make_model
from acdc.common import GROUPS, RESULTS, WORK, patient_dir, read_info
from acdc.features import feature_table



def half(sd):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_cfg", default="unet2d")
    ap.add_argument("--cnn_tag", default="resnet18")
    ap.add_argument("--cnn_seed", type=int, default=0)
    ap.add_argument("--out", default="deploy/models")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for f in range(5):
        ck = torch.load(WORK / "runs" / f"{args.seg_cfg}_f{f}" / "model.pt", map_location="cpu", weights_only=False)
        torch.save({"model": half(ck["model"]), "args": ck["args"]}, out / f"unet_f{f}.pt")
        ck = torch.load(WORK / "runs" / f"cls_{args.cnn_tag}_s{args.cnn_seed}_f{f}" / "model.pt", map_location="cpu", weights_only=False)
        torch.save({"model": half(ck["model"])}, out / f"cnn_f{f}.pt")
    in_ch = torch.load(out / "unet_f0.pt", weights_only=False)["args"]["in_ch"]

    ft = feature_table(args.seg_cfg)
    feats = [c for c in ft.columns if c not in ("pid", "split", "fold", "group")]
    tr = ft[ft.split == "train"]
    enc = {g: i for i, g in enumerate(GROUPS)}
    clf = make_model("logreg", len(feats)).fit(tr[feats].values, tr.group.map(enc).values)
    joblib.dump(clf.best_estimator_, out / "feature_clf.joblib")

    # the app predicts without test-time augmentation -> threshold from the no-TTA out-of-fold predictions if available
    m = RESULTS / "seg" / f"{args.seg_cfg}_notta_metrics.csv"
    seg = pd.read_csv(m if m.exists() else RESULTS / "seg" / f"{args.seg_cfg}_metrics.csv")
    thr = float(seg[seg.split == "train"].ent_fg.quantile(0.9))
    temp = None
    seeds = RESULTS / "cls" / "cnn_seeds.csv"
    if seeds.exists():
        s = pd.read_csv(seeds)
        temp = float(s.loc[s.seed == args.cnn_seed, "T"].iloc[0]) if (s.seed == args.cnn_seed).any() else None
    cfg = {"seg_cfg": args.seg_cfg, "in_ch": int(in_ch), "features": feats, "groups": GROUPS,
           "uncertainty_threshold_ent_fg": thr, "cnn_temperature": temp}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    examples = meta[meta.split == "test"].groupby("group").pid.first().loc[GROUPS].tolist()  # one test patient per group
    for pid in examples:
        info = read_info(pid)
        dst = out / "examples" / pid
        dst.mkdir(exist_ok=True)
        for ph in ("ED", "ES"):
            shutil.copy(patient_dir(pid) / f"{pid}_frame{info[ph]:02d}.nii.gz", dst / f"{pid}_{ph}.nii.gz")
        shutil.copy(patient_dir(pid) / "Info.cfg", dst / "Info.cfg")
    (out / "examples" / "CITATION.md").write_text(
        "Example images from the ACDC dataset (test patients), CC BY-NC-SA 4.0, non-commercial research use only.\n\n"
        "O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. \"Deep Learning Techniques for Automatic MRI Cardiac "
        "Multi-structures Segmentation and Diagnosis: Is the Problem Solved?\" IEEE TMI 37(11), 2514-2525, 2018. "
        "doi: 10.1109/TMI.2018.2837502\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"exported to {out} ({total:.0f} MB)", json.dumps(cfg)[:300])


if __name__ == "__main__":
    main()
