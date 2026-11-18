"""Collect the final weights the demo needs into one folder (used by app.py locally and by the Fly deploy).

  uv run python export_models.py --out deploy/models

Copies: CNN checkpoints for each label type (seed 42, with pos_weight; weights saved in fp16 to halve size),
baseline logistic regression, YOLOv8 weights if trained, and temperatures.json (T per model, from evaluate.py).
No images or training data are copied.
"""
import argparse
import json
import os
import shutil

import pandas as pd
import torch

R = "/workspace/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deploy/models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    temps = {}
    table = pd.read_csv(f"{R}/tables/all_models.csv").set_index("model") if os.path.exists(f"{R}/tables/all_models.csv") else None

    for labels in ["majority", "soft", "union"]:
        src = f"{R}/cnn_{labels}_s{args.seed}/best.pt"
        if not os.path.exists(src):
            continue
        ck = torch.load(src, map_location="cpu")
        sd = {k: v.half() if v.is_floating_point() else v for k, v in ck["state_dict"].items()}
        torch.save({"state_dict": sd, "labels": labels, "seed": args.seed, "targets": ck["targets"]}, f"{args.out}/cnn_{labels}.pt")
        if table is not None and f"cnn_{labels}_s{args.seed}" in table.index:
            temps[f"cnn_{labels}"] = float(table.loc[f"cnn_{labels}_s{args.seed}", "T"])
        print("exported", src)
    if os.path.exists(f"{R}/baseline/logreg.joblib"):
        shutil.copy(f"{R}/baseline/logreg.joblib", f"{args.out}/logreg.joblib")
        if table is not None and "baseline_logreg" in table.index:
            temps["logreg"] = float(table.loc["baseline_logreg", "T"])
        print("exported logreg")
    yolo = f"{R}/yolo/yolov8s_majority/weights/best.pt"
    if os.path.exists(yolo):
        shutil.copy(yolo, f"{args.out}/yolov8s.pt")
        print("exported yolo")
    json.dump(temps, open(f"{args.out}/temperatures.json", "w"), indent=2)
    print("temperatures:", temps)


if __name__ == "__main__":
    main()
