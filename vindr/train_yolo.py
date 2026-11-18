"""Step 4b - YOLOv8s on the majority-fused boxes, then mAP@0.4 on the internal test split.

  uv run python train_yolo.py                 # train (30 epochs) + evaluate
  uv run python train_yolo.py --eval_only     # evaluate an existing best.pt

Training uses ultralytics defaults except: no horizontal flip (left/right anatomy matters),
images cached in RAM. Evaluation is our own code: VOC-style all-point AP at IoU >= 0.4 per class
(the VinBigData challenge threshold), against fused boxes supported by >= 2 readers.
Writes results/yolo/map40_test.csv and results/yolo/preds_test.csv.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from prepare import FINDINGS

DATA = "/workspace/data"
OUT = "/workspace/results/yolo"
IOU_THR = 0.4


def box_iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def average_precision(preds, gts):
    """preds: DataFrame(image_id, score, x1..y2) for one class; gts: dict image_id -> [M,4] array.
    Greedy matching by descending score; AP = area under the interpolated precision-recall curve."""
    n_gt = sum(len(g) for g in gts.values())
    if n_gt == 0:
        return np.nan
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = []
    for p in preds.sort_values("score", ascending=False).itertuples():
        g = gts.get(p.image_id)
        hit = False
        if g is not None and len(g):
            ious = box_iou(np.array([p.x1, p.y1, p.x2, p.y2]), g)
            j = int(np.argmax(ious))
            if ious[j] >= IOU_THR and not used[p.image_id][j]:
                used[p.image_id][j] = True
                hit = True
        tp.append(hit)
    tp = np.array(tp, float)
    if len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    # precision envelope, then integrate over recall
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(weights, imgsz):
    from ultralytics import YOLO
    model = YOLO(weights)
    splits = pd.read_csv(f"{DATA}/splits.csv")
    test_ids = splits[splits.split == "test"].image_id.tolist()
    paths = [f"{DATA}/yolo/images/test/{i}.png" for i in test_ids]
    rows = []
    t0 = time.time()
    for k in range(0, len(paths), 64):
        for img_id, r in zip(test_ids[k:k + 64], model.predict(paths[k:k + 64], imgsz=imgsz, conf=0.001, iou=0.6,
                                                               max_det=100, verbose=False, half=True)):
            xyxy = r.boxes.xyxyn.cpu().numpy()
            for (x1, y1, x2, y2), s, c in zip(xyxy, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                rows.append((img_id, int(c), float(s), x1, y1, x2, y2))
    ms = 1000 * (time.time() - t0) / len(paths)
    P = pd.DataFrame(rows, columns=["image_id", "class_id", "score", "x1", "y1", "x2", "y2"])
    P.to_csv(f"{OUT}/preds_test.csv", index=False, float_format="%.5f")

    gt = pd.read_csv(f"{DATA}/boxes_fused.csv")
    gt = gt[(gt.split == "test") & (gt.support >= 2)]
    res = []
    for c, name in enumerate(FINDINGS):
        g = {i: d[["x1", "y1", "x2", "y2"]].values for i, d in gt[gt.class_id == c].groupby("image_id")}
        res.append({"class": name, "n_gt_boxes": int(sum(len(v) for v in g.values())),
                    "AP@0.4": average_precision(P[P.class_id == c], g)})
    res = pd.DataFrame(res)
    res.loc[len(res)] = {"class": "mean (mAP@0.4)", "n_gt_boxes": int(res.n_gt_boxes.sum()), "AP@0.4": res["AP@0.4"].mean()}
    res["AP@0.4"] = res["AP@0.4"].round(4)
    res.to_csv(f"{OUT}/map40_test.csv", index=False)
    print(res.to_string(index=False))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    weights = f"{OUT}/yolov8s_majority/weights/best.pt"
    info = {}
    if not args.eval_only:
        from ultralytics import YOLO
        t0 = time.time()
        model = YOLO("yolov8s.pt")
        model.train(data=f"{DATA}/yolo/data.yaml", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                    workers=args.workers, fliplr=0.0, cache="ram", seed=42, project=OUT, name="yolov8s_majority",
                    exist_ok=True, plots=True, verbose=False)
        info["train_gpu_minutes"] = round((time.time() - t0) / 60, 1)
    info["gpu_ms_per_image_predict"] = round(evaluate(weights, args.imgsz), 2)
    info.update({"epochs": args.epochs, "imgsz": args.imgsz, "model": "yolov8s"})
    json.dump(info, open(f"{OUT}/run.json", "w"), indent=2)


if __name__ == "__main__":
    main()
