"""Step 4a - build the YOLO detection dataset from the 3 readers' boxes.

For every image and finding:
  1. rescale each reader's boxes from original DICOM pixels to 512x512
     (x * 512 / width, y * 512 / height; the PNGs were resized without keeping aspect ratio),
  2. merge all readers' boxes with weighted boxes fusion (WBF, IoU 0.4),
  3. count how many distinct readers support each fused box (a reader supports it if one of
     their boxes of that class overlaps it with IoU >= 0.4); keep boxes with support >= 2 (majority).
"No finding" is excluded. Images without boxes stay in the dataset as background images.

Writes /workspace/data/yolo/{images,labels}/{train,val,test}/, data.yaml and
data/boxes_fused.csv (all fused boxes with their reader support, for evaluation and the demo).
"""
import os

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

from prepare import FINDINGS

DATA = "/workspace/data"
PNG_DIR = f"{DATA}/png512/vinbigdata/train"
OUT = f"{DATA}/yolo"
IOU = 0.4


def iou(a, b):
    """IoU between box a [4] and boxes b [M, 4] (x1, y1, x2, y2)."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def fuse_image(rows):
    """rows: one image's finding boxes (normalised 0-1). Returns list of (class, x1, y1, x2, y2, support)."""
    out = []
    for cls, g in rows.groupby("class_id"):
        readers = sorted(g.rad_id.unique())
        boxes = [g[g.rad_id == r][["x1", "y1", "x2", "y2"]].values.tolist() for r in readers]
        fused, _, _ = weighted_boxes_fusion(boxes, [[1.0] * len(b) for b in boxes], [[0] * len(b) for b in boxes],
                                            iou_thr=IOU, skip_box_thr=0.0)
        for fb in fused:
            support = sum(int((iou(fb, np.array(b)) >= IOU).any()) for b in boxes)
            out.append((int(cls), *fb.tolist(), support))
    return out


def main():
    df = pd.read_csv(f"{DATA}/kaggle/train.csv")
    meta = pd.read_csv(f"{DATA}/meta.csv").set_index("image_id")
    splits = pd.read_csv(f"{DATA}/splits.csv").set_index("image_id").split

    f = df[df.class_id < 14].copy()
    w = f.image_id.map(meta.width)
    h = f.image_id.map(meta.height)
    f["x1"], f["x2"] = (f.x_min / w).clip(0, 1), (f.x_max / w).clip(0, 1)
    f["y1"], f["y2"] = (f.y_min / h).clip(0, 1), (f.y_max / h).clip(0, 1)
    f = f[(f.x2 > f.x1) & (f.y2 > f.y1)]

    fused = []
    for img, rows in f.groupby("image_id"):
        fused += [(img, *b) for b in fuse_image(rows)]
    fused = pd.DataFrame(fused, columns=["image_id", "class_id", "x1", "y1", "x2", "y2", "support"])
    fused["class_name"] = fused.class_id.map(dict(enumerate(FINDINGS)))
    fused["split"] = fused.image_id.map(splits)
    fused.to_csv(f"{DATA}/boxes_fused.csv", index=False, float_format="%.5f")
    maj = fused[fused.support >= 2]
    print(f"fused boxes: {len(fused)}, with >=2 readers: {len(maj)}")
    print(maj.groupby("class_name").size().to_string())

    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
        ids = splits[splits == split].index
        boxes = maj[maj.split == split].groupby("image_id")
        for img in ids:
            link = f"{OUT}/images/{split}/{img}.png"
            if not os.path.exists(link):
                os.symlink(f"{PNG_DIR}/{img}.png", link)
            lines = []
            if img in boxes.groups:
                for b in boxes.get_group(img).itertuples():
                    lines.append(f"{b.class_id} {(b.x1 + b.x2) / 2:.6f} {(b.y1 + b.y2) / 2:.6f} "
                                 f"{b.x2 - b.x1:.6f} {b.y2 - b.y1:.6f}")
            with open(f"{OUT}/labels/{split}/{img}.txt", "w") as fh:
                fh.write("\n".join(lines))
        print(split, len(ids), "images,", int((maj.split == split).sum()), "boxes")

    with open(f"{OUT}/data.yaml", "w") as fh:
        fh.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        fh.writelines(f"  {i}: {n}\n" for i, n in enumerate(FINDINGS))


if __name__ == "__main__":
    main()
