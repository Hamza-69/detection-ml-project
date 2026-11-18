"""Step 1 - turn the VinBigData train.csv (one row per box per reader) into
image-level labels, fixed splits, disagreement statistics and an image cache.

Outputs (all row-aligned with data/image_ids.txt and the memmap):
  data/votes.csv            n_votes in {0,1,2,3} per finding (+ abnormal)
  data/labels_union.csv     1 if >=1 of 3 readers marked it
  data/labels_majority.csv  1 if >=2 of 3 readers marked it
  data/labels_soft.csv      n_votes / 3
  data/splits.csv           train / val / test / disagree (seed 42, never changed)
  data/meta.csv             original width/height (for rescaling boxes)
  data/images_512.npy       uint8 memmap [N, 512, 512] of all training images
  results/disagreement_stats.csv, results/fig_disagreement.png, results/split_counts.csv

Usage:
  uv run python prepare.py --labels_csv /workspace/data/kaggle/train.csv \
      --img_dir /workspace/data/png512/train --meta_csv <csv with image_id,width,height>
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The 14 findings in class_id order (class_id 14 is "No finding").
FINDINGS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
TARGETS = FINDINGS + ["Abnormal"]  # 15 model outputs
IMG_SIZE = 512
SEED = 42


def compute_votes(df):
    """Return (image_ids, votes[N, 15]) where votes[i, k] = number of readers
    (0-3) who drew at least one box of finding k on image i. The last column
    ("Abnormal") counts readers who marked any finding at all."""
    image_ids = np.array(sorted(df.image_id.unique()))
    row = {img: i for i, img in enumerate(image_ids)}

    # Every reader of an image appears at least once (a "No finding" row if they saw nothing).
    readers_per_image = df.groupby("image_id").rad_id.nunique()
    print("readers per image:", readers_per_image.value_counts().to_dict())

    findings = df[df.class_id < 14]
    # one row per (image, reader, finding) -> that reader marked that finding
    marks = findings[["image_id", "rad_id", "class_id"]].drop_duplicates()
    votes = np.zeros((len(image_ids), 15), dtype=np.int64)
    for (img, cls), n in marks.groupby(["image_id", "class_id"]).size().items():
        votes[row[img], cls] = n
    # abnormal: number of distinct readers who marked any finding
    for img, n in findings.groupby("image_id").rad_id.nunique().items():
        votes[row[img], 14] = n
    return image_ids, votes


def make_splits(image_ids, votes):
    """Stratified split on finding presence: each image gets a key = its rarest
    finding marked by any reader (or 'none'); keys are kept balanced across splits.
    test 15% | disagreement 15% | val 10% | train 60%."""
    union = votes[:, :14] >= 1
    class_counts = union.sum(0)
    keys = []
    for u in union:
        present = np.where(u)[0]
        keys.append("none" if len(present) == 0 else FINDINGS[present[np.argmin(class_counts[present])]])
    keys = pd.Series(keys)
    small = keys.map(keys.value_counts()) < 30  # too few to stratify on their own
    keys[small] = "rare"
    keys = keys.values

    idx = np.arange(len(image_ids))
    rest, test = train_test_split(idx, test_size=0.15, stratify=keys, random_state=SEED)
    rest, disagree = train_test_split(rest, test_size=0.15 / 0.85, stratify=keys[rest], random_state=SEED)
    train, val = train_test_split(rest, test_size=0.10 / 0.70, stratify=keys[rest], random_state=SEED)

    split = np.empty(len(image_ids), dtype=object)
    split[train], split[val], split[test], split[disagree] = "train", "val", "test", "disagree"
    return pd.DataFrame({"image_id": image_ids, "split": split, "strat_key": keys})


def disagreement_stats(votes, df):
    """Per finding: how many images were marked by exactly 1, 2 or 3 readers."""
    boxes = df[df.class_id < 14].class_id.value_counts()
    rows = []
    for k, name in enumerate(TARGETS):
        v = votes[:, k]
        n1, n2, n3 = int((v == 1).sum()), int((v == 2).sum()), int((v == 3).sum())
        union = n1 + n2 + n3
        rows.append({
            "finding": name, "n_1of3": n1, "n_2of3": n2, "n_3of3": n3,
            "union_pos": union, "majority_pos": n2 + n3,
            "pct_split_vote": round(100 * (n1 + n2) / max(union, 1), 1),  # readers disagreed
            "n_boxes": int(boxes.get(k, 0)) if k < 14 else int(boxes.sum()),
        })
    return pd.DataFrame(rows)


def plot_stats(stats, path):
    s = stats[stats.finding != "Abnormal"].sort_values("union_pos")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(s))
    left = np.zeros(len(s))
    for col, label, color in [("n_3of3", "3/3 readers", "#1b7837"),
                              ("n_2of3", "2/3 readers", "#7fbf7b"),
                              ("n_1of3", "1/3 readers", "#d9d9d9")]:
        ax.barh(y, s[col], left=left, color=color, label=label)
        left += s[col].values
    ax.set_yticks(y, s.finding)
    ax.set_xlabel("number of training images (of 15,000)")
    ax.set_title("Reader agreement per finding (image level, VinDr-CXR train)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def build_cache(image_ids, img_dir, out_path, workers=12):
    """Decode every PNG once into a uint8 memmap so training never touches PNGs."""
    paths = [os.path.join(img_dir, f"{i}.png") for i in image_ids]
    cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                      shape=(len(paths), IMG_SIZE, IMG_SIZE))
    with Pool(workers) as pool:
        for i, img in enumerate(pool.imap(load_image, paths, chunksize=32)):
            cache[i] = img
            if i % 2000 == 0:
                print(f"  cached {i}/{len(paths)}", flush=True)
    cache.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--meta_csv", default=None, help="csv with image_id,width,height (original DICOM size)")
    ap.add_argument("--data_dir", default="/workspace/data")
    ap.add_argument("--results_dir", default="/workspace/results")
    ap.add_argument("--skip_cache", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    print(f"{len(df)} rows, {df.image_id.nunique()} images, {df.rad_id.nunique()} radiologists")
    image_ids, votes = compute_votes(df)
    np.savetxt(os.path.join(args.data_dir, "image_ids.txt"), image_ids, fmt="%s")

    # --- label files: union / majority / soft (+ raw votes) ---
    def save(arr, name):
        out = pd.DataFrame(arr, columns=TARGETS)
        out.insert(0, "image_id", image_ids)
        out.to_csv(os.path.join(args.data_dir, name), index=False)

    save(votes, "votes.csv")
    save((votes >= 1).astype(int), "labels_union.csv")
    save((votes >= 2).astype(int), "labels_majority.csv")
    save(np.round(votes / 3.0, 4), "labels_soft.csv")

    # --- splits (seed 42; never regenerate after training starts) ---
    splits_path = os.path.join(args.data_dir, "splits.csv")
    if os.path.exists(splits_path):
        splits = pd.read_csv(splits_path)
        assert (splits.image_id.values == image_ids).all(), "splits.csv does not match image ids"
        print("splits.csv already exists - kept unchanged")
    else:
        splits = make_splits(image_ids, votes)
        splits.to_csv(splits_path, index=False)
    print("split sizes:", splits.split.value_counts().to_dict())

    # majority positives per class per split (to check stratification)
    maj = pd.DataFrame((votes >= 2).astype(int), columns=TARGETS)
    maj["split"] = splits.split.values
    counts = maj.groupby("split").sum().T[["train", "val", "test", "disagree"]]
    counts.to_csv(os.path.join(args.results_dir, "split_counts.csv"))
    print(counts)

    # --- disagreement statistics ---
    stats = disagreement_stats(votes, df)
    stats.to_csv(os.path.join(args.results_dir, "disagreement_stats.csv"), index=False)
    plot_stats(stats, os.path.join(args.results_dir, "fig_disagreement.png"))
    print(stats.to_string(index=False))

    # --- original image sizes (needed to rescale boxes for YOLO) ---
    if args.meta_csv:
        meta = pd.read_csv(args.meta_csv)[["image_id", "width", "height"]].drop_duplicates("image_id")
        meta = pd.DataFrame({"image_id": image_ids}).merge(meta, on="image_id", how="left")
        print("images missing original size:", int(meta.width.isna().sum()))
        meta.to_csv(os.path.join(args.data_dir, "meta.csv"), index=False)

    # --- image cache ---
    if not args.skip_cache:
        build_cache(image_ids, args.img_dir, os.path.join(args.data_dir, "images_512.npy"))
        print("image cache written")


if __name__ == "__main__":
    main()
