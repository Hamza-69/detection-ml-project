"""Check that the 512 px PNGs we train on are the official competition images.

For a random sample of training images we download the official DICOM from the
Kaggle competition, convert it ourselves (VOI LUT, invert MONOCHROME1, min-max to
uint8, bilinear resize to 512x512 with cv2) and compare with the PNG of the same image_id.
Also records each DICOM's original width/height and compares with the sizes we use for boxes.

  uv run python verify_images.py --dicom_dir /workspace/data/dicom_check/train \
      --png_dir /workspace/data/png512/vinbigdata/train
Writes results/image_verification.csv
"""
import argparse
import glob
import io
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def dicom_to_uint8(ds):
    img = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if ds.PhotometricInterpretation == "MONOCHROME1":  # inverted: high = dark
        img = img.max() - img
    img -= img.min()
    img /= max(img.max(), 1e-6)
    return (img * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", required=True)
    ap.add_argument("--png_dir", required=True)
    ap.add_argument("--sizes_csv", default="/workspace/data/dl_awsaf/vinbigdata/train.csv",
                    help="csv with image_id,width,height used to rescale boxes")
    ap.add_argument("--out", default="/workspace/results/image_verification.csv")
    args = ap.parse_args()
    sizes = pd.read_csv(args.sizes_csv)[["image_id", "width", "height"]].drop_duplicates("image_id").set_index("image_id")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dicom_dir, "*.dicom*"))):
        image_id = os.path.basename(path).split(".")[0]
        if path.endswith(".zip"):  # Kaggle wraps single-file downloads in a zip
            with zipfile.ZipFile(path) as z:
                ds = pydicom.dcmread(io.BytesIO(z.read(z.namelist()[0])))
        else:
            ds = pydicom.dcmread(path)
        ours = dicom_to_uint8(ds)
        h, w = ours.shape
        ours512 = cv2.resize(ours, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        png = cv2.imread(os.path.join(args.png_dir, f"{image_id}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        rows.append({
            "image_id": image_id, "dicom_width": w, "dicom_height": h,
            "csv_width": int(sizes.loc[image_id, "width"]), "csv_height": int(sizes.loc[image_id, "height"]),
            "photometric": ds.PhotometricInterpretation, "png_shape": "x".join(map(str, png.shape)),
            "pearson_r": round(float(np.corrcoef(ours512.ravel(), png.ravel())[0, 1]), 5),
            "mean_abs_diff_gray": round(float(np.abs(ours512 - png).mean()), 3),
            "max_abs_diff_gray": int(np.abs(ours512 - png).max()),
            "identical_pixels_frac": round(float((ours512 == png).mean()), 5),
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["size_match"] = (df.dicom_width == df.csv_width) & (df.dicom_height == df.csv_height)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} images: min Pearson r {df.pearson_r.min():.4f}, "
          f"mean |diff| {df.mean_abs_diff_gray.mean():.2f} gray levels, sizes match: {df.size_match.all()}, "
          f"max pixel diff {df.max_abs_diff_gray.max()} gray level(s), "
          f"images bit-identical: {(df.identical_pixels_frac == 1).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
