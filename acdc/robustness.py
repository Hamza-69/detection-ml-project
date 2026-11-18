"""Step 7 - robustness to image-quality degradations (failure analysis), on the 50 test patients.

Each degradation is applied to the RAW image on its original grid (before normalisation / resampling), at increasing
severity, then the full pipeline runs unchanged: normalise -> resample -> segmentation (5-fold ensemble, no TTA) ->
clinical features -> diagnosis (logistic regression trained on the out-of-fold predicted features of the 100 training
patients of the same segmentation model, i.e. the automatic pipeline "<cfg>-><cfg>").

  noise      Rician noise (Gaussian on real + imaginary channel), sigma = level x 99th intensity percentile
  blur       Gaussian in-plane blur, sigma in mm
  lowres     in-plane down-sampling by a factor, then bilinear up-sampling back
  bias       smooth multiplicative intensity field (coil inhomogeneity), exp(level * random smooth field in [-1, 1])
  ghosting   motion: 30% of phase-encoding lines (k-space rows) get a random in-plane shift of up to `level` pixels
  gamma      contrast change x -> x^level on the [0, 1]-rescaled image

    python -m acdc.robustness --cfgs unet2d unet2d_noaug unet25d
Writes results/robustness/robustness_seg.csv, robustness_cls.csv, fig_robustness.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from acdc.classify import make_model
from acdc.common import CLASSES, GROUPS, RESULTS, WORK, dice, keep_largest_component, load_phase, normalize, \
    probs_to_original, resample_image
from acdc.features import feature_table, patient_features
from acdc.infer import load_model, predict_volume

OUT = RESULTS / "robustness"
LEVELS = {"noise": [0.02, 0.05, 0.1, 0.15], "blur": [1.0, 2.0, 3.0, 4.0], "lowres": [1.5, 2.0, 3.0, 4.0],
          "bias": [0.2, 0.4, 0.6, 0.8], "ghosting": [2, 4, 8, 12], "gamma": [0.5, 0.7, 1.5, 2.0]}


def degrade(img, spacing, kind, level, rng):
    """img: (Z,X,Y) float32 raw intensities on the original grid."""
    if kind == "noise":
        s = level * np.percentile(img, 99)
        return np.sqrt((img + rng.normal(0, s, img.shape)) ** 2 + rng.normal(0, s, img.shape) ** 2).astype(np.float32)
    if kind == "blur":
        return ndimage.gaussian_filter(img, sigma=(0, level / spacing[0], level / spacing[1])).astype(np.float32)
    if kind == "lowres":
        t = torch.from_numpy(img)[:, None]
        small = torch.nn.functional.interpolate(t, scale_factor=1 / level, mode="area")
        return torch.nn.functional.interpolate(small, size=img.shape[1:], mode="bilinear", align_corners=False)[:, 0].numpy()
    if kind == "bias":
        Z, X, Y = img.shape
        coarse = rng.uniform(-1, 1, (4, 4))  # smooth random field from a 4x4 grid of control values
        field = ndimage.zoom(coarse, (X / 4, Y / 4), order=3)[:X, :Y]
        field = field / (np.abs(field).max() + 1e-8)
        return (img * np.exp(level * field)[None]).astype(np.float32)
    if kind == "ghosting":
        out = np.empty_like(img)
        for z in range(img.shape[0]):
            k = np.fft.fftshift(np.fft.fft2(img[z]))
            rows = rng.random(k.shape[0]) < 0.3  # corrupted phase-encode lines
            shifts = rng.uniform(-level, level, rows.sum())
            ky = np.fft.fftshift(np.fft.fftfreq(k.shape[1]))
            k[rows] *= np.exp(-2j * np.pi * shifts[:, None] * ky[None, :])
            out[z] = np.abs(np.fft.ifft2(np.fft.ifftshift(k)))
        return out.astype(np.float32)
    if kind == "gamma":
        lo, hi = img.min(), img.max()
        return (((img - lo) / (hi - lo + 1e-8)) ** level * (hi - lo) + lo).astype(np.float32)
    raise ValueError(kind)


def segment(models, in_ch, img, spacing):
    x = torch.from_numpy(resample_image(normalize(img), spacing[:2]))
    p = predict_volume(models, x, in_ch, tta=False)
    return keep_largest_component(probs_to_original(p, img.shape[1:]).argmax(1).numpy().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", default=["unet2d", "unet2d_noaug", "unet25d"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    test = meta[meta.split == "test"]
    enc = {g: i for i, g in enumerate(GROUPS)}
    seg_rows, cls_rows = [], []
    conditions = [("clean", 0)] + [(k, v) for k, vs in LEVELS.items() for v in vs]
    for cfg in args.cfgs:
        if not (WORK / "runs" / f"{cfg}_f4" / "model.pt").exists():
            print("skip", cfg)
            continue
        models = [load_model(WORK / "runs" / f"{cfg}_f{f}" / "model.pt")[0] for f in range(5)]
        in_ch = load_model(WORK / "runs" / f"{cfg}_f0" / "model.pt", "cpu")[1]["args"]["in_ch"]
        # diagnosis model of the automatic pipeline (trained on out-of-fold predicted features of the 100 training patients)
        ft = feature_table(cfg)
        feats = [c for c in ft.columns if c not in ("pid", "split", "fold", "group")]
        tr = ft[ft.split == "train"]
        clf = make_model("logreg", len(feats)).fit(tr[feats].values, tr.group.map(enc).values)
        for kind, level in conditions:
            feat_rows = []
            for r in test.itertuples():
                rng = np.random.default_rng(int(r.pid[-3:]) * 1000 + int(level * 100))
                segs = {}
                for ph in ("ED", "ES"):
                    img, gt, sp = load_phase(r.pid, ph)
                    im = img if kind == "clean" else degrade(img, sp, kind, level, rng)
                    segs[ph] = segment(models, in_ch, im, sp)
                    seg_rows.append({"cfg": cfg, "degradation": kind, "level": level, "pid": r.pid, "phase": ph,
                                     **{f"dice_{s}": dice(segs[ph] == c, gt == c) for c, s in CLASSES.items()}})
                feat_rows.append(patient_features(segs["ED"], segs["ES"], sp, r.height, r.weight))
            X = pd.DataFrame(feat_rows)[feats].values
            prob = clf.predict_proba(X)
            acc = float((prob.argmax(1) == test.group.map(enc).values).mean())
            d = pd.DataFrame(seg_rows)
            d = d[(d.cfg == cfg) & (d.degradation == kind) & (d.level == level)]
            md = d[[f"dice_{s}" for s in CLASSES.values()]].mean().mean()
            cls_rows.append({"cfg": cfg, "degradation": kind, "level": level, "diag_acc": acc, "mean_dice": md})
            print(f"{cfg:14s} {kind:9s} {level:5}: mean Dice {md:.3f}  diagnosis acc {acc:.2f}", flush=True)
        pd.DataFrame(seg_rows).to_csv(OUT / "robustness_seg.csv", index=False)
        pd.DataFrame(cls_rows).to_csv(OUT / "robustness_cls.csv", index=False)
    plot()


def plot():
    c = pd.read_csv(OUT / "robustness_cls.csv")
    kinds = list(LEVELS)
    fig, ax = plt.subplots(2, len(kinds), figsize=(4 * len(kinds), 7.5))
    for j, k in enumerate(kinds):
        for cfg, d in c.groupby("cfg"):
            clean = d[d.degradation == "clean"]
            dd = pd.concat([clean.assign(level=1.0 if k in ("lowres", "gamma") else 0.0), d[d.degradation == k]])
            dd = dd.sort_values("level")
            ax[0, j].plot(dd.level, dd.mean_dice, "o-", label=cfg)
            ax[1, j].plot(dd.level, dd.diag_acc, "o-", label=cfg)
        ax[0, j].set_title(k); ax[1, j].set_xlabel("severity")
        ax[0, j].grid(alpha=0.3); ax[1, j].grid(alpha=0.3)
        ax[0, j].set_ylim(0, 1); ax[1, j].set_ylim(0, 1)
    ax[0, 0].set_ylabel("mean Dice (test, 3 structures)"); ax[1, 0].set_ylabel("diagnosis accuracy (test)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Robustness to image degradations (50 test patients; leftmost point = clean image)")
    fig.tight_layout(); fig.savefig(OUT / "fig_robustness.png", dpi=110); plt.close(fig)


def show_degradations(pid="patient101"):
    """Figure: the mid slice of one test patient under every degradation and level."""
    OUT.mkdir(parents=True, exist_ok=True)
    img, _, sp = load_phase(pid, "ED")
    z = img.shape[0] // 2
    fig, ax = plt.subplots(len(LEVELS), 5, figsize=(13, 2.7 * len(LEVELS)))
    for i, (k, levels) in enumerate(LEVELS.items()):
        ax[i, 0].imshow(img[z].T, cmap="gray"); ax[i, 0].set_title("clean", fontsize=9)
        for j, lv in enumerate(levels):
            d = degrade(img, sp, k, lv, np.random.default_rng(0))
            ax[i, j + 1].imshow(d[z].T, cmap="gray"); ax[i, j + 1].set_title(f"{k} {lv}", fontsize=9)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_degradations.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
