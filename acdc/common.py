"""Shared helpers for the ACDC pipeline: paths, loading, resampling, metrics and clinical indices.

ACDC label convention (ground-truth files *_gt.nii.gz):
    0 = background, 1 = right-ventricle (RV) cavity, 2 = left-ventricle myocardium (MYO), 3 = left-ventricle (LV) cavity.
Arrays from nibabel are (X, Y, Z); we work with (Z, X, Y) stacks of 2D short-axis slices.
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# ----------------------------------------------------------------------------------------------- paths
DATA_ROOT = Path(os.environ.get("ACDC_ROOT", "/home/ubuntu/data/acdc/raw/ACDC/database"))
WORK = Path(os.environ.get("ACDC_WORK", "/home/ubuntu/data/acdc/work"))  # caches, checkpoints, predictions
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"  # small tables / figures / json that go into git

CLASSES = {1: "RV", 2: "MYO", 3: "LV"}
GROUPS = ["NOR", "MINF", "DCM", "HCM", "RV"]  # NOR normal, MINF previous infarction, DCM dilated, HCM hypertrophic, RV abnormal RV
TARGET_SPACING = 1.25  # mm, in-plane spacing all slices are resampled to for the networks
MYO_DENSITY = 1.05  # g/mL, to turn myocardial volume into mass


# ----------------------------------------------------------------------------------------------- loading
def patient_dir(pid: str) -> Path:
    n = int(pid[-3:])
    return DATA_ROOT / ("training" if n <= 100 else "testing") / pid


def read_info(pid: str) -> dict:
    """Info.cfg -> dict(ED, ES, Group, Height, Weight, NbFrame)."""
    info = {}
    for line in (patient_dir(pid) / "Info.cfg").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return {"ED": int(info["ED"]), "ES": int(info["ES"]), "group": info["Group"], "height": float(info["Height"]),
            "weight": float(info["Weight"]), "n_frames": int(info["NbFrame"])}


def load_phase(pid: str, phase: str, info: dict | None = None):
    """Return image (Z,X,Y) float32, ground truth (Z,X,Y) uint8, spacing (sx, sy, sz) in mm for phase 'ED' or 'ES'."""
    info = info or read_info(pid)
    stem = patient_dir(pid) / f"{pid}_frame{info[phase]:02d}"
    im = nib.load(f"{stem}.nii.gz")
    img = np.asarray(im.dataobj, dtype=np.float32).transpose(2, 0, 1)
    gt = np.asarray(nib.load(f"{stem}_gt.nii.gz").dataobj).astype(np.uint8).transpose(2, 0, 1)
    spacing = tuple(float(s) for s in im.header.get_zooms()[:3])
    return img, gt, spacing


# ----------------------------------------------------------------------------------------------- preprocessing
def normalize(img: np.ndarray) -> np.ndarray:
    """Clip to the volume's 0.5-99.5 percentiles, then z-score (MRI intensities have no absolute scale)."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip(img, lo, hi)
    return ((img - img.mean()) / (img.std() + 1e-8)).astype(np.float32)


def resampled_shape(shape_xy, spacing_xy, target=TARGET_SPACING):
    return tuple(int(round(n * s / target)) for n, s in zip(shape_xy, spacing_xy))


def resample_image(img: np.ndarray, spacing_xy, target=TARGET_SPACING) -> np.ndarray:
    """(Z,X,Y) image -> in-plane spacing `target` (bilinear, slice by slice; z untouched)."""
    out = resampled_shape(img.shape[1:], spacing_xy, target)
    t = torch.from_numpy(img)[:, None]
    return F.interpolate(t, size=out, mode="bilinear", align_corners=False)[:, 0].numpy()


def resample_labels(lab: np.ndarray, size_xy, n_classes=4) -> np.ndarray:
    """Resample a label map by linearly interpolating its one-hot encoding and taking argmax (smoother than nearest)."""
    t = F.one_hot(torch.from_numpy(lab.astype(np.int64)), n_classes).permute(0, 3, 1, 2).float()
    return F.interpolate(t, size=tuple(size_xy), mode="bilinear", align_corners=False).argmax(1).numpy().astype(np.uint8)


def probs_to_original(prob: torch.Tensor, orig_xy) -> torch.Tensor:
    """(Z,C,H,W) class probabilities at 1.25 mm -> (Z,C,X,Y) at the original grid (bilinear)."""
    return F.interpolate(prob, size=tuple(orig_xy), mode="bilinear", align_corners=False)


def keep_largest_component(seg: np.ndarray) -> np.ndarray:
    """Post-processing: for each structure keep only its largest 3D connected component (removes stray blobs)."""
    out = np.zeros_like(seg)
    for c in CLASSES:
        m = seg == c
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
        out[m] = c
    return out


# ----------------------------------------------------------------------------------------------- metrics
def dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.sum(), b.sum()
    if sa + sb == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / (sa + sb))


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing_zxy) -> np.ndarray:
    """Distances (mm) from the surface voxels of a to the surface of b. Surface = mask minus its erosion
    (3D, 6-connectivity), as in medpy which the ACDC challenge evaluation used."""
    st = ndimage.generate_binary_structure(3, 1)
    sa = a ^ ndimage.binary_erosion(a, structure=st, iterations=1)
    sb = b ^ ndimage.binary_erosion(b, structure=st, iterations=1)
    dt = ndimage.distance_transform_edt(~sb, sampling=spacing_zxy)
    return dt[sa]


def surface_metrics(a: np.ndarray, b: np.ndarray, spacing_zxy):
    """Hausdorff distance (max), 95th-percentile HD and average symmetric surface distance in mm.
    NaN when either mask is empty (counted as a failure in the tables)."""
    if not a.any() or not b.any():
        return float("nan"), float("nan"), float("nan")
    d1, d2 = _surface_distances(a, b, spacing_zxy), _surface_distances(b, a, spacing_zxy)
    both = np.concatenate([d1, d2])
    return float(max(d1.max(), d2.max())), float(max(np.percentile(d1, 95), np.percentile(d2, 95))), float(both.mean())


def seg_metrics(pred: np.ndarray, gt: np.ndarray, spacing_xyz) -> dict:
    """Per-structure Dice / HD / HD95 / ASSD for one (Z,X,Y) volume. spacing given as nibabel (sx, sy, sz)."""
    sp = (spacing_xyz[2], spacing_xyz[0], spacing_xyz[1])
    out = {}
    for c, name in CLASSES.items():
        p, g = pred == c, gt == c
        out[f"dice_{name}"] = dice(p, g)
        out[f"hd_{name}"], out[f"hd95_{name}"], out[f"assd_{name}"] = surface_metrics(p, g, sp)
    return out


# ----------------------------------------------------------------------------------------------- clinical indices
def volume_ml(mask: np.ndarray, spacing_xyz) -> float:
    return float(mask.sum() * np.prod(spacing_xyz) / 1000.0)


def clinical_indices(seg_ed: np.ndarray, seg_es: np.ndarray, spacing_xyz) -> dict:
    """Standard indices used in the ACDC challenge: LV/RV end-diastolic and end-systolic volume (mL), ejection
    fraction (%), myocardial mass at ED (g)."""
    lv_edv, lv_esv = volume_ml(seg_ed == 3, spacing_xyz), volume_ml(seg_es == 3, spacing_xyz)
    rv_edv, rv_esv = volume_ml(seg_ed == 1, spacing_xyz), volume_ml(seg_es == 1, spacing_xyz)
    ef = lambda edv, esv: 100.0 * (edv - esv) / edv if edv > 0 else float("nan")  # noqa: E731
    return {"LV_EDV": lv_edv, "LV_ESV": lv_esv, "LV_EF": ef(lv_edv, lv_esv), "RV_EDV": rv_edv, "RV_ESV": rv_esv,
            "RV_EF": ef(rv_edv, rv_esv), "MYO_mass": MYO_DENSITY * volume_ml(seg_ed == 2, spacing_xyz)}


def body_surface_area(height_cm: float, weight_kg: float) -> float:
    """Mosteller formula, m^2."""
    return float(np.sqrt(height_cm * weight_kg / 3600.0))


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
