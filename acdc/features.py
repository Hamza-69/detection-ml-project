"""Interpretable cardiac features computed from a segmentation (ground truth or predicted), used for diagnosis.

These are the quantities a cardiologist reads from a cine-MRI report (they are the basis of the ACDC group definitions):
  volumes      LV / RV end-diastolic and end-systolic volume (mL), also indexed by body-surface area (mL/m^2)
  function     LV / RV ejection fraction (%)
  myocardium   mass at ED (g, indexed), volume at ES, mass / LV EDV ratio
  ratios       RV / LV volume at ED and ES
  wall         myocardial thickness (mm) at ED and ES: max, mean, and the spread (std / mean) around the ring, which is
               high when one wall segment is thin (infarct scar) or thick (asymmetric hypertrophy).
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from acdc.common import RESULTS, WORK, body_surface_area, clinical_indices, load_phase, volume_ml


def wall_thickness(seg_slice: np.ndarray, spacing_xy) -> np.ndarray:
    """Thickness samples (mm) of the myocardial ring in one slice: for each epicardial-border pixel, the distance to the
    LV cavity. Empty if the slice has no closed LV + myocardium."""
    myo, lv = seg_slice == 2, seg_slice == 3
    if myo.sum() < 10 or lv.sum() < 5:
        return np.zeros(0)
    dist_to_lv = ndimage.distance_transform_edt(~lv, sampling=spacing_xy)
    wall = myo | lv
    epi_border = myo & ~ndimage.binary_erosion(wall)  # outer contour of the myocardium
    return dist_to_lv[epi_border]


def thickness_stats(seg: np.ndarray, spacing_xyz):
    per_slice = [wall_thickness(s, spacing_xyz[:2]) for s in seg]
    per_slice = [t for t in per_slice if len(t) > 0]
    if not per_slice:
        return {"max": 0.0, "mean": 0.0, "cv": 0.0}
    allt = np.concatenate(per_slice)
    cv = np.mean([t.std() / (t.mean() + 1e-6) for t in per_slice])
    # robust maximum: 95th percentile within each slice, max over slices (single-pixel spikes ignored)
    return {"max": float(max(np.percentile(t, 95) for t in per_slice)), "mean": float(allt.mean()), "cv": float(cv)}


def patient_features(seg_ed, seg_es, spacing_xyz, height, weight) -> dict:
    ci = clinical_indices(seg_ed, seg_es, spacing_xyz)
    bsa = body_surface_area(height, weight)
    myo_es = volume_ml(seg_es == 2, spacing_xyz)
    f = dict(ci)
    f.update({
        "LV_EDVi": ci["LV_EDV"] / bsa, "LV_ESVi": ci["LV_ESV"] / bsa, "RV_EDVi": ci["RV_EDV"] / bsa,
        "RV_ESVi": ci["RV_ESV"] / bsa, "MYO_massi": ci["MYO_mass"] / bsa, "MYO_vol_ES": myo_es,
        "RV_LV_ED": ci["RV_EDV"] / max(ci["LV_EDV"], 1e-3), "RV_LV_ES": ci["RV_ESV"] / max(ci["LV_ESV"], 1e-3),
        "MYO_LV_ED": ci["MYO_mass"] / max(ci["LV_EDV"], 1e-3),
    })
    for ph, seg in (("ED", seg_ed), ("ES", seg_es)):
        for k, v in thickness_stats(seg, spacing_xyz).items():
            f[f"thick_{k}_{ph}"] = v
    # an empty structure gives EF = nan; replace with 0 so classifiers can still run (rare, only on failed predictions)
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in f.items()}


FEATURES = None  # filled on first call (column order)


def feature_table(source: str) -> pd.DataFrame:
    """Features for all 150 patients from source 'gt' (ground-truth masks) or a prediction folder name in WORK/preds."""
    global FEATURES
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    cache = WORK / "features" / f"{source}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for r in meta.itertuples():
        if source == "gt":
            _, s_ed, sp = load_phase(r.pid, "ED")
            _, s_es, _ = load_phase(r.pid, "ES")
        else:
            d = np.load(WORK / "preds" / source / f"{r.pid}.npz")
            s_ed, s_es, sp = d["seg_ED"], d["seg_ES"], (r.sx, r.sy, r.sz)
        rows.append({"pid": r.pid, "split": r.split, "fold": r.fold, "group": r.group,
                     **patient_features(s_ed, s_es, sp, r.height, r.weight)})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    df = feature_table("gt")
    print(df.groupby("group").mean(numeric_only=True).T.round(1))
