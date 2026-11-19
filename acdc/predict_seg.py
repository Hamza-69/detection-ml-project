"""Step 3 - predict segmentations for all 150 patients with one trained configuration and score them.

    python -m acdc.predict_seg --cfg unet2d

Protocol (no test-set leakage):
  training patients 001-100: out-of-fold prediction, i.e. by the model of the fold the patient was held out in;
  test patients 101-150:     average of the 5 fold models' softmax outputs (ensemble).
Both use mirroring test-time augmentation (4 flips) unless --no_tta. Probabilities (1.25 mm) are resampled to the
original grid, argmax'ed, and each structure is reduced to its largest connected component.

Writes WORK/preds/<cfg>/<pid>.npz (seg_ED, seg_ES on the original grid, uint8; per-slice mean predictive entropy) and
results/seg/<cfg>_metrics.csv (one row per patient x phase: Dice, HD, HD95, ASSD per structure + uncertainty summaries).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from acdc.common import CLASSES, RESULTS, WORK, keep_largest_component, load_phase, probs_to_original, read_info, seg_metrics
from acdc.infer import load_model, predict_volume


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Predictive entropy per pixel of (Z,C,H,W) probabilities, normalised to [0,1] by log(C)."""
    return -(p.clamp_min(1e-8).log() * p).sum(1) / np.log(p.shape[1])


def finalize(prob_r: torch.Tensor, orig_xy, postprocess=True):
    """1.25 mm probabilities -> (segmentation on the original grid, entropy map on the original grid)."""
    p = probs_to_original(prob_r.float(), orig_xy)
    seg = p.argmax(1).numpy().astype(np.uint8)
    if postprocess:
        seg = keep_largest_component(seg)
    return seg, entropy(p).numpy().astype(np.float16)


def uncertainty_summary(seg, ent, member_segs=None):
    """Volume-level uncertainty scores used for failure detection:
    ent_fg   mean entropy inside the (dilated) predicted heart region,
    disagree fraction of predicted-heart voxels on which ensemble members / TTA variants disagree (if available)."""
    from scipy import ndimage
    heart = ndimage.binary_dilation(seg > 0, iterations=3)
    out = {"ent_fg": float(ent[heart].mean()) if heart.any() else float("nan")}
    if member_segs is not None:
        stack = np.stack(member_segs)
        dis = (stack != stack[0:1]).any(0)
        out["disagree"] = float(dis[heart].mean()) if heart.any() else float("nan")
    return out


def score_and_save(cfg, pid, phase_out, meta_row, extra=None):
    """phase_out: {'ED': (seg, ent, member_segs), 'ES': ...}. Saves the npz and returns metric rows."""
    rows, arrays = [], {}
    for ph, (seg, ent, members) in phase_out.items():
        _, gt, sp = load_phase(pid, ph)
        arrays[f"seg_{ph}"] = seg
        arrays[f"ent_slice_{ph}"] = ent.reshape(ent.shape[0], -1).mean(1)
        r = {"pid": pid, "phase": ph, "split": meta_row.split, "group": meta_row.group, "fold": meta_row.fold,
             **seg_metrics(seg, gt, sp), **uncertainty_summary(seg, ent, members)}
        if extra:
            r.update(extra)
        rows.append(r)
    np.savez_compressed(WORK / "preds" / cfg / f"{pid}.npz", **arrays)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="unet2d")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--no_pp", action="store_true", help="no largest-connected-component post-processing")
    ap.add_argument("--out", default=None, help="name for outputs (default: cfg, + suffixes)")
    args = ap.parse_args()
    name = args.out or args.cfg + ("_notta" if args.no_tta else "") + ("_nopp" if args.no_pp else "")
    # Mirroring TTA is only valid for models trained with mirroring augmentation: a model trained without
    # augmentation has never seen flipped hearts (RV on the other side) and its flipped predictions are wrong.
    trained_aug = load_model(WORK / "runs" / f"{args.cfg}_f0" / "model.pt", "cpu")[1]["args"]["aug"]
    if trained_aug == "none" and not args.no_tta:
        print("model trained without augmentation -> predicting without mirroring TTA")
        args.no_tta = True
    (WORK / "preds" / name).mkdir(parents=True, exist_ok=True)
    (RESULTS / "seg").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    models = [load_model(WORK / "runs" / f"{args.cfg}_f{f}" / "model.pt")[0] for f in range(5)]
    in_ch = load_model(WORK / "runs" / f"{args.cfg}_f0" / "model.pt", "cpu")[1]["args"]["in_ch"]

    rows, t0 = [], time.time()
    for r in meta.itertuples():
        d = np.load(WORK / "cache" / f"{r.pid}.npz")
        orig_xy = tuple(int(v) for v in d["orig_shape"][1:])
        members = [models[r.fold]] if r.split == "train" else models
        out = {}
        for ph in ("ED", "ES"):
            img = torch.from_numpy(d[f"img_{ph}"].astype(np.float32))
            per_member = [predict_volume(m, img, in_ch, tta=not args.no_tta) for m in members]
            prob = torch.stack(per_member).mean(0)
            seg, ent = finalize(prob, orig_xy, not args.no_pp)
            msegs = [finalize(p, orig_xy, not args.no_pp)[0] for p in per_member] if len(members) > 1 else None
            out[ph] = (seg, ent, msegs)
        rows += score_and_save(name, r.pid, out, r)
        print(r.pid, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith("dice")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "seg" / f"{name}_metrics.csv", index=False)
    print(f"{name}: {(time.time() - t0) / 60:.1f} min")
    print(df.groupby(["split", "phase"])[[f"dice_{c}" for c in CLASSES.values()]].mean().round(4))


if __name__ == "__main__":
    main()
