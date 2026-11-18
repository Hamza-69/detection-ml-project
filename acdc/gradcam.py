"""Step 9 - Grad-CAM for the end-to-end CNN: does it look at the heart?

For each test patient and each of the 5 fold models of one seed, Grad-CAM (Selvaraju et al., 2017) is computed for the
predicted class on the last ResNet block, for every slice (the model sees all slices at once; the gradient of the
patient-level logit gives one map per slice). Maps are averaged over the 5 models and compared with the ground-truth
heart (RV + myocardium + LV, ED or ES, dilated by 10 px = 12.5 mm):
  energy_in_heart   fraction of the total CAM energy (all slices) that falls inside the heart region
  heart_area        fraction of the image area covered by the heart region (= expected energy fraction if the CAM
                    were spread uniformly); enrichment = energy_in_heart / heart_area
  pointing_game     whether the single highest CAM pixel lies inside the heart region
Results are split by correct vs wrong predictions. This replaces the planned "Grad-CAM vs radiologist boxes" analysis
for the new dataset.

    python -m acdc.gradcam --seed 0
Writes results/cls/gradcam.csv, gradcam_summary.json, fig_gradcam.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from acdc.augment import to_canvas
from acdc.common import GROUPS, RESULTS, WORK
from acdc.train_cls import CROP, SliceNet, load_stack

OUT = RESULTS / "cls"


def gradcam(model, stack):
    """stack: (Z,3,H,W) on GPU. Returns (CAM (Z,H,W) in [0,1] scale-free, predicted class, probabilities)."""
    feats = {}
    h = model.encoder.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    model.eval()
    logits = model([stack])
    h.remove()
    a = feats["a"]  # (Z,512,8,8)
    c = int(logits.argmax(1))
    g = torch.autograd.grad(logits[0, c], a)[0]
    w = g.mean((2, 3), keepdim=True)  # channel weights per slice
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=stack.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.detach().cpu().numpy(), c, torch.softmax(logits, 1)[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="resnet18")
    args = ap.parse_args()
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    test = meta[meta.split == "test"]
    models = []
    for f in range(5):
        ck = torch.load(WORK / "runs" / f"cls_{args.tag}_s{args.seed}_f{f}" / "model.pt", map_location="cpu", weights_only=False)
        m = SliceNet(pretrained=False)
        m.load_state_dict(ck["model"])
        models.append(m.cuda().eval())
    rows, examples = [], []
    for r in test.itertuples():
        x = load_stack(r.pid).cuda().float()
        cams, probs = [], []
        for m in models:
            cam, _, p = gradcam(m, x)
            cams.append(cam / (cam.max() + 1e-8))  # each model's maps scaled to [0,1] before averaging
            probs.append(p)
        cam, prob = np.mean(cams, 0), np.mean(probs, 0)
        d = np.load(WORK / "cache" / f"{r.pid}.npz")
        heart = torch.from_numpy(((d["lab_ED"] > 0) | (d["lab_ES"] > 0)).astype(np.uint8))
        heart = to_canvas(heart, 0, CROP).numpy().astype(bool)
        heart = np.stack([ndimage.binary_dilation(s, iterations=10) if s.any() else s for s in heart])
        has = heart.any((1, 2))  # slices that contain the heart
        e_in = cam[heart].sum() / (cam.sum() + 1e-8)
        area = heart.mean()  # over all slices, like the energy fraction
        z, yy, xx = np.unravel_index(np.argmax(cam), cam.shape)
        pred = int(prob.argmax())
        rows.append({"pid": r.pid, "group": r.group, "pred": GROUPS[pred], "correct": GROUPS[pred] == r.group,
                     "energy_in_heart": float(e_in), "heart_area": float(area), "enrichment": float(e_in / area),
                     "energy_on_heart_slices": float(cam[has].sum() / (cam.sum() + 1e-8)),
                     "pointing_game": bool(heart[z, yy, xx])})
        if len(examples) < 5 and r.group not in [e[1] for e in examples]:
            zc = int(np.argmax(heart.sum((1, 2))))
            examples.append((r.pid, r.group, GROUPS[pred], x[zc, 0].cpu().numpy(), cam[zc], heart[zc]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gradcam.csv", index=False)
    summ = {"seed": args.seed, "n": len(df),
            "all": df[["energy_in_heart", "heart_area", "enrichment", "pointing_game", "energy_on_heart_slices"]].mean().to_dict()}
    for k, d in df.groupby("correct"):
        summ["correct" if k else "wrong"] = {"n": len(d), **d[["energy_in_heart", "enrichment", "pointing_game"]].mean().to_dict()}
    (OUT / "gradcam_summary.json").write_text(json.dumps(summ, indent=2))
    fig, ax = plt.subplots(2, len(examples), figsize=(3.4 * len(examples), 7))
    for j, (pid, g, p, im, cam, heart) in enumerate(examples):
        ax[0, j].imshow(im.T, cmap="gray"); ax[0, j].contour(heart.T, levels=[0.5], colors="lime", linewidths=0.8)
        ax[0, j].set_title(f"{pid}: true {g}, pred {p}", fontsize=9)
        ax[1, j].imshow(im.T, cmap="gray"); ax[1, j].imshow(cam.T, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax[1, j].contour(heart.T, levels=[0.5], colors="white", linewidths=0.8)
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM of the end-to-end CNN (ED channel shown; green/white = heart region). "
                 f"Energy in heart {summ['all']['energy_in_heart']:.2f} vs area {summ['all']['heart_area']:.2f}")
    fig.tight_layout(); fig.savefig(OUT / "fig_gradcam.png", dpi=110); plt.close(fig)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
