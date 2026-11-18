"""Step 5 - end-to-end CNN diagnosis (no segmentation), one fold + seed per call.

    python -m acdc.train_cls --fold 0 --seed 0

Input per patient: every short-axis slice as a 3-channel image [ED, ES, ED-ES] (same slice position at the two phases,
so the difference channel shows how much the wall moves), 1.25 mm pixels, 256x256 centre crop (320 mm, contains the
heart in every ACDC case). Model: ImageNet ResNet-18 applied to each slice -> 512-d vector per slice -> mean and max over
slices -> linear layer -> 5 classes. Augmentation (same transform for all slices of a patient): rotation, scaling,
mirroring, brightness/contrast/gamma, noise. AdamW + cosine schedule, fixed number of epochs (no early stopping on the
validation fold, which stays an honest out-of-fold estimate). Saves the model and probabilities for the validation fold
(out-of-fold) and the 50 test patients.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from acdc.augment import to_canvas
from acdc.common import GROUPS, RESULTS, WORK, set_seed

CROP = 256


def load_stack(pid):
    """(Z, 3, 256, 256) float16 tensor [ED, ES, ED-ES] per slice."""
    d = np.load(WORK / "cache" / f"{pid}.npz")
    ed, es = torch.from_numpy(d["img_ED"].astype(np.float32)), torch.from_numpy(d["img_ES"].astype(np.float32))
    x = torch.stack([ed, es, ed - es], 1)
    lo = torch.tensor([float(ed.min()), float(es.min()), 0.0]).view(1, 3, 1, 1)
    canvas = to_canvas(x - lo, 0.0, CROP) + lo  # pad with each channel's minimum (0 for the difference)
    return canvas.half()


class SliceNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = nn.Sequential(*list(r.children())[:-2])  # up to layer4 (B,512,8,8)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, len(GROUPS)))

    def slice_features(self, x):
        return F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)

    def forward(self, stacks):
        """stacks: list of (Z_i,3,H,W) tensors (one per patient) -> (B,5) logits."""
        z = torch.cat(stacks)
        f = self.slice_features(z)
        out, i = [], 0
        for s in stacks:
            fi = f[i:i + len(s)]
            out.append(torch.cat([fi.mean(0), fi.max(0).values]))
            i += len(s)
        return self.head(torch.stack(out))


def augment(x):
    """Same random geometric transform for all slices of one patient, then intensity changes. x: (Z,3,H,W) on GPU."""
    dev = x.device
    ang = (torch.rand(1).item() * 2 - 1) * math.pi
    sc = 0.8 + 0.4 * torch.rand(1).item()
    c, s = math.cos(ang) / sc, math.sin(ang) / sc
    fx, fy = (1 if torch.rand(1) < 0.5 else -1), (1 if torch.rand(1) < 0.5 else -1)
    tx, ty = (torch.rand(2) * 0.1 - 0.05).tolist()
    theta = torch.tensor([[c * fx, -s * fy, tx], [s * fx, c * fy, ty]], device=dev, dtype=torch.float32)
    grid = F.affine_grid(theta.expand(len(x), 2, 3), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    x = x * (0.8 + 0.4 * torch.rand(1, device=dev)) + 0.2 * (torch.rand(1, device=dev) - 0.5)
    if torch.rand(1) < 0.2:
        x = x + torch.randn_like(x) * 0.1 * torch.rand(1, device=dev)
    return x


@torch.no_grad()
def predict(model, stacks, bs=8):
    model.eval()
    out = []
    for i in range(0, len(stacks), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(model([s.cuda().float() for s in stacks[i:i + bs]]).float(), 1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="resnet18")
    args = ap.parse_args()
    set_seed(args.seed * 100 + args.fold)
    out = WORK / "runs" / f"cls_{args.tag}_s{args.seed}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    enc = {g: i for i, g in enumerate(GROUPS)}
    tr = meta[(meta.split == "train") & (meta.fold != args.fold)]
    va = meta[(meta.split == "train") & (meta.fold == args.fold)]
    te = meta[meta.split == "test"]
    stacks = {p: load_stack(p) for p in meta.pid}
    tr_x = [stacks[p].cuda() for p in tr.pid]
    tr_y = torch.tensor(tr.group.map(enc).values)

    model = SliceNet().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * math.ceil(len(tr_x) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    t0, log = time.time(), []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_x))
        losses = []
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = [augment(tr_x[j].float()) for j in idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb).float(), tr_y[idx].cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        log.append({"epoch": ep + 1, "loss": float(np.mean(losses))})
        if (ep + 1) % 20 == 0:
            pv = predict(model, [stacks[p] for p in va.pid])
            log[-1]["val_acc"] = float((pv.argmax(1) == va.group.map(enc).values).mean())
            print(json.dumps(log[-1]), flush=True)
    minutes = (time.time() - t0) / 60
    torch.save({"model": model.state_dict(), "args": vars(args), "train_minutes": minutes}, out / "model.pt")
    rows = []
    for split, df in (("cv", va), ("test", te)):
        p = predict(model, [stacks[q] for q in df.pid])
        rows.append(pd.DataFrame({"pid": df.pid.values, "eval": split, "group": df.group.values, "fold": args.fold,
                                  "seed": args.seed, **{f"p_{g}": p[:, i] for i, g in enumerate(GROUPS)}}))
    (RESULTS / "cls" / "cnn_runs").mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(RESULTS / "cls" / "cnn_runs" / f"{args.tag}_s{args.seed}_f{args.fold}.csv", index=False)
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    print(f"done in {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
