"""Step 2 - train the 2D U-Net on one cross-validation fold.

    python -m acdc.train_seg --cfg unet2d --fold 0              # main configuration
    python -m acdc.train_seg --cfg unet25d --in_ch 3 --fold 0   # 2.5D ablation (3 neighbouring slices as input)
    python -m acdc.train_seg --cfg unet2d_ce --loss ce --fold 0 # loss ablation (cross-entropy only)
    python -m acdc.train_seg --cfg unet2d_noaug --aug none ...  # augmentation ablation

Training set = the 80 training patients outside the fold (ED + ES slices); the 20 patients of the fold are the
validation set. The test patients (101-150) are never loaded here. Recipe (nnU-Net defaults): Dice + cross-entropy loss
with deep supervision, SGD (Nesterov, momentum 0.99), lr 0.01 with polynomial decay, 250 iterations per epoch, batch 32
patches of 256x256. The model of the LAST epoch is kept (no checkpoint picking on the validation fold, so the validation
fold stays an honest out-of-fold estimate).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from acdc.augment import Sampler, neighbour_stack, to_canvas
from acdc.common import CLASSES, RESULTS, WORK, dice, set_seed
from acdc.infer import predict_volume
from acdc.unet import UNet2D, n_params


def load_patients(pids):
    """Returns per-volume list of (img (Z,H,W) float tensor, lab (Z,H,W) uint8 tensor) for ED and ES."""
    vols = []
    for pid in pids:
        d = np.load(WORK / "cache" / f"{pid}.npz")
        for ph in ("ED", "ES"):
            vols.append((pid, ph, torch.from_numpy(d[f"img_{ph}"].astype(np.float32)), torch.from_numpy(d[f"lab_{ph}"])))
    return vols


def dc_ce_loss(logits, target, use_dice=True):
    """Cross-entropy + soft Dice over the 3 foreground classes, Dice computed over the whole batch ("batch Dice")."""
    ce = F.cross_entropy(logits, target)
    if not use_dice:
        return ce
    p = torch.softmax(logits.float(), 1)[:, 1:]
    oh = F.one_hot(target, 4).permute(0, 3, 1, 2)[:, 1:].float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return ce + (1 - (2 * inter + 1e-5) / (den + 1e-5)).mean()


def deep_supervision_loss(outs, target, use_dice):
    w = torch.tensor([1 / 2 ** i for i in range(len(outs))])
    w = w / w.sum()
    loss = 0
    for wi, o in zip(w, outs):
        t = target if o.shape[-1] == target.shape[-1] else \
            F.interpolate(target[:, None].float(), size=o.shape[-2:], mode="nearest")[:, 0].long()
        loss = loss + wi * dc_ce_loss(o, t, use_dice)
    return loss


@torch.no_grad()
def validate(model, vols, in_ch):
    """Mean Dice per structure over validation volumes, at 1.25 mm (monitoring only; final metrics use the original grid)."""
    model.deep_supervision = False
    model.eval()
    torch.backends.cudnn.benchmark = False  # every volume has another size: don't re-benchmark kernels for each
    d = {n: [] for n in CLASSES.values()}
    for _, _, img, lab in vols:
        seg = predict_volume(model, img, in_ch, tta=False).argmax(1).numpy()
        for c, n in CLASSES.items():
            d[n].append(dice(seg == c, lab.numpy() == c))
    model.deep_supervision = True
    model.train()
    torch.backends.cudnn.benchmark = True
    return {n: float(np.mean(v)) for n, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="unet2d")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--in_ch", type=int, default=1)
    ap.add_argument("--loss", choices=["dce", "ce"], default="dce")
    ap.add_argument("--aug", choices=["full", "none"], default="full")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--n_train", type=int, default=0, help="use only the first n training patients (smoke tests)")
    args = ap.parse_args()
    set_seed(args.seed + 1000 * args.fold)
    torch.backends.cudnn.benchmark = True
    out = WORK / "runs" / f"{args.cfg}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(RESULTS / "acdc_meta.csv")
    tr_ids = meta[(meta.split == "train") & (meta.fold != args.fold)].pid.tolist()
    va_ids = meta[(meta.split == "train") & (meta.fold == args.fold)].pid.tolist()
    if args.n_train:
        tr_ids = tr_ids[: args.n_train]
    tr_vols, va_vols = load_patients(tr_ids), load_patients(va_ids)

    # all training slices -> one GPU canvas tensor
    imgs, labs = [], []
    for _, _, img, lab in tr_vols:
        imgs.append(to_canvas(neighbour_stack(img, args.in_ch), float(img.min())).half())
        labs.append(to_canvas(lab, 0))
    sampler = Sampler(torch.cat(imgs), torch.cat(labs), "cuda", augment=args.aug == "full")
    print(f"fold {args.fold}: {len(tr_ids)} train patients ({len(sampler.imgs)} slices), {len(va_ids)} val patients", flush=True)

    model = UNet2D(in_ch=args.in_ch).cuda().to(memory_format=torch.channels_last)
    fwd = torch.compile(model)  # ~25% faster training step (fuses InstanceNorm etc.); validation uses the plain module
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    total = args.epochs * args.iters
    log, t0, step = [], time.time(), 0
    for ep in range(args.epochs):
        te, losses = time.time(), []
        for _ in range(args.iters):
            lr = args.lr * (1 - step / total) ** 0.9  # polynomial decay
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = sampler.batch(args.bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = fwd(x.contiguous(memory_format=torch.channels_last))
            loss = deep_supervision_loss([o.float() for o in outs], y, args.loss == "dce")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
            losses.append(loss.item())
            step += 1
        row = {"epoch": ep + 1, "loss": float(np.mean(losses)), "lr": lr, "sec": time.time() - te}
        if (ep + 1) % args.val_every == 0 or ep + 1 == args.epochs:
            row.update({f"val_dice_{k}": v for k, v in validate(model, va_vols, args.in_ch).items()})
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args), "train_pids": tr_ids, "val_pids": va_ids,
                "n_params": n_params(model), "train_minutes": (time.time() - t0) / 60}, out / "model.pt")
    (RESULTS / "seg_runs").mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(RESULTS / "seg_runs" / f"{args.cfg}_f{args.fold}.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
