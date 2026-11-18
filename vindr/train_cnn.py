"""Step 2 - fine-tune an ImageNet DenseNet-121 on one label type.

  uv run python train_cnn.py --labels majority --seed 42
  uv run python train_cnn.py --labels soft --seed 42 --smoke   # 1 epoch on 500 images

Design:
  * The uint8 image cache for each split is copied to the GPU once (about 2.4 GB for train),
    so there is no CPU data loader. Augmentation (small rotation / shift / scale,
    brightness / contrast) is done on the GPU with affine_grid + grid_sample. No flips.
  * 15 sigmoid outputs (14 findings + Abnormal), BCE-with-logits on the chosen targets.
    Soft targets (0, 1/3, 2/3, 1) are used directly as BCE targets.
  * pos_weight = (#neg / #pos) from the train split, capped (default 10). It is computed from
    the MAJORITY labels for every run so the only thing that changes between runs is the target.
  * AdamW, linear warm-up + cosine schedule, bf16 autocast, channels_last.
  * Model selection: val macro-AUROC over the 14 findings, scored against MAJORITY labels.

Outputs in results/cnn_<labels>_s<seed>/ :
  best.pt, train.log, epochs.csv (per-epoch loss / val AUROC / time),
  preds_{val,test,disagree}.csv (image_id + 15 logits + 15 probabilities), run.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score

from prepare import FINDINGS, TARGETS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model():
    model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(TARGETS))
    return model


def to_input(x_uint8):
    """uint8 [B, H, W] on GPU -> normalised float [B, 3, H, W] (grayscale copied to 3 channels)."""
    x = x_uint8.float().div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


def augment(x_uint8):
    """Random affine (rotation +-7 deg, scale 0.9-1.1, shift +-5%) and brightness/contrast jitter, on GPU."""
    B = x_uint8.shape[0]
    dev = x_uint8.device
    x = x_uint8.float().div_(255.0).unsqueeze(1)  # [B,1,H,W] in [0,1]
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(7)
    scale = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.10
    shift = (torch.rand(B, 2, device=dev) * 2 - 1) * 0.10  # 0.10 in [-1,1] coords = 5% of width
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)  # [B,2,3]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    bright = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.08
    x = ((x - 0.5) * contrast + 0.5 + bright).clamp_(0, 1)
    x = x.expand(-1, 3, -1, -1)
    return ((x - MEAN.to(dev)) / STD.to(dev)).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict_logits(model, images_gpu, batch_size=128):
    model.eval()
    out = []
    for i in range(0, len(images_gpu), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(to_input(images_gpu[i:i + batch_size])).float())
    return torch.cat(out).cpu().numpy()


def macro_auroc(y_true, scores):
    """Mean AUROC over the 14 findings (classes with no positives in y_true are skipped)."""
    aucs = [roc_auc_score(y_true[:, k], scores[:, k]) for k in range(len(FINDINGS))
            if 0 < y_true[:, k].sum() < len(y_true)]
    return float(np.mean(aucs))


def save_preds(path, image_ids, logits):
    probs = 1 / (1 + np.exp(-logits))
    df = pd.DataFrame({"image_id": image_ids})
    for k, name in enumerate(TARGETS):
        df[f"logit_{name}"] = logits[:, k]
    for k, name in enumerate(TARGETS):
        df[name] = probs[:, k]
    df.to_csv(path, index=False, float_format="%.6f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["majority", "soft", "union"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pos_weight_cap", type=float, default=10.0, help="0 disables pos_weight")
    ap.add_argument("--patience", type=int, default=4, help="early stopping on val macro-AUROC")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 500 train / 300 val images")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--data_dir", default="/workspace/data")
    ap.add_argument("--results_dir", default="/workspace/results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    name = f"cnn_{args.labels}_s{args.seed}{args.tag}" + ("_smoke" if args.smoke else "")
    out_dir = os.path.join(args.results_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "train.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    # ---------------- data ----------------
    image_ids = np.loadtxt(os.path.join(args.data_dir, "image_ids.txt"), dtype=str)
    splits = pd.read_csv(os.path.join(args.data_dir, "splits.csv"))
    assert (splits.image_id.values == image_ids).all()
    targets = pd.read_csv(os.path.join(args.data_dir, f"labels_{args.labels}.csv"))[TARGETS].values.astype(np.float32)
    majority = pd.read_csv(os.path.join(args.data_dir, "labels_majority.csv"))[TARGETS].values.astype(np.float32)
    cache = np.load(os.path.join(args.data_dir, "images_512.npy"), mmap_mode="r")

    idx = {s: np.where(splits.split.values == s)[0] for s in ["train", "val", "test", "disagree"]}
    if args.smoke:
        rng = np.random.default_rng(0)
        idx["train"] = np.sort(rng.choice(idx["train"], 500, replace=False))
        idx["val"] = np.sort(rng.choice(idx["val"], 300, replace=False))
        args.epochs = 1

    t0 = time.time()
    gpu_imgs = {s: torch.from_numpy(np.ascontiguousarray(cache[i])).to(dev) for s, i in idx.items()}
    log(f"loaded images to GPU in {time.time() - t0:.0f}s: " + ", ".join(f"{s}={len(i)}" for s, i in idx.items()))
    y_train = torch.from_numpy(targets[idx["train"]]).to(dev)

    # pos_weight from the train split, majority labels, capped
    if args.pos_weight_cap > 0:
        pos = majority[idx["train"]].sum(0)
        neg = len(idx["train"]) - pos
        pos_weight = torch.tensor(np.minimum(neg / np.maximum(pos, 1), args.pos_weight_cap), dtype=torch.float32, device=dev)
    else:
        pos_weight = None
    log(f"args: {vars(args)}")
    log(f"pos_weight: {None if pos_weight is None else np.round(pos_weight.cpu().numpy(), 2).tolist()}")

    # ---------------- model / optimiser ----------------
    model = build_model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(idx["train"]) // args.batch_size
    total = steps_per_epoch * args.epochs
    warmup = min(200, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_epoch, bad_epochs, history = -1.0, -1, 0, []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        perm = torch.randperm(len(idx["train"]), device=dev)
        running = 0.0
        for step in range(steps_per_epoch):
            b = perm[step * args.batch_size:(step + 1) * args.batch_size]
            x = augment(gpu_imgs["train"][b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits.float(), y_train[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() if step % 20 == 0 else 0  # cheap progress estimate
        torch.cuda.synchronize()
        train_time = time.time() - t_ep

        val_logits = predict_logits(model, gpu_imgs["val"])
        val_auc = macro_auroc(majority[idx["val"]], val_logits)
        val_abn = roc_auc_score(majority[idx["val"], 14], val_logits[:, 14])
        ep_time = time.time() - t_ep
        history.append({"epoch": epoch, "train_loss_est": running / max(1, math.ceil(steps_per_epoch / 20)),
                        "val_macro_auroc": val_auc, "val_abnormal_auroc": val_abn,
                        "train_sec": round(train_time, 1), "epoch_sec": round(ep_time, 1),
                        "imgs_per_sec": round(steps_per_epoch * args.batch_size / train_time, 1)})
        log(f"epoch {epoch}/{args.epochs} loss~{history[-1]['train_loss_est']:.4f} "
            f"val macroAUROC {val_auc:.4f} abnormal {val_abn:.4f} | {ep_time:.0f}s "
            f"({history[-1]['imgs_per_sec']} img/s)")
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "epochs.csv"), index=False)

        if val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "labels": args.labels, "seed": args.seed,
                        "epoch": epoch, "val_macro_auroc": val_auc, "targets": TARGETS},
                       os.path.join(out_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"early stopping (no val improvement for {args.patience} epochs)")
                break
    train_minutes = (time.time() - train_start) / 60

    # ---------------- predictions with the best checkpoint ----------------
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)["state_dict"])
    for s in ["val", "test", "disagree"]:
        save_preds(os.path.join(out_dir, f"preds_{s}.csv"), image_ids[idx[s]], predict_logits(model, gpu_imgs[s]))
    info = {"name": name, "labels": args.labels, "seed": args.seed, "best_epoch": best_epoch,
            "best_val_macro_auroc": best_auc, "epochs_run": len(history), "train_gpu_minutes": round(train_minutes, 2),
            "n_params": sum(p.numel() for p in model.parameters()), "gpu": torch.cuda.get_device_name(),
            "args": vars(args)}
    json.dump(info, open(os.path.join(out_dir, "run.json"), "w"), indent=2)
    log(f"done: best epoch {best_epoch}, val macroAUROC {best_auc:.4f}, {train_minutes:.1f} GPU-min")


if __name__ == "__main__":
    main()
