"""Whole-slice inference for the 2D U-Net (used for validation during training, prediction, robustness and the demo)."""
import torch
import torch.nn.functional as F

from acdc.augment import neighbour_stack
from acdc.unet import UNet2D


@torch.no_grad()
def predict_volume(models, img, in_ch=1, tta=True, device="cuda", batch=16):
    """img: (Z,H,W) float tensor at 1.25 mm, normalised. Returns softmax probabilities (Z,4,H,W) on CPU, averaged over
    the given models and (if tta) over the 4 mirrorings of the slice. Slices are padded to a multiple of 64 (6 poolings)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    x = neighbour_stack(img.float(), in_ch)  # (Z,C,H,W)
    Z, C, H, W = x.shape
    ph, pw = (-H) % 64, (-W) % 64
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=float(img.min()))
    flips = [(), (2,), (3,), (2, 3)] if tta else [()]
    out = torch.zeros(Z, 4, H + ph, W + pw)
    for i in range(0, Z, batch):
        xb = x[i:i + batch].to(device)
        acc = 0
        for m in models:
            for f in flips:
                inp = xb.flip(f) if f else xb
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    logits = m(inp)
                p = torch.softmax(logits.float(), 1)
                acc = acc + (p.flip(f) if f else p)
        out[i:i + batch] = (acc / (len(models) * len(flips))).cpu()
    return out[:, :, ph // 2: ph // 2 + H, pw // 2: pw // 2 + W]


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = UNet2D(in_ch=ck["args"]["in_ch"])
    m.load_state_dict(ck["model"])
    m.deep_supervision = False
    return m.to(device).eval(), ck
