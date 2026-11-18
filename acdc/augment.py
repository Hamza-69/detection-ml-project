"""GPU data sampling + augmentation for 2D segmentation training (nnU-Net-style, simplified).

All training slices live on the GPU as a (N, C, 400, 400) canvas (1.25 mm pixels, heart roughly central). A batch is made
by one `grid_sample` call that crops a 256x256 patch and applies rotation / scaling / mirroring at the same time, then
intensity augmentations are applied per sample. Probabilities follow nnU-Net's defaults.
"""
import math

import torch
import torch.nn.functional as F

CANVAS = 400
PATCH = 256


def to_canvas(vol: torch.Tensor, value: float, size=CANVAS) -> torch.Tensor:
    """Centre a (..., H, W) tensor on a size x size canvas (pad with `value`, or centre-crop if larger)."""
    H, W = vol.shape[-2:]
    out = torch.full((*vol.shape[:-2], size, size), value, dtype=vol.dtype)
    ys, xs = max(0, (H - size) // 2), max(0, (W - size) // 2)  # crop offsets (if larger)
    yd, xd = max(0, (size - H) // 2), max(0, (size - W) // 2)  # paste offsets (if smaller)
    h, w = min(H, size), min(W, size)
    out[..., yd:yd + h, xd:xd + w] = vol[..., ys:ys + h, xs:xs + w]
    return out


def neighbour_stack(vol: torch.Tensor, in_ch: int) -> torch.Tensor:
    """(Z,H,W) -> (Z,in_ch,H,W): slice z plus its neighbours (edge slices repeated). in_ch=1 is plain 2D."""
    k = in_ch // 2
    idx = torch.arange(vol.shape[0])
    return torch.stack([vol[(idx + d).clamp(0, vol.shape[0] - 1)] for d in range(-k, k + 1)], 1)


def _gaussian_blur(x, sigma):
    """Blur one sample (C,H,W) with a separable Gaussian of std `sigma` pixels."""
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[0]
    x = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x[0]


class Sampler:
    def __init__(self, imgs, labs, device, augment=True, fg_prob=1 / 3):
        """imgs: (N,C,400,400) float16, labs: (N,400,400) uint8 (CPU tensors)."""
        self.imgs, self.labs = imgs.to(device), labs.to(device)
        self.device, self.augment, self.fg_prob = device, augment, fg_prob
        # foreground pixel coordinates of every slice, used to centre 1/3 of the patches on the heart (class balance)
        self.fg = [torch.nonzero(l > 0) for l in self.labs]
        self.has_fg = torch.tensor([len(f) > 0 for f in self.fg])

    def batch(self, B):
        dev = self.device
        idx = torch.randint(len(self.imgs), (B,))
        # patch centres in canvas pixel coordinates
        centre = torch.empty(B, 2)
        for b in range(B):
            i = int(idx[b])
            if torch.rand(1) < self.fg_prob and self.has_fg[i]:
                centre[b] = self.fg[i][torch.randint(len(self.fg[i]), (1,))].float().cpu()[0]
            else:
                centre[b] = torch.rand(2) * (CANVAS - PATCH) + PATCH / 2
        # affine map from output patch coordinates ([-1,1]) to canvas coordinates ([-1,1])
        A = torch.eye(2).repeat(B, 1, 1) * (PATCH / CANVAS)
        if self.augment:
            rot = torch.rand(B) < 0.2
            ang = torch.where(rot, (torch.rand(B) * 2 - 1) * math.pi, torch.zeros(B))
            sc = torch.where(torch.rand(B) < 0.2, 0.7 + 0.7 * torch.rand(B), torch.ones(B))  # zoom 0.7-1.4
            R = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1), torch.stack([ang.sin(), ang.cos()], -1)], 1)
            A = A @ R / sc[:, None, None]
            flip = torch.where(torch.rand(B, 2) < 0.5, -1.0, 1.0)  # mirror each axis with p=0.5
            A = A * flip[:, None, :]
        # grid_sample's (x, y) = (last dim, second-to-last dim); centre is (row, col)
        t = (2 * torch.stack([centre[:, 1], centre[:, 0]], 1) + 1) / CANVAS - 1  # pixel -> normalised (align_corners=False)
        theta = torch.cat([A, t[:, :, None]], 2).to(dev)
        img = self.imgs[idx.to(dev)].float()
        lab = self.labs[idx.to(dev)].float()[:, None]
        grid = F.affine_grid(theta, (B, 1, PATCH, PATCH), align_corners=False)
        img = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
        lab = F.grid_sample(lab, grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
        if self.augment:
            img = self.intensity(img)
        return img, lab

    @torch.no_grad()
    def intensity(self, x):
        B = x.shape[0]
        dev = x.device
        r = lambda: torch.rand(B, device=dev)  # noqa: E731
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=dev)  # noqa: E731
        sel = lambda p: (r() < p).float()[:, None, None, None]  # noqa: E731
        # additive Gaussian noise (variance 0-0.1)
        x = x + sel(0.1) * torch.randn_like(x) * u(0, 0.1).sqrt()
        # Gaussian blur, sigma 0.5-1 px
        for b in torch.nonzero(r() < 0.2).flatten().tolist():
            x[b] = _gaussian_blur(x[b], float(0.5 + 0.5 * torch.rand(1)))
        # brightness (multiplicative)
        x = x * (1 + sel(0.15) * (u(0.75, 1.25) - 1))
        # contrast: stretch around the mean, keep the original range
        m = sel(0.15)
        mean = x.mean((1, 2, 3), keepdim=True)
        lo, hi = x.amin((1, 2, 3), keepdim=True), x.amax((1, 2, 3), keepdim=True)
        x = m * ((x - mean) * u(0.75, 1.25) + mean).clamp(lo, hi) + (1 - m) * x
        # low-resolution simulation: downsample by 1-2x (nearest) and upsample back (bicubic)
        for b in torch.nonzero(r() < 0.25).flatten().tolist():
            s = int(PATCH * float(0.5 + 0.5 * torch.rand(1)))
            small = F.interpolate(x[b:b + 1], size=(s, s), mode="nearest")
            x[b] = F.interpolate(small, size=(PATCH, PATCH), mode="bicubic", align_corners=False)[0]
        # gamma 0.7-1.5 on the [0,1]-rescaled image (sometimes on the inverted image), original range restored
        for p_g, inv in ((0.3, False), (0.1, True)):
            m = sel(p_g)
            lo, hi = x.amin((1, 2, 3), keepdim=True), x.amax((1, 2, 3), keepdim=True)
            z = (x - lo) / (hi - lo + 1e-8)
            z = 1 - z if inv else z
            g = torch.where(torch.rand(B, 1, 1, 1, device=dev) < 0.5, u(0.7, 1.0), u(1.0, 1.5))
            z = z.clamp(min=0) ** g
            z = 1 - z if inv else z
            x = m * (z * (hi - lo) + lo) + (1 - m) * x
        return x
