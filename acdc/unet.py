"""2D U-Net following the nnU-Net recipe (Isensee et al., Nature Methods 2021), written out so it can be explained.

encoder: 7 stages, each = 2 x (3x3 conv -> InstanceNorm -> LeakyReLU); stage s>0 starts with a stride-2 conv.
         channels 32, 64, 128, 256, 512, 512, 512  (a 256x256 patch ends at 4x4).
decoder: transposed conv (x2 upsampling) -> concatenate the encoder features of the same size -> 2 conv blocks.
deep supervision: a 1x1 conv head on the 5 highest-resolution decoder outputs; the loss is computed at every one of
         them (targets downsampled), weighted 1, 1/2, 1/4, ... This helps gradients reach the deep layers.
in_ch > 1 = 2.5D input (neighbouring slices as extra channels).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class Stage(nn.Sequential):
    def __init__(self, cin, cout, stride):
        super().__init__(ConvBlock(cin, cout, stride), ConvBlock(cout, cout))


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, features=(32, 64, 128, 256, 512, 512, 512), n_heads=5):
        super().__init__()
        self.enc = nn.ModuleList()
        c = in_ch
        for i, f in enumerate(features):
            self.enc.append(Stage(c, f, stride=1 if i == 0 else 2))
            c = f
        self.up, self.dec, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for f_skip in reversed(features[:-1]):  # from deepest to shallowest
            self.up.append(nn.ConvTranspose2d(c, f_skip, 2, stride=2))
            self.dec.append(Stage(2 * f_skip, f_skip, stride=1))
            self.heads.append(nn.Conv2d(f_skip, n_classes, 1))
            c = f_skip
        self.n_heads = n_heads
        self.deep_supervision = True  # switched off at inference: only the full-resolution output is returned

    def forward(self, x):
        skips = []
        for st in self.enc:
            x = st(x)
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec, head in zip(self.up, self.dec, self.heads):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            outs.append(head(x))
        if not self.deep_supervision:
            return outs[-1]
        return outs[::-1][: self.n_heads]  # full resolution first


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = UNet2D()
    y = m(torch.randn(2, 1, 256, 256))
    print([t.shape for t in y], f"{n_params(m) / 1e6:.1f} M params")
