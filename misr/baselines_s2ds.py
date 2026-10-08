"""E2 -- published multi-image SR baselines, re-implemented on S2DS.

Why this file exists: the 2x2 compares four variants of ONE architecture against each
other.  That says how much dt and q are worth *inside our network*, and says nothing about
whether the network itself is competitive.  These two baselines give the absolute anchor.

  HighResNet   Deudon et al., "HighRes-net: Recursive fusion for multi-frame
               super-resolution of satellite imagery", CVPR EarthVision 2018.
               Shared per-frame encoder -> recursive PAIRWISE fusion -> upsampling.
  RAMS         Salvetti et al., "Multi-image super-resolution of remotely sensed images
               using residual attention deep neural networks", Remote Sensing 12(14):2207,
               2020.  Each LR frame is BICUBIC-UPSAMPLED FIRST and encoded at HR
               resolution; per-pixel attention over frames; residual reconstruction.

Adaptations, stated because they are departures from the originals:
  * Both are cloud-blind and time-blind: they receive only the LR stack, exactly like our
    arm A.  They cannot use cld or dt, which is the point -- the anchor tells us what a
    competent fusion network achieves WITHOUT the two factors under study.
  * The explicit optical-flow / registration stage of HighRes-net is omitted.  Sentinel-2
    L2A tiles of the same AOI are co-registered to sub-pixel accuracy by the provider, so
    the registration stage has little to do here; the encoder sees the whole stack anyway.
  * Channel width is tuned so that both baselines land near our arms' 208k parameters
    (HighResNet c=40 -> ~193k, RAMS c=40 -> ~195k), so the anchor is not just a
    parameter-count artefact.
  * Both add a parameter-free global skip = bicubic of the temporal mean of the LR stack.
    Neither our arms nor these baselines can reach the reported accuracy without it, and
    it uses no cloud and no date information.

The target date's own LR frame is NOT in the stack (build_dataset.py excludes it), so
every method here is doing temporal interpolation + 4x SR jointly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=k // 2), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, k, padding=k // 2))

    def forward(self, x):
        return x + self.body(x)


def _up_mean(lr, scale):
    """Bicubic upsample of the temporal mean -- the parameter-free global skip."""
    return F.interpolate(lr.mean(1), scale_factor=scale, mode="bicubic",
                         align_corners=False)


class HighResNet(nn.Module):
    """Recursive pairwise fusion in the LR domain, then two x2 pixel-shuffle stages."""

    def __init__(self, cin=4, c=40, scale=4, n_blocks=2):
        super().__init__()
        assert scale == 4, "two x2 stages assume scale 4"
        self.cin, self.c, self.scale = cin, c, scale
        self.stem = nn.Conv2d(cin, c, 3, padding=1)
        self.enc = nn.Sequential(*[ConvBlock(c, c) for _ in range(n_blocks)])
        # fusion block: takes the concatenation of two feature maps, returns one
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1))
        self.up1 = nn.Sequential(nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))
        self.rec = nn.Sequential(*[ConvBlock(c, c) for _ in range(2)])
        self.head = nn.Conv2d(c, cin, 3, padding=1)

    def _fuse_pair(self, a, b):
        return 0.5 * (a + b) + self.fuse(torch.cat([a, b], dim=1))

    def _stack(self, lr, cld=None):
        """Input channels fed to the stem; subclasses may append side channels."""
        return lr

    def forward(self, lr, q=None, cld=None, dt=None):
        B, T, C, H, W = lr.shape
        x = self._stack(lr, cld)
        f = self.enc(self.stem(x.reshape(B * T, x.shape[2], H, W)))
        f = f.reshape(B, T, self.c, H, W)
        # recursive pairwise fusion; an odd frame is carried to the next round untouched
        cur = [f[:, i] for i in range(T)]
        while len(cur) > 1:
            nxt = [self._fuse_pair(cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt
        x = cur[0]
        x = self.up2(self.up1(x))
        x = self.rec(x)
        return self.head(x) + _up_mean(lr, self.scale)


class RAMS(nn.Module):
    """Bicubic upsample first, encode at HR, per-pixel attention over frames, residual."""

    def __init__(self, cin=4, c=40, scale=4, n_blocks=2, n_res=4, T=12):
        super().__init__()
        self.cin, self.c, self.scale, self.T = cin, c, scale, T
        self.stem = nn.Conv2d(cin, c, 3, padding=1)
        self.enc = nn.Sequential(*[ConvBlock(c, c) for _ in range(n_blocks)])
        self.n_res = n_res
        self.res = nn.Sequential(*[ConvBlock(c, c) for _ in range(n_res)])
        self.head = nn.Conv2d(c, cin, 3, padding=1)
        # the attention mixing width depends on T, so it is built here (not lazily inside
        # forward): a submodule created during the first forward would never reach the
        # optimizer that was constructed before that forward.
        self.attn = nn.Sequential(
            nn.Conv2d(T * c, c, 1), nn.ReLU(inplace=True), nn.Conv2d(c, T, 1))
        nn.init.zeros_(self.attn[-1].weight)     # start from a uniform average over frames
        nn.init.zeros_(self.attn[-1].bias)

    def _attn(self, f):
        """f: B,T,c,H,W -> per-pixel softmax weights over T."""
        B, T, c, H, W = f.shape
        g = f.permute(0, 2, 1, 3, 4).reshape(B, T * c, H, W)
        logits = self.attn(g).reshape(B, T, 1, H, W)
        return torch.softmax(logits, dim=1)

    def forward(self, lr, q=None, cld=None, dt=None):
        B, T, C, H, W = lr.shape
        up = F.interpolate(lr.reshape(B * T, C, H, W), scale_factor=self.scale,
                           mode="bicubic", align_corners=False)
        f = self.enc(self.stem(up))                       # B*T,c,H*4,W*4
        _, c, h2, w2 = f.shape
        f = f.reshape(B, T, c, h2, w2)
        w = self._attn(f)
        agg = (w * f).sum(1)
        out = self.head(self.res(agg))
        self.last_weights = w.detach()
        return out + _up_mean(lr, self.scale)


class HighResNetCld(HighResNet):
    """HighRes-net fed with the per-pixel cloud mask as an extra input channel.

    Reviewer control: the published baselines are cloud-blind, so half of their failure
    on S2DS is decided by the data construction.  This variant receives exactly the same
    cloud mask our q-armed models see (as a fifth input channel per frame) and nothing
    else -- no gate, no reliability reweighting -- so that the comparison against our
    arms isolates *how* the mask is used, not *whether* it is available.
    Only the stem conv changes (cin+1 -> +c*9 parameters, ~0.2%).
    """

    def __init__(self, cin=4, c=40, scale=4, n_blocks=2):
        super().__init__(cin=cin, c=c, scale=scale, n_blocks=n_blocks)
        self.stem = nn.Conv2d(cin + 1, c, 3, padding=1)

    def _stack(self, lr, cld=None):
        return torch.cat([lr, cld.unsqueeze(2).float()], dim=2)


BUILDERS = {"highresnet": HighResNet, "rams": RAMS, "highresnet_cld": HighResNetCld}


def build_baseline(name, cin=4, c=40, scale=4, **kw):
    return BUILDERS[name](cin=cin, c=c, scale=scale, **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
