"""Switchable MISR fusion model.

2x2 factor = {learned dt conditioning} x {learned cloud-reliability q conditioning}.

Round-3 architecture changes (targeted at the diagnosed failure: the model could not beat
a zero-training cloud-aware weighted average):

  att_mode="pixel" (new default in `build_model`)
      The temporal aggregation is a PER-PIXEL softmax over frames instead of one scalar
      weight per frame.  A single global weight per frame cannot express "down-weight this
      frame only where it is cloudy", which is exactly what cloud removal requires; the
      trivial baseline is per-pixel weighted, so the old model was structurally dominated.
      The gate sees the per-frame features plus (only when the corresponding factor is
      enabled) the cloud fraction and the time distance -- keeping the 2x2 attribution
      clean: a gate without cld cannot use cloud information.

  base_resid (optional)
      Adds a fixed, non-learned cloud-aware weighted mean of the LR frames, bicubic
      upsampled, as a residual base.  It is OFF by default because it would hand the cloud
      mask and dt to every arm and break the factorial; turn it on (`--base-resid`) as a
      guarantee-the-floor option.

  frame mode ("frame") is kept for backward compatibility with the PROBA-V/MuS2 runs.
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


class MISRNet(nn.Module):
    def __init__(self, cin=1, c=32, scale=3, use_dt=False, use_q=False, n_blocks=2,
                 att_mode="frame", base_resid=False, base_tau=30.0, base_p=2.0,
                 head_init=0.0, dt_mode="add"):
        super().__init__()
        self.cin, self.c, self.scale = cin, c, scale
        self.use_dt, self.use_q = use_dt, use_q
        self.att_mode = att_mode
        self.dt_mode = dt_mode
        self.base_resid = base_resid
        self.base_tau, self.base_p = base_tau, base_p
        self.stem = nn.Conv2d(cin, c, 3, padding=1)
        self.enc = nn.Sequential(*[ConvBlock(c, c) for _ in range(n_blocks)])
        if use_dt:
            if dt_mode == "add":
                # LEGACY: raw day count (0..hundreds) added straight into the features.
                # Measured to be harmful (-0.15..-0.24 dB), probably an embedding-scale issue.
                self.dt_mlp = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True), nn.Linear(c, c))
            elif dt_mode == "film":
                # FiLM on the *normalised* time distance; zero-initialised so the arm starts
                # exactly where the no-dt arm starts (no dead branch: f itself is non-zero).
                self.dt_gam = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True), nn.Linear(c, c))
                self.dt_bet = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True), nn.Linear(c, c))
                for m in (self.dt_gam[-1], self.dt_bet[-1]):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            # dt_mode == "gate": dt enters the attention gate only, never the features.
        if use_q:
            self.q_mlp = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True), nn.Linear(c, c))
            self.q_gate = nn.Parameter(torch.zeros(1))
        if att_mode == "pixel":
            g_in = c + (1 if use_q else 0) + (1 if use_dt else 0)
            self.att_conv = nn.Sequential(
                nn.Conv2d(g_in, c, 1), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1))
            nn.init.zeros_(self.att_conv[-1].weight)     # start from a uniform average
            nn.init.zeros_(self.att_conv[-1].bias)
        else:
            self.att_mlp = nn.Sequential(nn.Linear(c, max(c // 2, 8)), nn.ReLU(inplace=True),
                                         nn.Linear(max(c // 2, 8), 1))
        self.fuse = nn.Sequential(ConvBlock(c, c))
        self.head = nn.Sequential(nn.Conv2d(c, c * scale * scale, 3, padding=1),
                                  nn.PixelShuffle(scale), nn.Conv2d(c, cin, 3, padding=1))
        if head_init:
            # small random init (NOT exact zeros: an exactly-zero last conv would block the
            # encoder's gradient on step 1 -- the project's "double zero" lesson)
            nn.init.normal_(self.head[-1].weight, std=head_init)
            nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _dt_norm(dt):
        """Signed, bounded, monotone normalisation of a time distance in days.

        log1p(|dt|/30) keeps the input to the dt branches O(1) instead of O(100),
        which is what the legacy raw-day embedding fed to the network.
        """
        return torch.log1p(dt.abs() / 30.0) * dt.sign()

    def _base_image(self, lr, dt, cld):
        """Fixed cloud-aware weighted mean in the LR domain (no parameters)."""
        B, T, C, H, W = lr.shape
        w = torch.ones(B, T, 1, H, W, device=lr.device, dtype=lr.dtype)
        if dt is not None:
            w = w * torch.exp(-torch.abs(dt).reshape(B, T, 1, 1, 1) / self.base_tau)
        if cld is not None:
            w = w * torch.clamp(1.0 - cld.reshape(B, T, 1, H, W), min=1e-3) ** self.base_p
        base = (lr * w).sum(1) / w.sum(1).clamp_min(1e-6)
        return F.interpolate(base, scale_factor=self.scale, mode="bicubic",
                             align_corners=False)

    def forward(self, lr, q=None, cld=None, dt=None):
        # lr: B,T,C,H,W ; q: B,T,1 ; cld: B,T,1,H,W ; dt: B,T
        B, T, C, H, W = lr.shape
        x = lr.reshape(B * T, C, H, W)
        f = self.enc(self.stem(x))                      # B*T,c,H,W
        if self.use_q:
            if cld is not None:
                f = f * (1.0 - torch.sigmoid(self.q_gate) * cld.reshape(B * T, 1, H, W))
            if q is not None:
                emb = self.q_mlp(q.reshape(B * T, 1)).unsqueeze(-1).unsqueeze(-1)
                f = f + emb
        if self.use_dt and dt is not None:
            if self.dt_mode == "add":
                emb = self.dt_mlp(dt.reshape(B * T, 1)).unsqueeze(-1).unsqueeze(-1)
                f = f + emb
            elif self.dt_mode == "film":
                dtn = self._dt_norm(dt).reshape(B * T, 1)
                gam = self.dt_gam(dtn).unsqueeze(-1).unsqueeze(-1)
                bet = self.dt_bet(dtn).unsqueeze(-1).unsqueeze(-1)
                f = f * (1.0 + gam) + bet
            # "gate": no feature-space injection at all
        f = f.reshape(B, T, self.c, H, W)
        fused = f.mean(1)

        if self.att_mode == "pixel":
            parts = [f]
            if self.use_q and cld is not None:
                parts.append(cld.reshape(B, T, 1, H, W).expand(B, T, 1, H, W))
            if self.use_dt and dt is not None:
                parts.append(self._dt_norm(dt).reshape(B, T, 1, 1, 1).expand(B, T, 1, H, W))
            g = torch.cat(parts, dim=2).reshape(B * T, -1, H, W)
            logits = self.att_conv(g).reshape(B, T, 1, H, W)
            w = torch.softmax(logits, dim=1)                     # per-pixel over frames
            agg = (w * f).sum(1)
            self.last_weights = w.detach()
        else:
            desc = f.mean(dim=(3, 4))
            w = torch.softmax(self.att_mlp(desc), dim=1)
            agg = (w.unsqueeze(-1).unsqueeze(-1) * f).sum(1)
            self.last_weights = w.detach()

        out = self.head(self.fuse(fused + agg))
        if self.base_resid:
            out = out + self._base_image(lr, dt if self.use_dt else None,
                                         cld if self.use_q else None)
        return out


def build_model(cin=1, c=32, scale=3, use_dt=False, use_q=False, n_blocks=2,
                att_mode="frame", base_resid=False, head_init=0.0, dt_mode="add"):
    return MISRNet(cin=cin, c=c, scale=scale, use_dt=use_dt, use_q=use_q,
                   n_blocks=n_blocks, att_mode=att_mode, base_resid=base_resid,
                   head_init=head_init, dt_mode=dt_mode)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
