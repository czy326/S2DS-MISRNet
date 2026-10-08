"""E1 -- internal ablation of the cloud-reliability factor q.

arm B currently spends its cloud budget through three routes at once:

  (i)   q_suppress      f <- f * (1 - sigmoid(q_gate) * cld)     PER-PIXEL feature suppression
  (ii)  q_attn_channel  cld is concatenated into the per-pixel attention conv   PER-PIXEL
  (iii) q_frame         f <- f + q_mlp(q),  q = 1 - mean(cld)    FRAME-LEVEL embedding

E1 turns (iii) off (arm B1 = per-pixel only) and turns (i)+(ii) off (arm B2 = frame-level
only), so that the gain measured for arm B can be attributed to a route instead of to
"cloud information" in the abstract.  It is the ablative counterpart of E5, which does the
same split by scrambling the mask instead of by surgery on the graph.

This module is a COPY of misr/models.py with three extra switches.  It is kept separate
deliberately: the E3/E5 training lanes import misr.models, and editing that file while
they are running would silently change the arms they train.
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


class MISRNetE1(nn.Module):
    """Same topology as MISRNet; the q routes are individually switchable."""

    def __init__(self, cin=1, c=32, scale=3, use_dt=False, use_q=False, n_blocks=2,
                 att_mode="pixel", base_resid=False, base_tau=30.0, base_p=2.0,
                 head_init=0.0, dt_mode="add",
                 q_suppress=True, q_attn_channel=True, q_frame=True, gamma0=0.0):
        super().__init__()
        self.cin, self.c, self.scale = cin, c, scale
        self.use_dt, self.use_q = use_dt, use_q
        self.att_mode = att_mode
        self.dt_mode = dt_mode
        self.base_resid = base_resid
        self.base_tau, self.base_p = base_tau, base_p
        self.q_suppress = bool(q_suppress) and bool(use_q)
        self.q_attn_channel = bool(q_attn_channel) and bool(use_q)
        self.q_frame = bool(q_frame) and bool(use_q)
        self.stem = nn.Conv2d(cin, c, 3, padding=1)
        self.enc = nn.Sequential(*[ConvBlock(c, c) for _ in range(n_blocks)])
        if use_dt:
            if dt_mode == "add":
                self.dt_mlp = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True),
                                            nn.Linear(c, c))
            elif dt_mode == "film":
                self.dt_gam = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True),
                                            nn.Linear(c, c))
                self.dt_bet = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True),
                                            nn.Linear(c, c))
                for m in (self.dt_gam[-1], self.dt_bet[-1]):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
        if self.q_frame:
            self.q_mlp = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True),
                                       nn.Linear(c, c))
        if self.q_suppress:
            # gamma0 = 0 in every main run, i.e. sigmoid(gamma) = 0.5 -- an initial HALF
            # suppression, not an identity.  gamma0 is exposed only for the E8 sensitivity
            # scan; the default reproduces the main runs bit for bit (torch.full of 0.0
            # is torch.zeros).
            self.q_gate = nn.Parameter(torch.full((1,), float(gamma0)))
        if att_mode == "pixel":
            g_in = c + (1 if self.q_attn_channel else 0) + (1 if use_dt else 0)
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
            nn.init.normal_(self.head[-1].weight, std=head_init)
            nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _dt_norm(dt):
        return torch.log1p(dt.abs() / 30.0) * dt.sign()

    def _base_image(self, lr, dt, cld):
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
        B, T, C, H, W = lr.shape
        x = lr.reshape(B * T, C, H, W)
        f = self.enc(self.stem(x))
        if self.use_q:
            if self.q_suppress and cld is not None:
                f = f * (1.0 - torch.sigmoid(self.q_gate) * cld.reshape(B * T, 1, H, W))
            if self.q_frame and q is not None:
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
        f = f.reshape(B, T, self.c, H, W)
        fused = f.mean(1)

        if self.att_mode == "pixel":
            parts = [f]
            if self.q_attn_channel and cld is not None:
                parts.append(cld.reshape(B, T, 1, H, W).expand(B, T, 1, H, W))
            if self.use_dt and dt is not None:
                parts.append(self._dt_norm(dt).reshape(B, T, 1, 1, 1).expand(B, T, 1, H, W))
            g = torch.cat(parts, dim=2).reshape(B * T, -1, H, W)
            logits = self.att_conv(g).reshape(B, T, 1, H, W)
            w = torch.softmax(logits, dim=1)
            agg = (w * f).sum(1)
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


# The paper's 2x2 arms A/B/C/D plus the two q-route ablations B1/B2.
#
# NOTE (2026-09-27): this dict previously held only B/B1/B2 while build_model_e1
# hardcoded use_dt=False -- a stale copy left over from the E1 ablation work.  The
# checkpoint args of the main arms record arm='A'/'C'/'D', so those runs could no longer
# be rebuilt from their own checkpoints (KeyError on the arm, and no dt branch at all).
# Restored: use_dt/use_q now live in the route table, and every route is reconstructed
# from the checkpoint with strict load_state_dict (dataset/_check_arm_rebuild.py).
#
#   A  = neither factor        B  = q only
#   C  = dt only               D  = dt + q
#   B1 = per-pixel q routes only (no frame-level clear-fraction embedding)
#   B2 = frame-level q route only (no per-pixel suppression, no cld in the attention gate)
Q_ROUTES = {
    "A":  dict(use_dt=False, use_q=False, q_suppress=False, q_attn_channel=False,
               q_frame=False),
    "B":  dict(use_dt=False, use_q=True,  q_suppress=True,  q_attn_channel=True,
               q_frame=True),
    "C":  dict(use_dt=True,  use_q=False, q_suppress=False, q_attn_channel=False,
               q_frame=False),
    "D":  dict(use_dt=True,  use_q=True,  q_suppress=True,  q_attn_channel=True,
               q_frame=True),
    "B1": dict(use_dt=False, use_q=True,  q_suppress=True,  q_attn_channel=True,
               q_frame=False),
    "B2": dict(use_dt=False, use_q=True,  q_suppress=False, q_attn_channel=False,
               q_frame=True),
}


def build_model_e1(cin=1, c=32, scale=3, arm="B", n_blocks=2, att_mode="pixel",
                   base_resid=False, head_init=0.0, dt_mode="gate", gamma0=0.0):
    r = Q_ROUTES[arm]
    return MISRNetE1(cin=cin, c=c, scale=scale,
                     n_blocks=n_blocks, att_mode=att_mode, base_resid=base_resid,
                     head_init=head_init, dt_mode=dt_mode, gamma0=gamma0, **r)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
