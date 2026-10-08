# -*- coding: utf-8 -*-
"""Item 7 of the major-revision checklist -- the most direct published competitor.

BreizhSR / MISR-S2 (Okabayashi, Audebert, Donike, Pelletier, "Cross-sensor
super-resolution of irregularly sampled Sentinel-2 time series", EarthVision @ CVPRW
2024) is the closest published work to this paper: it also super-resolves Sentinel-2 with
a *time series* of acquisitions, and its central architectural claim is a **time-equivariant
fusion module with a temporal positional encoding measured relative to the target date**.

This file re-implements the paper's MISR model (HighRes-net encoder + L-TAE fusion) from
the official source (github.com/aimiokab/MISR-S2, models/misr_module.py and
models/positional_encoding.py) so that it can be trained on S2DS under this paper's own
fixed-iteration protocol.

Why it matters for the paper: BreizhSR reports that the temporal positional encoding is
what makes multi-date fusion work.  This paper reports that dt is *not* established on
S2DS.  Running BreizhSR's model on S2DS separates "the architecture is wrong" from "the
clause really does not hold here": the model here gets a strictly richer time signal than
our dt factor does (it sees signed day offsets through a dedicated encoding, not a
scalar), so if dt still fails to help, the failure lives in the data, not in our model.

Faithfulness / deliberate adaptations (all departures are listed in the paper text):
  * in_channels 8 = 4 LR bands + 4 reference bands (original: 6 = RGB + RGB, because
    BreizhSR keeps only the 10 m RGB bands; S2DS carries B02/B03/B04/B08).
  * the reference frame is the per-sample median over all T frames (original: median of
    the chronologically earliest sen2_amount+1 frames; S2DS frames are stored in
    |dt|-ascending order, so "earliest" has no counterpart).
  * a parameter-free bicubic skip of the temporal mean is added, exactly as in this
    paper's other methods (HighRes-net / RAMS / every arm).  Without it none of the
    methods in this comparison reaches its reported accuracy, so leaving it out would
    handicap BreizhSR for a reason unrelated to its fusion rule.
  * the positional encoding is fed the *signed day difference* t_k - t_HR, i.e. exactly
    the quantity the original uses, recovered from S2DS's normalised dt by multiplying
    by 30, and divided by the original period tau = 1000.
  * training protocol (30k iterations, batch 8, lr 2e-4 cosine) is this paper's, not the
    original's (300k steps, batch 32, lr 7e-4 with step decay), so that the comparison is
    matched.  Note this is *not* favourable to us: the original trains 10x longer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DT_DAYS = 30.0          # s2ds_dataset.py stores dt in units of 30 days


class ResidualBlock(nn.Module):
    """conv-PReLU-conv-PReLU with an identity skip (HRNet's block)."""

    def __init__(self, c, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c, c, k, padding=p), nn.PReLU(),
            nn.Conv2d(c, c, k, padding=p), nn.PReLU())

    def forward(self, x):
        return x + self.block(x)


class HighResnetEncoder(nn.Module):
    def __init__(self, cin, c=64, num_layers=2, k=3):
        super().__init__()
        p = k // 2
        self.init_layer = nn.Sequential(nn.Conv2d(cin, c, k, padding=p), nn.PReLU())
        self.res_layers = nn.Sequential(*[ResidualBlock(c, k) for _ in range(num_layers)])
        self.final = nn.Sequential(nn.Conv2d(c, c, k, padding=p))

    def forward(self, x):
        return self.final(self.res_layers(self.init_layer(x)))


class PositionalEncoder(nn.Module):
    """Verbatim from models/positional_encoding.py of the official repo.

    denom_i = T ** (2 * (i // 2) / d);  sin on the even lanes, cos on the odd ones; the
    d-dim block is then repeated `repeat` times to fill the model width.
    """

    def __init__(self, d, T=1000, repeat=None, offset=0):
        super().__init__()
        self.d, self.T, self.repeat = d, T, repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        table = batch_positions[:, :, None] / self.denom[None, None, :]     # B x T x C
        table[:, :, 0::2] = torch.sin(table[:, :, 0::2])
        table[:, :, 1::2] = torch.cos(table[:, :, 1::2])
        if self.repeat is not None:
            table = torch.cat([table for _ in range(self.repeat)], dim=-1)
        return table


class MultiHeadAttention(nn.Module):
    """L-TAE's attention: a LEARNED master query per head, keys from the features."""

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head, self.d_k, self.d_in = n_head, d_k, d_in
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))

        # the official code calls nn.init.normal_ here, which OVERWRITES the zeros created
        # by nn.Parameter(torch.zeros(...)); reproduced as written.
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        vh = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1)

        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        attn = self.attn_dropout(self.softmax(attn))
        out = torch.matmul(attn, vh)

        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
        out = out.view(n_head, sz_b, 1, d_in // n_head).squeeze(dim=2)
        return out, attn


class LTAE2d(nn.Module):
    """Light-weight Temporal Attention Encoder, 2d variant; shared across all pixels."""

    def __init__(self, in_channels=64, n_head=16, d_k=4, mlp=(128, 64), dropout=0.2,
                 d_model=128, T=1000, positional_encoding=True):
        super().__init__()
        assert in_channels % n_head == 0, "GroupNorm(in_norm) needs n_head | in_channels"
        assert mlp[0] == d_model, "the paper's default is mlp[0] == d_model == 128"
        self.in_channels, self.n_head = in_channels, n_head
        self.d_model = d_model
        self.inconv = nn.Conv1d(in_channels, d_model, 1) if d_model is not None else None
        self.positional_encoder = (PositionalEncoder(d_model // n_head, T=T, repeat=n_head)
                                   if positional_encoding else None)
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers = []
        for i in range(len(mlp) - 1):
            layers += [nn.Linear(mlp[i], mlp[i + 1]), nn.BatchNorm1d(mlp[i + 1]),
                       nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.out_dim = mlp[-1]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, return_att=False):
        """x: B,T,d,h,w.  batch_positions: B,T (signed day offsets)."""
        B, T, d, h, w = x.shape
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(B * h * w, T, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.positional_encoder is not None:
            bp = (batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1)
                 .repeat((1, 1, 1, w)))
            bp = bp.permute(0, 2, 3, 1).contiguous().view(B * h * w, T)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out)
        out = out.permute(1, 0, 2).contiguous().view(B * h * w, -1)
        out = self.out_norm(self.dropout(self.mlp(out)))
        out = out.view(B, h, w, -1).permute(0, 3, 1, 2)

        if return_att:
            return out, attn.view(self.n_head, B, T, h, w)
        return out


class BreizhSR(nn.Module):
    """HighRes-net encoder -> L-TAE fusion with a target-relative temporal PE -> decoder."""

    def __init__(self, cin=4, c=64, scale=4, position_days=True):
        super().__init__()
        assert scale == 4, "two x2 deconv stages assume scale 4"
        self.cin, self.c, self.scale = cin, c, scale
        self.position_days = position_days
        self.encoder = HighResnetEncoder(2 * cin, c=c, num_layers=2, k=3)
        self.temporal_encoder = LTAE2d(in_channels=c, n_head=16, d_k=4, mlp=(128, 64),
                                       dropout=0.2, d_model=128, T=1000,
                                       positional_encoding=position_days)
        dmid = self.temporal_encoder.out_dim
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(dmid, dmid, 2, stride=2), nn.PReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(dmid, dmid, 2, stride=2), nn.PReLU())
        self.final = nn.Conv2d(dmid, cin, 1)

    def forward(self, lr, q=None, cld=None, dt=None):
        """lr: B,T,cin,h,w.  dt: B,T signed days / 30.  q,cld unused (time-only method,
        it is the counterpart of arm C for the cloud factor)."""
        B, T, C, h, w = lr.shape
        ref = lr.median(dim=1, keepdim=True).values.repeat(1, T, 1, 1, 1)
        stack = torch.cat([lr, ref], dim=2).reshape(B * T, 2 * C, h, w)
        f = self.encoder(stack).view(B, T, self.c, h, w)

        pos = None
        if self.position_days and dt is not None:
            pos = dt * DT_DAYS            # dataset stores dt in units of 30 days
        f = self.temporal_encoder(f, pos)

        x = self.deconv2(self.deconv1(f))
        return self.final(x) + F.interpolate(lr.mean(1), scale_factor=self.scale,
                                             mode="bicubic", align_corners=False)
