# -*- coding: utf-8 -*-
"""Smoke test for the BreizhSR reproduction: parameter count + a real forward/backward."""
import sys
sys.path.insert(0, "/mnt/e/论文2")
import torch
from misr.breizhsr_s2ds import BreizhSR

for name, pe in (("breizhsr", True), ("breizhsr_nope", False)):
    m = BreizhSR(cin=4, c=64, scale=4, position_days=pe)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    lr = torch.randn(2, 12, 4, 48, 48)
    dt = torch.randn(2, 12)
    out = m(lr, dt=dt)
    loss = out.mean()
    loss.backward()
    print("%-16s params=%d out=%s" % (name, n, tuple(out.shape)))

# the PE itself, against the formula in the paper
from misr.breizhsr_s2ds import PositionalEncoder
pe = PositionalEncoder(8, T=1000, repeat=16)
pos = torch.tensor([[0.0, 10.0, -10.0, 200.0]])
v = pe(pos)
print("pe shape", tuple(v.shape))
print("pe[0,0,:8]", [round(float(x), 5) for x in v[0, 0, :8]])
print("pe[0,1,:8]", [round(float(x), 5) for x in v[0, 1, :8]])
print("tile consistent:", bool(torch.allclose(v[0, 0, :8], v[0, 0, 8:16])))
