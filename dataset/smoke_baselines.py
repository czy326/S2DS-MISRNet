# -*- coding: utf-8 -*-
"""Smoke test for the E2 baselines: parameter count, step time, peak GPU memory.

The answer decides the batch size and how many seeds E2 can afford (RAMS encodes every
frame at HR resolution, which is ~16x the pixel count of the LR grid).
"""
import sys
import time
import torch

sys.path.insert(0, "/mnt/e/论文2")
from misr.baselines_s2ds import build_baseline, n_params

C = 40
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 2
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
dev = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
lr = torch.rand(BATCH, 12, 4, 48, 48, device=dev)
gt = torch.rand(BATCH, 4, 192, 192, device=dev)

for name in ["highresnet", "rams"]:
    m = build_baseline(name, cin=4, c=C, scale=4).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(N):
        out = m(lr=lr)
        loss = (out - gt).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    mem = (torch.cuda.max_memory_allocated() / 2 ** 30) if dev == "cuda" else float("nan")
    print("%-11s c=%d params=%6d  batch=%d  %.1f ms/iter  peak %.2f GiB  "
          "=> 30k iters ~%.0f min"
          % (name, C, n_params(m), BATCH, dt * 1000, mem, dt * 30000 / 60))
    del m, opt
    torch.cuda.empty_cache()
