# -*- coding: utf-8 -*-
"""GPU latency benchmark for the S2DS models (table B1 / 附录 A 的"推理耗时"列).

Protocol identical to _sh/bench_efficiency.py:
  * batch size 1, input = T=12 frames of 4-band 48x48 LR -> 192x192 HR
  * latency = pure model forward (tensors already resident on the device)
  * warmup 50, timed repetitions 200, report the MEDIAN, CUDA events
Run with an idle GPU.  Output: dataset/model_latency_s2ds.json
"""
import os
import sys
import json
import statistics

sys.path.insert(0, "/mnt/e/论文2")
sys.stdout.reconfigure(encoding="utf-8")

import torch                                                    # noqa: E402
from misr.build_any import build_from_spec                      # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/model_latency_s2ds.json"
WARMUP, REPS = 50, 200
CKPT_PREF = ["ckpt_030000.pt", "last.pt", "best.pt"]

ITEMS = [
    ("MISRNet (arm A)", "s2dsV2_armA_s2026"),
    ("MISRNet (arm D, ours)", "s2dsV2_armD_s2026"),
    ("HighRes-net", "s2dsE2_highresnet_s2026"),
    ("HighRes-net + cld", "s2dsE9_highresnet_cld_s2026"),
    ("RAMS", "s2dsE2_rams_s2026"),
    ("BreizhSR", "s2dsE7_breizhsr_s2026"),
]
# arm A / B checkpoints were pruned; reuse arm C's checkpoint for the structure only
CK_FALLBACK = {"s2dsV2_armA_s2026": ("s2dsV2_armC_s2026", "A"),
               "s2dsV2_armB_s2026": ("s2dsV2_armC_s2026", "B")}

dev = "cuda"
T, H, W = 12, 48, 48
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))

out = {}
for name, run in ITEMS:
    ck_run, arm_override = CK_FALLBACK.get(run, (run, None))
    ck_path = None
    for c in CKPT_PREF:
        p = os.path.join(RUNS, ck_run, c)
        if os.path.exists(p):
            ck_path = p
            break
    if ck_path is None:
        print("  !! no checkpoint for", run)
        continue
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    spec = dict(ck["args"])
    if arm_override:
        spec["arm"] = arm_override
    model = build_from_spec(spec, cin=4, scale=4).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())

    lr = torch.randn(1, T, 4, H, W, device=dev)
    q = torch.rand(1, T, device=dev)
    cld = (torch.rand(1, T, H, W, device=dev) > 0.5).float()
    dt = torch.randn(1, T, device=dev)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(lr=lr, q=q, cld=cld, dt=dt)
        torch.cuda.synchronize()
        times = []
        for _ in range(REPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(lr=lr, q=q, cld=cld, dt=dt)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * len(times)) - 1]
    out[name] = dict(run=run, ckpt=os.path.basename(ck_path), params=n_par,
                     latency_ms_median=round(med, 3), latency_ms_p90=round(p90, 3),
                     n_reps=REPS, device=torch.cuda.get_device_name(0))
    print("%-24s params=%8d  median=%.3f ms  p90=%.3f ms" % (name, n_par, med, p90))
    del model
    torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("saved", OUT)
