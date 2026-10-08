# -*- coding: utf-8 -*-
"""Per-model parameter counts and MACs (thop) for the PSNR-vs-params bubble chart.

Each model is rebuilt from the `args` stored in its own 30k checkpoint, so the
counts match exactly what was evaluated.  MACs are measured on the standard
S2DS test input (T=12, 4-band 48x48 LR -> 192x192 HR), CPU, single sample.
"""
import os, sys, json, glob
import torch

sys.path.insert(0, "/mnt/e/论文2")
sys.stdout.reconfigure(encoding="utf-8")

from misr.build_any import build_from_spec  # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
ITEMS = [
    ("MISRNet (arm A)", "s2dsV2_armA_s2026"),
    ("MISRNet (arm B)", "s2dsV2_armB_s2026"),
    ("MISRNet (arm C)", "s2dsV2_armC_s2026"),
    ("MISRNet (arm D, ours)", "s2dsV2_armD_s2026"),
    ("HighRes-net", "s2dsE2_highresnet_s2026"),
    ("HighRes-net + cld", "s2dsE9_highresnet_cld_s2026"),
    ("RAMS", "s2dsE2_rams_s2026"),
    ("BreizhSR", "s2dsE7_breizhsr_s2026"),
]

dev = "cpu"
out = {}
CK_FALLBACK = {"s2dsV2_armA_s2026": ("s2dsV2_armC_s2026", "A"),   # ckpts pruned; structure only
               "s2dsV2_armB_s2026": ("s2dsV2_armC_s2026", "B")}
for name, run in ITEMS:
    ck_run, arm_override = CK_FALLBACK.get(run, (run, None))
    ck = torch.load(os.path.join(RUNS, ck_run, "last.pt"), map_location=dev,
                    weights_only=False)
    a = dict(ck["args"])
    if arm_override:
        a["arm"] = arm_override
    model = build_from_spec(a, cin=4, scale=4).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())

    T, H, W = 12, 48, 48
    lr = torch.randn(1, T, 4, H, W, device=dev)
    q = torch.rand(1, T, device=dev)
    cld = (torch.rand(1, T, H, W, device=dev) > 0.5).float()
    dt = torch.randn(1, T, device=dev)

    with torch.no_grad():
        y = model(lr=lr, q=q, cld=cld, dt=dt)
    macs = None
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(lr, q, cld, dt), verbose=False)
    except Exception as e:
        print("  thop failed:", repr(e))
    out[name] = dict(run=run, params=n_par, macs=(float(macs) if macs else None),
                     out_shape=list(y.shape))
    print("%-24s params=%9d  MACs=%s  out=%s"
          % (name, n_par, ("%.3g" % macs) if macs else "n/a", list(y.shape)))

json.dump(out, open("/mnt/e/论文2/dataset/model_params_flops.json", "w"), indent=1)
print("saved dataset/model_params_flops.json")
