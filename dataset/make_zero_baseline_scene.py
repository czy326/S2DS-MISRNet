"""Per-scene PSNR of the zero-training gatekeeper baseline `cloudaware_mean`.

Everything the paper reports as "相对零训练云感知基线" is a per-sample paired
contrast against this deterministic rule, so we need its per-scene values keyed
exactly like the evaluation JSONs (dg_north_20240119_0_0, ...).

Rule (identical to §4.4 and bench_zerotrain_2x2.py):
    w = exp(-|dt|/30) * (1 - cld)^2      (cld stored as 0..255 uint8)
    LR-domain per-pixel weighted mean -> bicubic x4 -> PSNR on the endpoint mask
"""
import glob
import json
import os
import numpy as np
import cv2
from scipy import stats

DST = "/home/czy/data/s2ds_built"
REFL = 10000.0
S = 4
OUT = "/mnt/e/论文2/dataset/zero_baseline_scene.json"


def up4(x):
    return np.stack([cv2.resize(x[c].astype(np.float32),
                                (x.shape[2] * S, x.shape[1] * S),
                                interpolation=cv2.INTER_CUBIC)
                     for c in range(x.shape[0])])


def psnr(pred, gt, mask, peak=1.0):
    m = mask.astype(bool)
    if int(m.sum()) < 32:
        return float("nan")
    return 10 * np.log10(peak ** 2 / (float(((pred - gt)[:, m] ** 2).mean()) + 1e-12))


out = {"hard": {}, "valid": {}}
for fp in sorted(glob.glob(os.path.join(DST, "test_*.npz"))):
    z = np.load(fp)
    hr, lr, dt, cld, hc = z["hr"], z["lr"], z["dt"], z["cld"], z["hr_cloud"]
    scenes = z["scene"]
    for i in range(hr.shape[0]):
        gt = hr[i].astype(np.float32) / REFL
        lrf = lr[i].astype(np.float32) / REFL
        valid = (hc[i] == 0)
        j0 = int(np.argmin(np.abs(dt[i])))
        c0 = cld[i][j0].astype(np.float32) / 255.0
        c0_hr = cv2.resize(c0, (gt.shape[-1], gt.shape[-2]),
                           interpolation=cv2.INTER_NEAREST)
        hard = valid & (c0_hr > 0.5)

        wt = np.exp(-np.abs(dt[i].astype(np.float32)) / 30.0)[:, None, None]
        wc = np.maximum(1.0 - cld[i].astype(np.float32) / 255.0, 1e-3) ** 2
        w = wt * wc
        syn = (lrf * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]
        pred = up4(syn)
        key = str(scenes[i])
        out["hard"][key] = psnr(pred, gt, hard)
        out["valid"][key] = psnr(pred, gt, valid)

json.dump(out, open(OUT, "w"), indent=1)
for ep in ["hard", "valid"]:
    v = np.array([x for x in out[ep].values() if np.isfinite(x)])
    print("%-6s n=%3d  mean=%.3f  (paper constant: %s)"
          % (ep, len(v), v.mean(), "26.52" if ep == "hard" else "27.046"))
print("WROTE", OUT)
