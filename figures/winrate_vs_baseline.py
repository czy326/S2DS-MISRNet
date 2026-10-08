# -*- coding: utf-8 -*-
"""Per-sample win rate of each arm against the zero-training cloud-aware mean.

The paper reports paired MEAN differences (B-A, D-A, arm vs baseline) with CIs.
A paired mean can be positive while fewer than half the samples improve, so the
win rate is a separate, non-redundant statistic: it says how often the learned
model actually beats the baseline on an individual sample, not just on average.

Baselines are recomputed here (they are closed-form, no model needed); the arm
numbers are read from the archived per-sample JSONs (last.pt / fixed iteration).
"""
import glob
import os
import sys

import numpy as np
import cv2
import json

sys.path.insert(0, "/mnt/e/论文2")
from misr.train_s2ds import psnr_masked                   # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
POS = [(r * 64, c * 64) for r in range(3) for c in range(3)]


def up4(x):
    return np.stack([cv2.resize(x[c], (192, 192), interpolation=cv2.INTER_CUBIC)
                     for c in range(4)])


def baselines():
    """key -> (cloudaware_mean, bicubic_nearest) PSNR on HARD pixels."""
    out = {}
    for f in sorted(glob.glob("/home/czy/data/s2ds_built/test_*.npz")):
        aoi = os.path.basename(f)[len("test_"):-len(".npz")]
        z = np.load(f, allow_pickle=True)
        for i in range(z["lr"].shape[0]):
            lr = z["lr"][i].astype(np.float32) / 10000.0
            hr = z["hr"][i].astype(np.float32) / 10000.0
            cld = z["cld"][i].astype(np.float32) / 255.0
            dt = z["dt"][i].astype(np.float32)
            j0 = int(np.argmin(np.abs(dt)))
            c0 = cv2.resize(cld[j0], (192, 192),
                            interpolation=cv2.INTER_NEAREST) > 0.5
            hm = ((z["hr_cloud"][i] < 0.5) & c0).astype(np.float32)
            if hm.sum() < 100:
                continue
            wt = np.exp(-np.abs(dt) / 30.0).astype(np.float32)
            wc = np.clip(1.0 - cld, 1e-3, None) ** 2
            w = (wt[:, None, None, None] * wc[:, None, :, :]).astype(np.float32)
            ca = up4((lr * w).sum(0) / w.sum(0).clip(min=1e-6))
            nb = up4(lr[j0])
            r, c = POS[i % 9]
            key = "%s_%s_%d_%d" % (aoi, z["date"][i], r, c)
            out[key] = (psnr_masked(ca, hr, hm), psnr_masked(nb, hr, hm))
    return out


def runof(seed, arm):
    pre = "s2dsR" if (seed in (2026, 2027) and arm in "AB") else "s2dsV2"
    return "%s_arm%s_s%d" % (pre, arm, seed)


def main():
    base = baselines()
    print("samples with a HARD mask: %d" % len(base))
    for seed in (2026, 2027, 2028):
        for arm in "BD":
            p = os.path.join(RUNS, runof(seed, arm), "scene_psnr_test_hard.json")
            d = json.load(open(p))
            keys = [k for k in d if k in base and d[k] == d[k]]
            a = np.array([d[k] for k in keys])
            b = np.array([base[k][0] for k in keys])
            n = a.size
            print("s%d arm %s vs cloud-aware mean: n=%d  mean delta %+.3f dB  "
                  "win %d/%d (%.0f%%)"
                  % (seed, arm, n, float((a - b).mean()),
                     int((a > b).sum()), n, 100.0 * (a > b).sum() / n))


if __name__ == "__main__":
    main()
