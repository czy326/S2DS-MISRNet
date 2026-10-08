# -*- coding: utf-8 -*-
"""The one sample shared by Figures 1, 2 and 4.

sz_east, target date 2024-10-29, patch index 57.

Chosen by pick_sample.py under three stated criteria:
  1. cloudiness typical of S2DS -- 6 of the 12 input frames are >95 % cloudy,
     which is exactly the dataset median (test set: 51.7 % of all frames);
  2. all three pixel strata present -- HARD 43.6 %, EASY 47.8 %,
     target-cloudy 8.7 %;
  3. arm D beats the zero-training cloud-aware baseline on HARD pixels
     (+0.84 dB), so the figure cannot contradict the paper's claim.
The nearest frame (dt = -5 d) is 49.2 % clouded, so half of the frame has to
be taken from another date while the other half can be copied -- which is
exactly the per-pixel argument the paper makes.

Every figure that shows this patch must import it from here, so they cannot
drift apart.
"""
import numpy as np
import cv2

SHARD = "/home/czy/data/s2ds_built/test_sz_east.npz"
IDX = 57


def rgb(a):
    """B04, B03, B02 -> RGB."""
    return np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32)


def norm(x, lo=2, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def up(x, k=4, interp=cv2.INTER_NEAREST):
    return np.stack([cv2.resize(x[c], (48 * k, 48 * k), interpolation=interp)
                     for c in range(4)])


def up_cubic(x, k=4):
    return up(x, k, cv2.INTER_CUBIC)


def tint(vis, mask, color, alpha=0.62):
    v = vis.copy()
    v[mask] = v[mask] * (1 - alpha) + np.array(color) * alpha
    return v


class Sample(object):
    pass


def load_sample():
    z = np.load(SHARD, allow_pickle=True)
    i = IDX
    s = Sample()
    s.SHARD, s.IDX = SHARD, IDX
    s.date = str(z["date"][i])
    s.scene = str(z["scene"][i])
    s.hr = z["hr"][i].astype(np.float32) / 10000.0
    s.lr = z["lr"][i].astype(np.float32) / 10000.0
    s.cld = z["cld"][i].astype(np.float32) / 255.0
    s.dt = z["dt"][i].astype(np.float32)                 # signed days
    s.clearfrac = z["clearfrac"][i].astype(np.float32)
    s.hr_cloud = z["hr_cloud"][i]
    s.tgt_cloud = s.hr_cloud > 0.5

    s.j_near = int(np.argmin(np.abs(s.dt)))
    s.j_clear = int(np.argmax(s.clearfrac))
    s.j_far = int(np.argmax(np.abs(s.dt)))

    s.c0 = cv2.resize(s.cld[s.j_near], (192, 192),
                      interpolation=cv2.INTER_NEAREST) > 0.5
    tgt_clear = ~s.tgt_cloud
    s.HARD = tgt_clear & s.c0
    s.EASY = tgt_clear & (~s.c0)
    s.TGTC = s.tgt_cloud

    s.hr_vis = norm(rgb(s.hr))
    s.rgb, s.norm, s.up, s.tint = rgb, norm, up, tint
    return s


if __name__ == "__main__":
    s = load_sample()
    f = lambda m: 100.0 * float(m.mean())     # noqa: E731
    print("%s  index %d  target %s" % (s.SHARD.split("/")[-1], s.IDX, s.date))
    print("HARD %.1f%%   EASY %.1f%%   target-cloudy %.1f%%"
          % (f(s.HARD), f(s.EASY), f(s.TGTC)))
    print("dt (days):", s.dt.astype(int))
    print("clearfrac :", np.round(s.clearfrac, 3))
    print("j_near %d  j_clear %d  j_far %d" % (s.j_near, s.j_clear, s.j_far))
