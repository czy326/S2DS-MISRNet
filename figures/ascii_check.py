# -*- coding: utf-8 -*-
"""Render a PNG as coarse ASCII so the layout can be checked without an image viewer."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

RAMP = " .:-=+*#%@"


def ascii_view(path, cols=118, rows=None, invert=True, lo=0.015, hi=0.30):
    im = mpimg.imread(path)
    if im.ndim == 3:
        im = im[:, :, :3]
    g = im.mean(axis=2)
    if invert:                       # dark ink -> high value
        g = 1.0 - g
    H, W = g.shape
    ar = H / W
    rows = rows or max(8, int(cols * ar * 0.48))
    # block-average
    ys = np.linspace(0, H, rows + 1).astype(int)
    xs = np.linspace(0, W, cols + 1).astype(int)
    out = []
    for r in range(rows):
        line = []
        for c in range(cols):
            blk = g[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            v = blk.max() if blk.size else 0.0
            v = np.clip((v - lo) / (hi - lo), 0, 1)
            line.append(RAMP[int(v * (len(RAMP) - 1))])
        out.append("".join(line))
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 118)
        print(p)
        print("=" * 118)
        print(ascii_view(p))
