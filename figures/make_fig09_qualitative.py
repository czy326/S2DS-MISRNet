# -*- coding: utf-8 -*-
"""Figure 9 -- visual comparison on HARD scenes (3 samples x 7 methods).

Layout: 3 rows (samples, stratified by zero-training-baseline difficulty)
      x 7 columns (HR / nearest bicubic / zero-training cloud-aware /
                   HighRes-net / arm A / arm B / arm D)

Deliberately minimal: the only text inside the figure is the column header
(model name, English, title case).  Row labels were removed on request -- the
row order (AOI + date) is spelled out in the (Chinese) caption instead.  All
numeric annotations (PSNR, Delta vs baseline) live in Table 5 and Table 7, and
the meaning of the yellow contour / white zoom box is given in the caption.

Conventions kept from the previous version:
  * the SAME percentile stretch within one sample (computed on target-clear
    HR pixels) so brightness differences between methods are real;
  * the nearest-frame cloud boundary as a 1-px yellow contour;
  * a white box + zoom inset on a window shared by all seven panels of the
    sample (chosen by HARD-fraction x gradient energy -> rule-based).

Data: working/qual_fig09.npz, produced by working/dump_qual_fig09.py from the
very checkpoints behind the paper's numbers (recomputed PSNR matches the
per-scene JSONs to 0).
"""
import sys, os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S                                    # noqa: E402

BASE = os.environ.get("PAPER_ROOT", "/mnt/e/论文2")
Z = np.load(os.path.join(BASE, "working/qual_fig09.npz"), allow_pickle=True)
# the row count is whatever dump_qual_fig09.py wrote (see QS there)
NR = len([k for k in Z.files if k.startswith("s") and k[1:].isdigit()])
META = [Z["meta_s%d" % i].item() for i in range(NR)]
SAMP = [Z["s%d" % i].item() for i in range(NR)]      # dicts stored as 0-d objects

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
})

# English labels only -- the manuscript is Chinese but figure text is English,
# which is what the target journal uses inside images.
COLS = [("hr", "HR"),
        ("nearest", "Nearest\nFrame Bicubic"),
        ("cloudaware", "Cloud Aware\nZero Shot"),
        ("HighRes-net", "HighRes-Net"),
        ("arm A", "Ours A"),
        ("arm B", "Ours B (q)"),
        ("arm D", "Ours D (q + $\\Delta$t)")]
ZW = 48                      # zoom window edge on the HR grid
INSET_W = 0.52               # fraction of the panel width used by the inset

# ---------------------------------------------------------------- geometry
# The panel size is the free parameter (square panels keep the 192x192 crops
# undistorted); the canvas is derived from it, so adding rows makes the figure
# taller instead of squashing the images.  PW=0.78in with 8 rows -> ~14.6 x 18.2 cm.
NC = len(COLS)
PW = float(os.environ.get("FIG9_PANEL", 0.78))   # panel edge, inches
GUT = 0.04                   # no row labels -- panels start at the left edge
MARGIN_R = 0.10
WSPACE = 0.035               # tight column gap
HSPACE = 0.12                # tight row gap (fraction of the axis height)
HEAD = 0.20                  # room above the first row for the column headers
BOT = 0.06

WIDTH = GUT + NC * PW + (NC - 1) * WSPACE * PW + MARGIN_R
HEIGHT = HEAD + BOT + PW * (NR + (NR - 1) * HSPACE)
left = GUT / WIDTH
right = 1.0 - MARGIN_R / WIDTH
top = 1.0 - HEAD / HEIGHT
bottom = BOT / HEIGHT

fig = plt.figure(figsize=(WIDTH, HEIGHT))
gs = fig.add_gridspec(NR, NC, left=left, right=right, top=top, bottom=bottom,
                      wspace=WSPACE, hspace=HSPACE)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(NC)] for r in range(NR)])


def rgb(a):
    """B04,B03,B02 -> RGB."""
    return np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32)


def stretch(hr_rgb, valid):
    """percentile stretch computed on the TARGET-CLEAR pixels only -- clouds are
    a tiny bright minority and would otherwise squeeze the land into black."""
    px = hr_rgb[valid > 0.5] if valid is not None else hr_rgb.reshape(-1, 3)
    lo, hi = np.percentile(px, 2), np.percentile(px, 98)
    return lambda x: np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def cloud_edge(c0):
    """1-px contour on the outer side of the nearest-frame cloud mask."""
    m = c0.astype(np.uint8)
    return (cv2.dilate(m, np.ones((5, 5), np.uint8)) - m) > 0


def pick_zoom(hr, hard):
    """rule-based: 48x48 window with the most HARD pixels and the most texture."""
    g = np.abs(cv2.Laplacian(cv2.cvtColor((rgb(hr) * 255).astype(np.uint8),
                                          cv2.COLOR_RGB2GRAY), cv2.CV_32F))
    k = np.ones((ZW, ZW), np.float32) / (ZW * ZW)
    integ = lambda a: cv2.filter2D(a, -1, k, borderType=cv2.BORDER_ISOLATED)   # noqa: E731
    lim = 192 - ZW
    hf = integ(hard.astype(np.float32))[ZW // 2: ZW // 2 + lim + 1,
                                        ZW // 2: ZW // 2 + lim + 1]
    en = integ(g)[ZW // 2: ZW // 2 + lim + 1, ZW // 2: ZW // 2 + lim + 1]
    score = (hf + 0.02) * (en / (en.max() + 1e-6) + 0.02)
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(c), int(r)


def fmt_date(d):
    d = str(d)
    if len(d) == 8 and d.isdigit():
        return "%s-%s-%s" % (d[:4], d[4:6], d[6:])
    return d


# ------------------------------------------------------------------- panels
for r in range(NR):
    s, meta = SAMP[r], META[r]
    hr, hard, c0 = s["hr"], s["hard"], s["c0"].astype(bool)
    f = stretch(rgb(hr), s["valid"])
    edge = cloud_edge(c0)
    zx, zy = pick_zoom(hr, hard)

    for c, (key, name) in enumerate(COLS):
        ax = axes[r, c]
        if key == "hr":
            vis = f(rgb(hr)).copy()
            vis[hard > 0.5] = vis[hard > 0.5] * 0.70 + np.array([0.86, 0.16, 0.16]) * 0.30
        else:
            vis = f(rgb(s[key]))
        vis[edge] = vis[edge] * 0.20 + np.array([0.99, 0.92, 0.25]) * 0.80
        S.show(ax, np.clip(vis, 0, 1))

        ax.add_patch(Rectangle((zx, zy), ZW, ZW, fill=False, ec="#FFFFFF",
                               lw=0.9, zorder=20))
        axin = ax.inset_axes([1.0 - INSET_W - 0.012, 0.012, INSET_W, INSET_W])
        axin.imshow(np.clip(vis[zy:zy + ZW, zx:zx + ZW], 0, 1),
                    interpolation="nearest")
        axin.set_xticks([])
        axin.set_yticks([])
        for sp in axin.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
            sp.set_edgecolor("#FFFFFF")

        if r == 0:
            ax.set_title(name, fontsize=7.0, pad=3.0, linespacing=1.30)
        S.frame_image(ax, color="#FFFFFF", lw=0.9)
    print("  sample %d zoom window = (%d, %d, %d, %d)" % (r + 1, zx, zy, ZW, ZW))

print("  %d rows, panel %.3f in square, figure %.2f x %.2f in (%.1f x %.1f cm)"
      % (NR, PW, WIDTH, HEIGHT, WIDTH * 2.54, HEIGHT * 2.54))
S.save(fig, "fig09_qualitative_hard.png")
