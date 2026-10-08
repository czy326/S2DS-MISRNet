# -*- coding: utf-8 -*-
"""Figure 2 -- how the pixel strata are defined (same sample as Figure 1).

    HR truth | nearest-frame cloud mask | the three strata | legend

Layout rules: a ONE-LINE label below each axes, and nothing else inside the
figure -- the figure title, the panel letters, the per-stratum percentages
and the 2 x 2 contingency that defines the strata all belong in the
manuscript caption, not here.

Compact geometry: the three image cells are width-matched to the square
images (no dead space left/right of each panel) and the legend column is
sized to its content.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402
from figsample import load_sample  # noqa: E402

sp = load_sample()
C_HARD, C_EASY, C_TGTC = "#B2182B", "#1B7837", "#2166AC"

# ------------------------------------------------------- geometry (inches)
S_IM, G_X, L_W = 1.12, 0.105, 1.27       # image cell, column gap, legend col
TOP_M, BOT_M = 0.03, 0.03                # margins above images / below labels
CAP = 0.055 * S_IM + 0.10                # one-line label block below an axes
W = 3 * S_IM + 3 * G_X + L_W + 0.155
H = TOP_M + S_IM + CAP + BOT_M

fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(
    1, 4, left=0.020, right=0.990,
    top=1.0 - TOP_M / H, bottom=1.0 - (TOP_M + S_IM) / H,
    width_ratios=[1.0, 1.0, 1.0, L_W / S_IM],
    wspace=G_X / ((3 * S_IM + L_W) / 4.0))
a0 = fig.add_subplot(gs[0, 0])
a1 = fig.add_subplot(gs[0, 1])
a2 = fig.add_subplot(gs[0, 2])
a3 = fig.add_subplot(gs[0, 3])

# --------------------------------------------------------------- the panels
S.show(a0, sp.hr_vis)
S.panel_cap(a0, "HR truth")

S.show(a1, np.where(sp.c0, 0.92, 0.10), cmap=plt.get_cmap("Greys"), vmin=0,
       vmax=1)
S.panel_cap(a1, "Nearest-frame cloud mask")

strata = np.zeros((192, 192, 3), np.float32)
strata[sp.EASY] = (0.106, 0.471, 0.216)
strata[sp.HARD] = (0.698, 0.094, 0.169)
strata[sp.TGTC] = (0.130, 0.400, 0.671)
S.show(a2, strata)
S.panel_cap(a2, "Three pixel strata")

# ------------------------------------------------------------------- legend
a3.axis("off")
a3.set_xlim(0, 1)
a3.set_ylim(0, 1)
for k, (nm, col) in enumerate([("HARD", C_HARD), ("EASY", C_EASY),
                               ("target cloudy", C_TGTC)]):
    y = 0.78 - k * 0.24
    a3.add_patch(Rectangle((0.04, y - 0.05), 0.11, 0.10, facecolor=col,
                           edgecolor="#333333", lw=0.7, transform=a3.transAxes))
    a3.text(0.19, y, nm, fontsize=7.0, fontweight="bold", color=col, ha="left",
            va="center")

S.save(fig, "fig02_endpoint_strata.png")
