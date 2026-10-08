# -*- coding: utf-8 -*-
"""Figure 3 -- MISRNet architecture, FLAT 2-D layout (journal-compliant rebuild).

Origin: a hand-drawn Chinese sketch that was reviewed and judged unsuitable as
Figure 3 as-is.  This script keeps that sketch's readable left-to-right layout
but repairs what made it unusable:

  F1  the sketch drew the residual blocks as "conv -> 1x1 conv -> skip"; the
      code is a plain identity residual (two 3x3 convs, cin == cout == 32), so
      there is no 1x1 projection anywhere.
  F2  the sketch drew q and dt as two switches that multiply the attention
      logits.  The real graph has three q routes (per-pixel feature
      suppression, cld concatenated into the gate conv, and a FRAME-LEVEL
      embedding that carries about 87% of the measured q gain), and dt enters
      the same gate conv as an extra channel in gate-only mode.  Drawing them
      as one multiplicative switch would assert the per-pixel-gating claim the
      paper deliberately does not make.
  F3  the sketch's weighted-sum node had only alpha as an input; the T-frame
      features being summed were missing.  Here f_t is drawn entering the sum.

Added relative to the sketch: the f_t (T x C) feature stack, the sources of the
two conditioning inputs (SCL cloud mask / acquisition dates), pixel-shuffle x4
in the head, the 10 m ground sampling distance, a colour legend, the (a)/(b)
panel letters, and panel (b) -- the 2 x 2 factorial wiring that is the figure's
whole point.

Facts verified against misr/models_e1.py:
    stem   Conv2d(4, 32, 3x3)                     -> "3x3 stem, 4 -> 32"
    blocks ConvBlock: conv3x3 -> conv3x3, x + body -> "identity skip"
    gate   Conv2d(c + 1 (+1), c, 1) -> ReLU -> Conv2d(c, 1, 1)
    head   Conv2d(c, c*16, 3) -> PixelShuffle(4) -> Conv2d(c, 4, 3)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402

FILL = dict(input="#DCEBF7", enc="#CFE3F3", cond="#FDE8C8", att="#FAD9C1",
            head="#E6DCF2", out="#F6D5D5")
EDGE = dict(input="#5A7D96", enc="#4A7FA5", cond="#B4842A", att="#B4652A",
            head="#6A4A96", out="#A53A3A")


def card(ax, x, y, w, h, fc, ec, lw=1.1, z=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,"
                                "rounding_size=0.06", facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=z))


def txt(ax, x, y, s, size=6.4, color="#222222", weight="normal", style="italic",
        ha="center", va="center", z=30, rotation=None):
    if style == "italic":
        style = "italic"
    elif style == "bold":
        weight = "bold"
        style = "normal"
    kw = dict(fontsize=size, color=color, fontweight=weight,
              fontstyle=style, ha=ha, va=va, zorder=z, linespacing=1.35)
    if rotation is not None:
        kw["rotation"] = rotation
        kw["rotation_mode"] = "anchor"
    return ax.text(x, y, s, **kw)


def arrow(ax, p0, p1, color="#3A3A3A", lw=1.2, ls="-", rad=0.0, z=20):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 color=color, linewidth=lw, linestyle=ls,
                                 zorder=z, shrinkA=2, shrinkB=2,
                                 connectionstyle="arc3,rad=%.2f" % rad))


# ----------------------------------------------------------------- figure shell
fig = plt.figure(figsize=(7.16, 5.30))
gs = fig.add_gridspec(2, 1, left=0.012, right=0.990, top=0.945, bottom=0.030,
                      height_ratios=[3.05, 1.15], hspace=0.10)
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 7.60)
ax.set_ylim(0.42, 5.00)
ax.axis("off")

# ------------------------------------------------------------------ legend row
LEG = [("LR input stack", "input"), ("shared encoder", "enc"),
       ("conditioning", "cond"), ("per-pixel attention", "att"),
       ("reconstruction head", "head"), ("HR output", "out")]
lx = 0.02
for name, key in LEG:
    ax.add_patch(Rectangle((lx, 4.62), 0.15, 0.11, facecolor=FILL[key],
                           edgecolor=EDGE[key], lw=0.7, zorder=25))
    txt(ax, lx + 0.20, 4.675, name, size=5.6, color="#333333", style="normal",
        ha="left")
    lx += 0.20 + 0.044 * len(name) + 0.07

# ------------------------------------------------------------------ conditioning
card(ax, 2.98, 3.45, 1.20, 0.86, FILL["cond"], EDGE["cond"])
txt(ax, 3.58, 4.13, "q  cloud gate", size=6.5, color="#4A3208", style="bold")
txt(ax, 3.58, 3.91, "cld   from SCL", size=5.4, color="#6A5220")
txt(ax, 3.58, 3.71, "suppress · gate chan", size=5.0, color="#7A5A2E",
    style="italic")
txt(ax, 3.58, 3.53, "frame emb  f ← f + q_mlp(q)", size=4.8, color="#7A5A2E",
    style="italic")
card(ax, 4.30, 3.45, 1.00, 0.86, FILL["cond"], EDGE["cond"])
txt(ax, 4.80, 4.13, "dt  time gate", size=6.5, color="#4A3208", style="bold")
txt(ax, 4.80, 3.91, "Δt   from dates", size=5.4, color="#6A5220")
txt(ax, 4.80, 3.71, "gate channel only", size=5.0, color="#7A5A2E",
    style="italic")
txt(ax, 4.80, 3.53, "never in features", size=5.0, color="#7A5A2E",
    style="italic")

# ------------------------------------------------------------------ module row
card(ax, 0.04, 1.62, 0.88, 1.32, FILL["input"], EDGE["input"])
txt(ax, 0.48, 2.72, "LR input stack", size=6.6, color="#14384F", style="bold")
txt(ax, 0.48, 2.45, "T = 12 frames", size=6.0, color="#333333", style="normal")
txt(ax, 0.48, 2.22, "48 × 48", size=5.6, color="#555555")
txt(ax, 0.48, 2.00, "4 bands", size=5.6, color="#555555")
txt(ax, 0.48, 1.78, "real S2 dates", size=5.2, color="#7A7A7A")

card(ax, 1.14, 1.62, 1.30, 1.32, FILL["enc"], EDGE["enc"])
txt(ax, 1.79, 2.78, "Shared encoder", size=6.6, color="#123", style="bold")
txt(ax, 1.79, 2.52, "3×3 stem, 4 → 32 ch", size=5.5, color="#334")
txt(ax, 1.79, 2.28, "residual block 1", size=5.5, color="#334")
txt(ax, 1.79, 2.05, "residual block 2", size=5.5, color="#334")
txt(ax, 1.79, 1.80, "identity skip (x + F(x))", size=5.2, color="#666",
    style="italic")
# the brace: one encoder, all T frames
xs = [1.14, 1.14, 2.44, 2.44]
ys = [1.52, 1.46, 1.46, 1.52]
ax.plot(xs, ys, color="#4A7FA5", lw=1.0, zorder=20)
txt(ax, 1.79, 1.34, "shared weights over T", size=5.6, color="#4A7FA5",
    style="italic")

# f_t stack: three receding planes (the feature maps the attention sums)
for i in range(3):
    ox, oy = 2.60 + 0.09 * i, 1.86 + 0.10 * i
    ax.add_patch(FancyBboxPatch((ox, oy), 0.46, 0.66,
                                boxstyle="round,pad=0.008,rounding_size=0.03",
                                facecolor=FILL["enc"], edgecolor=EDGE["enc"],
                                linewidth=0.8, alpha=0.55 + 0.225 * i, zorder=6 + i))
txt(ax, 3.01, 2.42, "f_t", size=6.4, color="#123", style="bold")
txt(ax, 3.01, 2.20, "T × C", size=5.4, color="#555")

card(ax, 3.44, 1.40, 1.56, 1.76, FILL["att"], EDGE["att"])
txt(ax, 4.22, 2.94, "Per-pixel temporal attention", size=6.5, color="#3A1E06",
    style="bold")
txt(ax, 4.22, 2.68, "1×1 conv → logits", size=5.5, color="#4A2E10")
txt(ax, 4.22, 2.46, "softmax over T", size=5.5, color="#4A2E10")
txt(ax, 4.22, 2.24, "α(p, t)", size=7.0, color="#B4652A", style="bold")
txt(ax, 4.22, 2.02, "Σ_t  α(p,t) · f_t", size=6.0, color="#3A1E06")
txt(ax, 4.22, 1.76, "agg  (weighted sum)", size=5.5, color="#4A2E10",
    style="italic")

# the sum node: the head consumes fused + agg, not agg alone
ax.add_patch(Circle((5.24, 2.28), 0.065, facecolor="#FFFFFF",
                    edgecolor="#7A5A2E", linewidth=1.0, zorder=9))
txt(ax, 5.24, 2.28, "+", size=9.0, color="#7A5A2E", style="bold")

card(ax, 5.54, 1.62, 0.96, 1.32, FILL["head"], EDGE["head"])
txt(ax, 6.02, 2.72, "Reconstruction", size=6.3, color="#231239", style="bold")
txt(ax, 6.02, 2.48, "head", size=6.3, color="#231239", style="bold")
txt(ax, 6.02, 2.22, "pixel-shuffle ×4", size=5.5, color="#3A2554")
txt(ax, 6.02, 2.00, "3×3 → ↑4 → 3×3", size=5.0, color="#3A2554",
    style="italic")

card(ax, 6.66, 1.72, 0.89, 1.10, FILL["out"], EDGE["out"])
txt(ax, 7.10, 2.56, "HR", size=8.6, color="#5A1010", style="bold")
txt(ax, 7.10, 2.28, "192 × 192", size=5.8, color="#6A2020")
txt(ax, 7.10, 2.06, "10 m", size=5.6, color="#6A2020")

# ------------------------------------------------------------------ arrows
arrow(ax, (0.92, 2.28), (1.14, 2.28))
arrow(ax, (2.44, 2.28), (2.60, 2.28))
arrow(ax, (3.24, 2.28), (3.44, 2.28))
arrow(ax, (5.00, 2.28), (5.17, 2.28))
arrow(ax, (5.31, 2.28), (5.54, 2.28))
arrow(ax, (6.50, 2.28), (6.66, 2.28))
# the unweighted temporal mean runs in parallel with the attention aggregate
ax.plot([2.83, 2.83, 5.24], [1.86, 1.26, 1.26], color="#4A7FA5", lw=1.0,
        zorder=8)
arrow(ax, (5.24, 1.26), (5.24, 2.20), color="#4A7FA5", lw=1.0, rad=0.0)
txt(ax, 4.10, 1.12, "temporal mean (unweighted)", size=5.2, color="#4A7FA5",
    style="italic")
# conditioning arrows: q reaches BOTH the frame features (per-pixel suppression
# and the frame-level embedding) and the gate conv (cld channel); dt reaches the
# gate conv only.
arrow(ax, (3.35, 3.45), (3.01, 2.78), color="#B4842A", lw=1.0,
      ls=(0, (3.2, 1.8)), rad=-0.12)
arrow(ax, (3.90, 3.45), (4.05, 3.18), color="#B4842A", lw=1.0,
      ls=(0, (3.2, 1.8)), rad=-0.18)
arrow(ax, (4.80, 3.45), (4.70, 3.18), color="#B4842A", lw=1.0,
      ls=(0, (3.2, 1.8)), rad=-0.18)

# ------------------------------------------------------------------ global skip
ax.plot([0.26, 7.10], [0.92, 0.92], color="#909090", lw=1.0, ls=(0, (4, 2)),
        zorder=20)
arrow(ax, (7.10, 0.92), (7.10, 1.68), color="#909090", lw=1.0, ls=(0, (4, 2)),
      rad=-0.25)
txt(ax, 3.60, 0.62, "global skip:  bicubic( LR temporal mean )", size=5.7,
    color="#777777")

S.panel_tag(ax, "a", x=-0.008, y=1.015, size=9.0)

# ------------------------------------------------------------------ panel (b)
axb = fig.add_subplot(gs[1])
axb.axis("off")
axb.set_xlim(0, 1)
axb.set_ylim(0, 1)
txt(axb, 0.005, 0.945, "2 × 2 factorial wiring", size=7.2, color="#111111",
    style="bold", ha="left")

CELL = dict(A=("#F2F2F2", "#9A9A9A"), B=("#FDE8C8", "#B4842A"),
            C=("#DCEBF7", "#3A7CA5"), D=("#DDEEDD", "#2E7A3E"))
NAME = dict(A="control", B="q only", C="dt only", D="q + dt")
WIRE = dict(A="q off · dt off", B="q on · dt off",
            C="q off · dt on", D="q on · dt on")
gw, gh, gap, gx, gy = 0.160, 0.330, 0.060, 0.090, 0.745
for r in range(2):                              # rows: dt gate
    for c in range(2):                          # columns: q gate
        arm = "ABCD"[r * 2 + c]
        x = gx + c * (gw + 0.030)
        y = gy - r * (gh + gap)
        fc, ec = CELL[arm]
        axb.add_patch(Rectangle((x, y - gh), gw, gh, facecolor=fc,
                                edgecolor=ec, lw=1.7 if arm == "D" else 0.9,
                                zorder=2))
        txt(axb, x + gw / 2, y - 0.075, arm, size=9.5, color="#111111",
            style="bold")
        txt(axb, x + gw / 2, y - 0.200, NAME[arm], size=6.0, color="#333333",
            style="normal")
        txt(axb, x + gw / 2, y - 0.290, WIRE[arm], size=5.0, color="#666666",
            style="normal")

for c, lab in enumerate(("q gate OFF", "q gate ON")):
    txt(axb, gx + c * (gw + 0.030) + gw / 2, gy + 0.060, lab, size=6.0,
        color="#333333", style="bold")
for r, lab in enumerate(("dt gate OFF", "dt gate ON")):
    txt(axb, gx - 0.018, gy - r * (gh + gap) - gh / 2, lab, size=5.4,
        color="#555555", ha="right", style="bold")

axb.add_patch(Rectangle((0.505, 0.330), 0.480, 0.530, facecolor="#FAFAF4",
                        edgecolor="#DDDDCC", lw=0.7, zorder=0))
txt(axb, 0.525, 0.800,
    "OFF is a structural removal, not a zero weight:\n"
    "the branch is not built at all.  Neither factor can\n"
    "therefore enter as a learned zero, which is what\n"
    "keeps the attribution of each factor clean.",
    size=5.8, color="#444444", ha="left", va="top", style="normal")

S.panel_tag(axb, "b", x=-0.004, y=1.02, size=9.0)

S.save(fig, "fig03_architecture_flat.png")
