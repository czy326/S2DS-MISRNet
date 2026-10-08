# -*- coding: utf-8 -*-
"""Figure 3 -- MISRNet architecture, drawn as a pseudo-3D (isometric) pipeline.

Panel (a) is the network: every module is an extruded box with three shaded
faces and a drop shadow, so the figure reads as a solid object the way Remote
Sensing / ISPRS architecture figures do.  The T input frames and the T feature
maps are drawn as receding slab stacks (the standard "feature-map stack" idiom).
The two conditioning branches (cloud gate q, time gate dt) enter from above with
dashed arrows, which makes the 2x2 factor wiring visually explicit.

Panel (b) is the 2x2 arm matrix: which of the two switches is on in each arm.

Every module box registers its projected bounding box in MODS; at the end the
script prints any overlap between non-adjacent modules, so a layout mistake is
caught without having to look at the render.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402

UX, UY, UZ = S.UX, S.UY, S.UZ
MODS = []          # (name, xmin, xmax, ymin, ymax)


def pt(o, w=0.0, d=0.0, h=0.0):
    return (o[0] + UX[0] * w + UY[0] * d + UZ[0] * h,
            o[1] + UX[1] * w + UY[1] * d + UZ[1] * h)


def label(ax, p, s, size=7.2, color="#1A1A1A", weight="normal", ha="center",
          va="center", style="normal"):
    return ax.text(p[0], p[1], s, fontsize=size, color=color,
                   fontweight=weight, ha=ha, va=va, fontstyle=style, zorder=60)


def register(name, o, w, d, h):
    xs = [pt(o, 0, 0, 0), pt(o, w, 0, 0), pt(o, w, d, 0), pt(o, 0, d, 0),
          pt(o, 0, 0, h), pt(o, w, 0, h), pt(o, w, d, h), pt(o, 0, d, h)]
    MODS.append((name, min(p[0] for p in xs), max(p[0] for p in xs),
                 min(p[1] for p in xs), max(p[1] for p in xs)))


def box(ax, name, o, w, d, h, fill, edge, **kw):
    S.iso_box(ax, o, w, d, h, fill, edge=edge, **kw)
    register(name, o, w, d, h)
    return o


# ----------------------------------------------------------------- figure shell
fig = plt.figure(figsize=(7.16, 4.78))
gs = fig.add_gridspec(2, 1, left=0.010, right=0.990, top=0.880, bottom=0.030,
                      height_ratios=[2.62, 1.25], hspace=0.30)
ax = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])

ax.set_xlim(-0.90, 15.60)
ax.set_ylim(-0.78, 5.36)
ax.set_aspect("equal")
ax.axis("off")

F_IN, F_ENC, F_COND, F_ATT, F_FUSE, F_HEAD, F_OUT = (
    S.FILL_INPUT, S.FILL_ENC, S.FILL_COND, S.FILL_ATT,
    S.FILL_FUSE, S.FILL_HEAD, S.FILL_OUT)

# ------------------------------------------------------------------ (1) input
N = 5                      # visible slabs; the "..." says the stack continues
STEP = 0.42                # d + gap inside iso_plane_stack
BACK = (N - 1) * STEP      # 1.68
o_in = (0.00, 0.85)
S.iso_plane_stack(ax, o_in, 1.45, 0.36, 0.60, N, F_IN, gap=0.06,
                  edge="#5A7D96", shade=(1.0, 0.90, 0.74))
register("input", o_in, 1.45, BACK + 0.36 * 0.55, 0.60)
label(ax, pt(o_in, 0.72, 0.10, 0.32), "LR", size=6.6, weight="bold",
      color="#14384F")
label(ax, pt(o_in, 0.72, -0.06, -0.30), "T = 12 frames", size=6.6,
      weight="bold", color="#333")
label(ax, pt(o_in, 0.72, -0.06, -0.66), "48 x 48, 4 bands", size=5.7,
      color="#666")
label(ax, pt(o_in, 0.72, -0.06, -0.96), "real acquisition dates", size=5.6,
      color="#7A7A7A", style="italic")
label(ax, pt(o_in, 0.72, BACK + 0.30, 0.34), "...", size=8.5, weight="bold",
      color="#5A7D96")

# ------------------------------------------------------------------ (2) encoder
o_en = (2.50, 0.55)
box(ax, "encoder", o_en, 1.42, 1.02, 1.70, F_ENC, "#4A7FA5")
label(ax, pt(o_en, 0.71, 0.51, 1.24), "Shared", size=7.2, weight="bold",
      color="#123")
label(ax, pt(o_en, 0.71, 0.51, 0.90), "encoder", size=7.2, weight="bold",
      color="#123")
label(ax, pt(o_en, 0.71, 0.51, 0.54), "3x3 stem", size=5.6, color="#334")
label(ax, pt(o_en, 0.71, 0.51, 0.22), "+ 2 resblk", size=5.6, color="#334")
label(ax, pt(o_en, 0.71, -0.02, -0.28), "shared over T", size=5.8, color="#666",
      style="italic")

# ------------------------------------------------------------------ (3) features
o_fe = (4.55, 1.17)
S.iso_plane_stack(ax, o_fe, 1.38, 0.36, 0.58, N, F_ENC, gap=0.06,
                  edge="#4A7FA5", shade=(1.0, 0.90, 0.74))
register("features", o_fe, 1.38, BACK + 0.36 * 0.55, 0.58)
label(ax, pt(o_fe, 0.69, 0.10, 0.30), "f_t", size=7.2, weight="bold",
      color="#123")
label(ax, pt(o_fe, 0.69, -0.06, -0.30), "T x C", size=6.4, color="#555")

# ------------------------------------------------------------------ (4) attention
o_at = (6.90, 0.40)
box(ax, "attention", o_at, 1.95, 1.25, 2.28, F_ATT, "#B4652A",
    shade=(1.0, 0.84, 0.66))
label(ax, pt(o_at, 0.98, 0.62, 1.80), "Per-pixel temporal", size=6.9,
      weight="bold", color="#3A1E06")
label(ax, pt(o_at, 0.98, 0.62, 1.53), "attention", size=6.9, weight="bold",
      color="#3A1E06")
label(ax, pt(o_at, 0.98, 0.62, 1.06), "1x1 conv -> logits", size=5.6,
      color="#4A2E10")
label(ax, pt(o_at, 0.98, 0.62, 0.82), "softmax over T", size=5.6,
      color="#4A2E10")
label(ax, pt(o_at, 0.98, 0.62, 0.58), "last conv zero-init", size=5.4,
      color="#7A5A2E", style="italic")
label(ax, pt(o_at, 0.98, -0.02, -0.26), "alpha (p, t)", size=6.6,
      weight="bold", color="#B4652A")
# conditioning boxes pulled apart so their two labels cannot touch
O_Q, O_DT = 6.15, 8.15

# ------------------------------------------------------- (5) conditioning blocks
for ox, t1, t2 in ((6.15, "q  cloud gate", "from SCL"),
                   (8.15, "dt  time gate", "from dates")):
    o_c = (ox, 3.62)
    box(ax, "cond-%.2f" % ox, o_c, 1.15, 0.62, 0.76, F_COND, "#B4842A",
        shade=(1.0, 0.86, 0.68))
    label(ax, pt(o_c, 0.575, 0.31, 0.46), t1, size=6.5, weight="bold",
          color="#4A3208")
    label(ax, pt(o_c, 0.575, 0.31, 0.18), t2, size=5.4, color="#6A5220")
    S.arrow(ax, pt(o_c, 0.575, 0.31, -0.02), pt(o_at, 0.98, 0.55, 2.42),
            color="#B4842A", lw=1.1, ls=(0, (3.4, 1.8)), style="-|>", rad=-0.10)

# ------------------------------------------------------------------ (6) fusion
o_fu = (9.55, 0.85)
box(ax, "fusion", o_fu, 1.34, 0.92, 1.42, F_FUSE, "#3E7A45")
label(ax, pt(o_fu, 0.67, 0.46, 0.92), "Fused", size=6.9, weight="bold",
      color="#123")
label(ax, pt(o_fu, 0.67, 0.46, 0.64), "feature", size=6.9, weight="bold",
      color="#123")
label(ax, pt(o_fu, 0.67, 0.46, 0.30), "sum_t  alpha * f_t", size=5.6,
      color="#334")

# ------------------------------------------------------------------ (7) head
o_he = (11.45, 0.55)
box(ax, "head", o_he, 1.38, 1.00, 1.66, F_HEAD, "#6A4A96")
label(ax, pt(o_he, 0.69, 0.50, 1.12), "Reconstruction", size=6.8,
      weight="bold", color="#231239")
label(ax, pt(o_he, 0.69, 0.50, 0.84), "head", size=6.8, weight="bold",
      color="#231239")
label(ax, pt(o_he, 0.69, 0.50, 0.44), "pixel-shuffle x4", size=5.6,
      color="#3A2554")

# ------------------------------------------------------------------ (8) output
o_ou = (13.40, 0.75)
box(ax, "output", o_ou, 1.42, 0.62, 1.52, F_OUT, "#A53A3A")
label(ax, pt(o_ou, 0.71, 0.31, 0.98), "HR", size=9.2, weight="bold",
      color="#5A1010")
label(ax, pt(o_ou, 0.71, 0.31, 0.56), "192 x 192", size=6.0, color="#6A2020")
label(ax, pt(o_ou, 0.71, 0.31, 0.24), "10 m", size=5.8, color="#6A2020")

# ------------------------------------------------------------------ flow arrows
chain = [(o_in, 1.45 + BACK * UX[0] + 0.36 * 0.55 * UX[0], 0.60, o_en, 1.70),
         (o_en, 1.42, 1.70, o_fe, 0.58),
         (o_fe, 1.38 + BACK * UX[0] + 0.36 * 0.55 * UX[0], 0.58, o_at, 2.28),
         (o_at, 1.95, 2.28, o_fu, 1.42),
         (o_fu, 1.34, 1.42, o_he, 1.66),
         (o_he, 1.38, 1.66, o_ou, 1.52)]
for a, aw, ah, b, bh in chain:
    S.arrow(ax, pt(a, aw + 0.08, 0.0, ah * 0.5), pt(b, -0.08, 0.0, bh * 0.5),
            color="#3A3A3A", lw=1.35, style="-|>", rad=0.0)

# ------------------------------------------------------------------ global skip
y_skip = -0.05
x_end = pt(o_ou, 0.71, 0.0, 0.0)[0]
ax.plot([0.30, x_end], [y_skip, y_skip], color="#909090", lw=1.0,
        ls=(0, (4, 2)), zorder=25)
S.arrow(ax, (x_end, y_skip), pt(o_ou, 0.55, 0.20, 0.06), color="#909090",
        lw=1.0, ls=(0, (4, 2)), style="-|>", rad=-0.22)
# label sits well below the line and well below alpha (p, t)
label(ax, (7.30, y_skip - 0.32), "global skip:  bicubic( LR temporal mean )",
      size=5.7, color="#777", style="italic")

# ------------------------------------------------------------------ legend
lg = [(F_IN, "#5A7D96", "LR input stack"),
      (F_ENC, "#4A7FA5", "shared encoder"),
      (F_COND, "#B4842A", "conditioning"),
      (F_ATT, "#B4652A", "per-pixel attention"),
      (F_FUSE, "#3E7A45", "aggregation"),
      (F_HEAD, "#6A4A96", "reconstruction head"),
      (F_OUT, "#A53A3A", "HR output")]
xx, yy = 0.02, 5.02
for i, (fc, ec, t) in enumerate(lg):
    cx = xx + i * 2.20
    ax.add_patch(Rectangle((cx, yy), 0.30, 0.24, facecolor=fc, edgecolor=ec,
                           lw=0.7, zorder=70))
    ax.text(cx + 0.40, yy + 0.12, t, fontsize=5.7, color="#333", ha="left",
            va="center", zorder=70)

S.panel_label(ax, "(a)", x=-0.058, y=1.010, size=9.5)

# ------------------------------------------------------------------ panel (b)
axb.axis("off")
axb.set_xlim(0, 1)
axb.set_ylim(0, 1)
axb.text(0.0, 0.90, "2 x 2 factorial wiring", fontsize=7.6, fontweight="bold",
         color="#111", ha="left", va="center")

CELL = dict(A=("#F2F2F2", "#9A9A9A"), B=("#FDE8C8", "#B4842A"),
            C=("#DCEBF7", "#3A7CA5"), D=("#DDEEDD", "#2E7A3E"))
TXT = dict(A="control", B="q only", C="dt only", D="q + dt")
gw, gh, gx, gy = 0.215, 0.34, 0.130, 0.34
for r, dt_on in enumerate((False, True)):        # rows: dt
    for c, q_on in enumerate((False, True)):     # cols: q
        arm = "ABCD"[r * 2 + c]
        x = gx + c * (gw + 0.050)
        y = gy - r * (gh + 0.060)
        fc, ec = CELL[arm]
        axb.add_patch(Rectangle((x, y - gh), gw, gh, facecolor=fc,
                                edgecolor=ec, lw=1.6 if arm == "D" else 0.9,
                                zorder=2))
        # the ON/OFF state is already given by the row and column headers,
        # so the cell only carries the arm letter and its short name
        axb.text(x + gw / 2, y - 0.075, arm, fontsize=9.0, fontweight="bold",
                 color="#111", ha="center", va="center", zorder=3)
        axb.text(x + gw / 2, y - 0.180, TXT[arm], fontsize=6.2,
                 color="#333", ha="center", va="center", zorder=3)

axb.text(gx + gw + 0.0225, gy + 0.055, "q gate  OFF            ON",
         fontsize=6.4, fontweight="bold", color="#333", ha="center",
         va="center")
axb.text(gx - 0.045, gy - gh / 2, "dt gate", fontsize=6.4, fontweight="bold",
         color="#333", ha="center", va="center", rotation=90)
axb.text(gx - 0.075, gy - 0.02, "ON", fontsize=6.0, color="#555", ha="center",
         va="center", rotation=90)
axb.text(gx - 0.075, gy - gh - 0.02, "OFF", fontsize=6.0, color="#555",
         ha="center", va="center", rotation=90)

axb.text(0.60, 0.62,
         "A switch that is OFF is\nstructurally unreachable:\nthe branch is not built at\nall -- not merely zero-\nweighted.  This is what\nkeeps each factor's\nattribution clean.",
         fontsize=6.0, color="#444", ha="left", va="center", linespacing=1.45)
axb.add_patch(Rectangle((0.575, 0.20), 0.42, 0.62, facecolor="#FAFAF4",
                        edgecolor="#DDDDCC", lw=0.7, zorder=0))

S.panel_label(axb, "(b)", x=-0.018, y=0.985, size=9.5)

# ------------------------------------------------------------------ geometry check
print("--- module bounding boxes (data units) ---")
for n, x0, x1, y0, y1 in MODS:
    print("  %-10s x [%6.2f, %6.2f]  y [%6.2f, %6.2f]" % (n, x0, x1, y0, y1))
bad = []
for i in range(len(MODS)):
    for j in range(i + 1, len(MODS)):
        a, b = MODS[i], MODS[j]
        ox = min(a[2], b[2]) - max(a[1], b[1])
        oy = min(a[4], b[4]) - max(a[3], b[3])
        if ox > 0.02 and oy > 0.02:
            sa = (a[2] - a[1]) * (a[4] - a[3])
            sb = (b[2] - b[1]) * (b[4] - b[3])
            bad.append("%s / %s overlap %.3f (%.0f%% of smaller)" %
                       (a[0], b[0], ox * oy, 100 * ox * oy / min(sa, sb)))
print("--- overlaps ---")
print("\n".join("  ! " + s for s in bad) if bad else "  none")

fig.text(0.5, 0.985, "MISRNet architecture and the 2 x 2 factor wiring",
         ha="center", va="top", fontsize=9.0, fontweight="bold")
fig.text(0.5, 0.948,
         "a shared encoder maps each of the T input frames to features; a per-pixel "
         "softmax attention aggregates them; q and dt condition the attention only",
         ha="center", va="top", fontsize=6.9, color="#444444")

S.save(fig, "fig03_architecture.png")
