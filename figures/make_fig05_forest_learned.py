# -*- coding: utf-8 -*-
"""Figure 5 -- forest plot of the learned 2x2 factor effects.

24 cells = 4 effects x 3 seeds x 2 endpoints.  Chinese labels (the manuscript
is Chinese); data come from working/learned2x2.json, which is produced by
working/recompute_2x2.py from the raw per-scene PSNR files and reproduces
dataset/appendix_stats.json to 1e-16.
"""
import sys, os, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402

BASE = os.environ.get("PAPER_ROOT", "/mnt/e/论文2")
CELLS = json.load(open(os.path.join(BASE, "working/learned2x2.json"), encoding="utf-8"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
})

SEEDS = (2026, 2027, 2028)
# colours back to the original palette (per user request)
C_H = S.C_HARD          # #B2182B
C_V = S.C_VALID         # #2166AC
MASKS = [("hard", "HARD (primary, n = 299)", C_H),
         ("valid", "valid (control, n = 396)", C_V)]
CLASSES = [("q_main", "q main effect"),
           ("dt_main", r"$\Delta$t main effect"),
           ("inter", r"interaction I = (D$-$C) $-$ (B$-$A)"),
           ("joint", "joint effect D $-$ A")]


def cell(fid, mask, seed):
    for c in CELLS["cells"]:
        if c["family"] == fid and c["mask"] == mask and c["seed"] == seed:
            return c
    raise KeyError((fid, mask, seed))


fig, ax = plt.subplots(figsize=(9.0, 4.70))
fig.subplots_adjust(left=0.235, right=0.975, top=0.850, bottom=0.125)

H = len(CLASSES)
diamonds = []          # (fid, mask, artist) for the ground-truth check below
for i, (fid, name) in enumerate(CLASSES):
    y0 = H - 1 - i
    if i % 2 == 0:
        ax.axhspan(y0 - 0.48, y0 + 0.48, color="#F4F4F4", zorder=0)
    for j, (mask, mname, mcol) in enumerate(MASKS):
        ys = y0 + (j - 0.5) * 0.30
        for k, s in enumerate(SEEDS):
            c = cell(fid, mask, s)
            yy = ys + (k - 1) * 0.095
            sig = c["p"] < 0.05
            ax.plot([c["ci"][0], c["ci"][1]], [yy, yy], color=mcol, lw=2.2,
                    zorder=3, solid_capstyle="butt", alpha=0.95)
            ax.plot(c["delta"], yy, "o", ms=5.8 if not sig else 6.6,
                    mec=mcol if not sig else "white",
                    mfc="white" if not sig else mcol,
                    mew=1.1, zorder=4)
        mus = np.array([cell(fid, mask, s)["delta"] for s in SEEDS])
        d_art, = ax.plot(mus.mean(), ys, "D", ms=7.4, color="#111111",
                         mec="white", mew=1.0, zorder=6)
        diamonds.append((fid, mask, d_art))

ax.axvline(0, color="#333333", lw=1.4, ls="--", zorder=2)
ax.set_yticks(np.arange(H))
# data of CLASSES[i] is drawn at y0 = H-1-i, so the label at tick k must be
# the name of CLASSES[H-1-k]  (i.e. the reversed list) to pair up correctly
ax.set_yticklabels([CLASSES[H - 1 - k][1] for k in range(H)], fontsize=12.5)
ax.set_ylim(-0.62, H - 1 + 0.62)
ax.set_xlabel("Paired PSNR effect size relative to the control (dB)",
              fontsize=14.0, labelpad=6)
ax.tick_params(labelsize=12.0, width=1.2, length=4.5)

lo = min(cell(f, m, s)["ci"][0] for f, _ in CLASSES for m, _, _ in MASKS for s in SEEDS)
hi = max(cell(f, m, s)["ci"][1] for f, _ in CLASSES for m, _, _ in MASKS for s in SEEDS)
ax.set_xlim(lo - 0.09, hi + 0.09)
S.style_axis(ax, grid=False)
ax.xaxis.grid(True, color="#CFCFCF", lw=0.9, zorder=0)
ax.set_axisbelow(True)
S.panel_tag(ax, "a", x=-0.175, y=1.045, size=14.0)

handles = [plt.Line2D([], [], marker="o", ls="none", ms=7.4, color=C_H,
                      mec="white", mew=0.9, label="HARD (primary, n = 299)"),
           plt.Line2D([], [], marker="o", ls="none", ms=7.4, color=C_V,
                      mec="white", mew=0.9, label="valid (control, n = 396)"),
           plt.Line2D([], [], marker="o", ls="none", ms=7.4, mfc="white",
                      mec="#333333", mew=1.1, color="#333333",
                      label="p $\\geq$ 0.05 (n.s.)"),
           plt.Line2D([], [], marker="D", ls="none", ms=7.8, color="#111111",
                      mec="white", mew=0.9, label="3-seed mean")]
leg = ax.legend(handles=handles, fontsize=11.4, loc="upper left", framealpha=0.96,
                edgecolor="#999999", borderpad=0.7, labelspacing=0.55,
                handletextpad=0.7)
leg.get_frame().set_linewidth(1.0)

fig.text(0.5, 0.985,
         "Learned 2$\\times$2 factor effects: forest plot "
         "(4 effects $\\times$ 3 seeds $\\times$ 2 endpoints = 24 cells)",
         ha="center", va="top", fontsize=13.0, fontweight="bold")
fig.text(0.5, 0.922,
         "Whiskers: per-sample paired 95% CI; fixed-iteration 30k protocol",
         ha="center", va="top", fontsize=11.4, color="#333333")

OUT = os.path.join(BASE, "output/a9ae06bf-ad32-4b7f-b35d-80fae7e57a6e/stage2/images",
                   "fig05_forest_learned.png")
png = OUT
pdf = OUT.replace(".png", ".pdf")
svg = OUT.replace(".png", ".svg")
fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.05, facecolor="white")
S.audit_text(fig, "fig05_forest_learned")

# ---- self-check: every group's data row and its y-tick label must sit on the
# same side of the plot (same half in display coordinates)
fig.canvas.draw()
r = fig.canvas.get_renderer()
ax_bb = ax.get_window_extent(r)
mid = 0.5 * (ax_bb.y0 + ax_bb.y1)
ok_all = True
for i, (fid, name) in enumerate(CLASSES):
    y0 = H - 1 - i
    dy = ax.transData.transform((0, y0))[1]
    # label at tick k describes the group drawn at y0 = k
    k = y0
    lab_bb = ax.get_yticklabels()[k].get_window_extent(r)
    same = (dy > mid) == ((lab_bb.y0 + lab_bb.y1) / 2.0 > mid)
    print("  group %-8s y0=%d  data_display=%.0f  label='%s' display=%.0f  %s"
          % (fid, y0, dy, ax.get_yticklabels()[k].get_text(),
             (lab_bb.y0 + lab_bb.y1) / 2.0, "OK" if same else "MISMATCH"))
    ok_all &= same
print("label/data pairing:", "ALL OK" if ok_all else "FAILED")
assert ok_all, "y-tick label / data group mismatch"

# ---- ground truth: each diamond's data coords must equal its family/mask mean,
# and CLASSES[0] (q) must display ABOVE CLASSES[-1] (joint)
for fid, mask, art in diamonds:
    yd = float(art.get_ydata()[0])
    xd = float(art.get_xdata()[0])
    want = float(np.mean([cell(fid, mask, s)["delta"] for s in SEEDS]))
    assert abs(xd - want) < 1e-9, (fid, mask, xd, want)
    print("  diamond %-8s %-6s data_y=%+.2f x=%+.3f (mean %+.3f)"
          % (fid, mask, yd, xd, want))
q_disp = ax.transData.transform((0, H - 1))[1]     # q group (y0 = H-1)
j_disp = ax.transData.transform((0, 0))[1]         # joint group (y0 = 0)
print("  q group display y=%.0f  joint group display y=%.0f  ->  %s"
      % (q_disp, j_disp, "q on TOP" if q_disp > j_disp else "q at BOTTOM"))
assert q_disp > j_disp, "q 主效应应在最上方"
print("saved", png)

print("\n== values drawn ==")
for fid, name in CLASSES:
    for mask, mname, _ in MASKS:
        mus = [cell(fid, mask, s)["delta"] for s in SEEDS]
        ps = [cell(fid, mask, s)["p"] for s in SEEDS]
        print("  %-24s %-6s %s | mean %+0.3f | p %s"
              % (name, mask, " ".join("%+0.3f" % v for v in mus), np.mean(mus),
                 " ".join("%.3g" % v for v in ps)))
