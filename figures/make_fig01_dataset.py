# -*- coding: utf-8 -*-
"""Figure 1 -- one S2DS sample: real multi-temporal inputs, SCL cloud
probability, HR truth and the three pixel strata.

Sample comes from figsample.py: sz_east, target date 2024-10-29, patch index
57.  Six of the 12 input frames are >95 % cloudy (the dataset median), all
three strata are present (HARD 43.6 %, EASY 47.8 %, target-cloudy 8.7 %), and
the nearest frame is 49.2 % clouded -- the regime the paper targets.

Layout rules (shared by every figure of the paper):
  * panel letters (a)(b)... sit OUTSIDE the axes, above-left
  * the per-panel caption sits BELOW the panel -- no collision with the letter
  * no explanatory prose inside the figure; anything the reader must be told
    lives in the figure caption
"""
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402
from figsample import load_sample  # noqa: E402

sp = load_sample()
hr, lr, cld, dt, clearfrac = sp.hr, sp.lr, sp.cld, sp.dt, sp.clearfrac
HARD, EASY, TGTC = sp.HARD, sp.EASY, sp.TGTC
frac = lambda m: 100.0 * float(m.mean())      # noqa: E731

fig = plt.figure(figsize=(7.16, 6.05))
gs = fig.add_gridspec(3, 4, left=0.014, right=0.986, top=0.898, bottom=0.170,
                      height_ratios=[1.0, 1.0, 0.50], hspace=0.62, wspace=0.075)
# NOTE: bottom is large because panel (i) carries both an x-axis label and a
# per-panel caption underneath it
axs = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

# ---------------------------------------------------------------- (a)(b)(c)
for k, (j, name) in enumerate([(sp.j_near, "Nearest"), (sp.j_clear, "Clearest"),
                               (sp.j_far, "Farthest")]):
    S.show(axs[0][k], sp.norm(sp.rgb(sp.up(lr[j]))))
    S.panel_tag(axs[0][k], "abc"[k])
    S.panel_cap(axs[0][k], "%s input frame" % name,
                "dt = %+d d,  clear %.0f%%" % (int(dt[j]), 100 * clearfrac[j]))

# ---------------------------------------------------------------------- (d)
S.show(axs[0][3], cv2.resize(cld[sp.j_near], (192, 192),
                             interpolation=cv2.INTER_NEAREST),
       cmap=plt.get_cmap("Blues"), vmin=0, vmax=1)
S.panel_tag(axs[0][3], "d")
S.panel_cap(axs[0][3], "SCL cloud probability", "nearest frame")
cb = fig.colorbar(axs[0][3].images[0],
                  cax=axs[0][3].inset_axes([1.045, 0.0, 0.055, 1.0]))
cb.ax.tick_params(labelsize=6, length=2, width=0.6)
cb.outline.set_linewidth(0.6)
cb.set_label("cloud prob.", fontsize=6.3, labelpad=2)

# ------------------------------------------------------------------- (e)(f)(g)
S.show(axs[1][0], sp.hr_vis)
S.panel_tag(axs[1][0], "e")
S.panel_cap(axs[1][0], "HR truth", "192 x 192 at 10 m")

S.show(axs[1][1], sp.tint(sp.hr_vis, HARD, (0.70, 0.09, 0.17)))
S.panel_tag(axs[1][1], "f")
S.panel_cap(axs[1][1], "HARD pixels", "%.0f%% of the frame" % frac(HARD))

S.show(axs[1][2], sp.tint(sp.hr_vis, TGTC, (0.13, 0.40, 0.67)))
S.panel_tag(axs[1][2], "g")
S.panel_cap(axs[1][2], "Target-cloudy pixels", "%.0f%%, excluded" % frac(TGTC))

# ---------------------------------------------------------------------- (h)
axh = axs[1][3]
order = [("HARD", frac(HARD), "#B2182B"), ("EASY", frac(EASY), "#1B7837"),
         ("target cloudy", frac(TGTC), "#2166AC")]
axh.barh(range(3), [v for _, v, _ in order], color=[c for _, _, c in order],
         height=0.62, edgecolor="#333333", lw=0.7)
axh.invert_yaxis()
axh.set_yticks([])
axh.set_yticklabels([])
axh.set_xlim(0, 55)
axh.set_xlabel("% of HR pixels", fontsize=7.2)
axh.tick_params(axis="x", labelsize=6.5)
# A y tick label would stick out into panel (g), so the name goes INSIDE the
# bar and the value OUTSIDE its end -- but only when the bar is long enough to
# hold the name.  For a short bar the name would spill past the bar end and
# collide with the value, so both are printed together on the outside.
for k, (nm, v, _) in enumerate(order):
    if v >= 25.0:
        axh.text(1.2, k, nm, fontsize=6.4, va="center", ha="left",
                 color="#FFFFFF", fontweight="bold")
        axh.text(v + 1.4, k, "%.0f%%" % v, fontsize=6.6, va="center",
                 ha="left", fontweight="bold", color="#222222")
    else:
        axh.text(v + 1.4, k, "%s   %.0f%%" % (nm, v), fontsize=6.4,
                 va="center", ha="left", fontweight="bold", color="#222222")
S.style_axis(axh, grid=True)
S.panel_tag(axh, "h")
S.panel_cap(axh, "Stratum composition", y=-0.30)

# ---------------------------------------------------------------------- (i)
axt = fig.add_subplot(gs[2, :])
for j in np.argsort(dt):
    cl = float(clearfrac[j])
    axt.vlines(dt[j], 0, cl, color="#3A7CA5", lw=2.4, zorder=3)
    axt.plot(dt[j], cl, "o", ms=4.2,
             color="#2E7A3E" if cl > 0.7 else ("#E08214" if cl > 0.3 else "#B2182B"),
             zorder=4)
axt.axvline(0, color="#333333", lw=1.0, ls="--", zorder=5)
axt.set_ylim(-0.06, 1.16)
axt.set_xlim(dt.min() - 6, dt.max() + 8)
axt.set_yticks([0, 0.5, 1.0])
axt.set_yticklabels(["0", "0.5", "1.0"], fontsize=6.5)
axt.set_ylabel("clear fraction", fontsize=7.0)
axt.set_xlabel("acquisition offset from the target date  dt  (days)",
               fontsize=7.2)
axt.tick_params(labelsize=6.5)
S.style_axis(axt, grid=True)
S.panel_tag(axt, "i")
S.panel_cap(axt, "Acquisition timeline of the T = 12 input frames", y=-0.62)

fig.text(0.5, 0.982, "An S2DS sample and its pixel strata",
         ha="center", va="top", fontsize=9.0, fontweight="bold")
S.save(fig, "fig01_dataset_sample.png")
