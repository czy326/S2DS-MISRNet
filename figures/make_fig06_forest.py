# -*- coding: utf-8 -*-
"""Figure 6 -- forest plot of the four effects (3 seeds x 2 endpoints).

Source: runs_s2ds/analysis_s{seed}_test_{mask}.json  (gate injection,
fixed-iteration endpoint).  One point per (effect, seed, endpoint) cell with its
paired 95 % CI; the black diamond is the 3-seed mean of that cell.
"""
import sys
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
SEEDS = (2026, 2027, 2028)
MASKS = [("hard", "HARD (primary)", S.C_HARD),
         ("valid", "valid (control)", S.C_VALID)]
CLASSES = [("F2 main effect", "q  (cloud gate)"),
           ("F1 main effect", "dt  (time gate)"),
           ("INTERACTION (D-C)-(B-A)", "interaction"),
           ("JOINT       (D-A)", "joint  (D - A)")]

cells = {}
for s in SEEDS:
    for m, _, _ in MASKS:
        p = os.path.join(RUNS, "analysis_s%d_test_%s.json" % (s, m))
        d = json.load(open(p, encoding="utf-8"))
        for key in [c[0] for c in CLASSES]:
            v = d["sample_level"][key]
            cells[(key, s, m)] = (v["mean"], v["ci"][0], v["ci"][1], v["n"])

fig, ax = plt.subplots(figsize=(7.16, 3.60))
fig.subplots_adjust(left=0.175, right=0.975, top=0.855, bottom=0.115)

H = len(CLASSES)
for i, (key, name) in enumerate(CLASSES):
    y0 = H - 1 - i
    # alternating group shading for readability
    if i % 2 == 0:
        ax.axhspan(y0 - 0.48, y0 + 0.48, color="#F6F6F6", zorder=0)
    for j, (m, mname, mcol) in enumerate(MASKS):
        ys = y0 + (j - 0.5) * 0.30
        for k, s in enumerate(SEEDS):
            mu, lo, hi, n = cells[(key, s, m)]
            yy = ys + (k - 1) * 0.095
            ax.plot([lo, hi], [yy, yy], color=mcol, lw=1.5, zorder=3,
                    solid_capstyle="butt", alpha=0.85)
            ax.plot(mu, yy, "o", ms=4.6, color=mcol, mec="white", mew=0.6,
                    zorder=4)
        mus = np.array([cells[(key, s, m)][0] for s in SEEDS])
        ax.plot(mus.mean(), ys, "D", ms=6.0, color="#111111", zorder=6,
                mec="white", mew=0.7)

ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=2)
ax.set_yticks(np.arange(H))
# data of CLASSES[i] is drawn at y0 = H-1-i, so tick k must carry the name of
# CLASSES[H-1-k] (reversed list) -- the previous unreversed list mislabeled
# every group
ax.set_yticklabels([CLASSES[H - 1 - k][1] for k in range(H)], fontsize=7.6)
ax.set_ylim(-0.62, H - 1 + 0.62)
ax.set_xlabel("paired effect on HARD / valid PSNR (dB)", fontsize=7.6)
ax.tick_params(labelsize=7.0)
S.style_axis(ax)

lo = min(cells[k][1] for k in cells)
hi = max(cells[k][2] for k in cells)
ax.set_xlim(lo - 0.07, hi + 0.07)
ax.axhspan(-0.62, H - 1 + 0.62, color="none")

handles = [plt.Line2D([], [], marker="o", ls="none", ms=5.0, color=S.C_HARD,
                      mec="white", mew=0.5, label="HARD (primary, n = 299)"),
           plt.Line2D([], [], marker="o", ls="none", ms=5.0, color=S.C_VALID,
                      mec="white", mew=0.5, label="valid (control, n = 396)"),
           plt.Line2D([], [], marker="D", ls="none", ms=5.6, color="#111111",
                      mec="white", mew=0.6, label="3-seed mean")]
ax.legend(handles=handles, fontsize=6.6, loc="lower right", framealpha=0.95)

fig.text(0.5, 0.985,
         "Forest plot of the four effects (3 seeds x 2 endpoints, gate "
         "injection, fixed-iteration endpoint)",
         ha="center", va="top", fontsize=8.6, fontweight="bold")
fig.text(0.5, 0.935, "whiskers are paired per-sample 95 % CIs",
         ha="center", va="top", fontsize=6.8, color="#444444")
S.save(fig, "fig06_forest.png")

for key, name in CLASSES:
    for m, mname, _ in MASKS:
        mus = [cells[(key, s, m)][0] for s in SEEDS]
        print("  %-26s %-6s seeds %s  mean %+0.3f"
              % (name, m, " ".join("%+0.3f" % v for v in mus), np.mean(mus)))
