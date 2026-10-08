# -*- coding: utf-8 -*-
"""Figure 8 -- checkpoint-selection noise.

Left:  the monitored val PSNR of all four arms rises to a peak and then falls
       for the rest of training, so "the last checkpoint" and "the best
       checkpoint" are two different models.
Right: for every run, how much the two differ and where the peak sits.  Because
       the peak lands at a different fraction of the budget for different arms,
       comparing val-best checkpoints means comparing models that were trained
       for different amounts of time -- enough to flip an effect of the size
       this paper measures.

Source: runs_s2ds/s2dsR_arm{X}_s{seed}/log.csv  (4 arms x 2 seeds = 8 runs)
"""
import sys
import csv
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
PREFIX = "s2dsR"
SEEDS = (2026, 2027)
ARMS = [("A", "#4D4D4D"), ("B", "#F0A868"), ("C", "#7FB2D5"), ("D", "#B2182B")]


def curve(arm, seed):
    p = os.path.join(RUNS, "%s_arm%s_s%d" % (PREFIX, arm, seed), "log.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    it = np.array([int(r["iter"]) for r in rows])
    v = np.array([float(r["val_psnr"]) for r in rows])
    return it, v


fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 3.60))
fig.subplots_adjust(left=0.070, right=0.975, top=0.790, bottom=0.310,
                    wspace=0.235)

# ------------------------------------------------------------------- (a) curves
for arm, col in ARMS:
    cs = [curve(arm, s) for s in SEEDS]
    cs = [c for c in cs if c is not None]
    if not cs:
        continue
    it = cs[0][0]
    V = np.array([c[1] for c in cs])
    for k in range(V.shape[0]):
        axa.plot(it, V[k], color=col, lw=0.8, alpha=0.45, zorder=2)
    m = V.mean(0)
    axa.plot(it, m, color=col, lw=1.7, label="arm %s" % arm, zorder=3)
    b = int(np.argmax(m))
    axa.plot(it[b], m[b], "o", ms=5.5, color=col, mec="white", mew=0.7,
             zorder=5)

axa.set_xlabel("training iteration", fontsize=7.6)
axa.set_ylabel("monitored val PSNR (dB)", fontsize=7.6)
axa.set_xticks([5000, 10000, 15000, 20000, 25000, 30000])
axa.set_xticklabels(["5k", "10k", "15k", "20k", "25k", "30k"])
axa.margins(y=0.10)
axa.tick_params(labelsize=6.8, pad=2.5)
axa.legend(fontsize=6.6, loc="lower left", framealpha=0.95, ncol=2)
S.style_axis(axa)
S.panel_tag(axa, "a")
S.panel_cap(axa, "Val curves", "dot = val peak", y=-0.42)

# --------------------------------------------------------- (b) best-vs-last gap
pts = []
for arm, col in ARMS:
    for s in SEEDS:
        c = curve(arm, s)
        if c is None:
            continue
        it, v = c
        b = int(np.argmax(v))
        pts.append((100.0 * it[b] / it[-1], v[b] - v[-1], arm, s, col))

xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
axb.axvspan(min(xs), max(xs), color="#F0F0F0", zorder=0)
# marker encodes the seed, colour the arm -- no per-point text, so nothing
# can collide
for xf, gap, arm, s, col in pts:
    mk = "o" if s == SEEDS[0] else "s"
    axb.plot(xf, gap, mk, ms=6.2, color=col, mec="white", mew=0.7, zorder=4)
axb.axhline(0, color="#333333", lw=1.0, ls="--", zorder=2)
axb.set_xlabel("training budget at the val peak (%)", fontsize=7.6)
axb.set_ylabel("val-best  -  last  (dB)", fontsize=7.6)
axb.set_xlim(0, 105)
axb.set_ylim(min(ys) - 0.16, max(ys) + 0.16)
axb.tick_params(labelsize=6.8)
axb.legend(handles=[Line2D([], [], marker="o", ls="none", ms=5.0,
                           color="#555555", label="seed %d" % SEEDS[0]),
                    Line2D([], [], marker="s", ls="none", ms=5.0,
                           color="#555555", label="seed %d" % SEEDS[1])],
           fontsize=5.8, loc="lower right", framealpha=0.95, handlelength=1.1)
S.style_axis(axb)
S.panel_tag(axb, "b")
S.panel_cap(axb, "Per-run gap between the two endpoints", "circle/square = seed",
            y=-0.30)

fig.text(0.5, 0.985, "Checkpoint-selection noise: the val curve peaks early and "
                     "then decays",
         ha="center", va="top", fontsize=8.6, fontweight="bold")
S.save(fig, "fig08_ckpt_noise.png")

print("runs plotted: %d" % len(pts))
print("val-best - last  : %.3f ... %.3f dB (mean %.3f)"
      % (min(ys), max(ys), float(np.mean(ys))))
print("budget at peak   : %.0f%% ... %.0f%%"
      % (min(xs), max(xs)))
