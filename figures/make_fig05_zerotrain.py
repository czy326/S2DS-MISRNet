# -*- coding: utf-8 -*-
"""Figure 5 -- the zero-training 2 x 2 control.

No network at all: four hand-written weighted averages of the T input frames,
    a   uniform                 (no time weight, no cloud weight)
    t   time weight only        exp(-|dt| / tau)
    c   cloud weight only       (1 - cld)^p
    ct  both
evaluated on the three pixel strata.  This is the reference against which the
learned fusion has to be judged.

Source: dataset/bench_zerotrain_2x2.json  (test split, n = 423, tau = 30 d, p = 2)

Layout notes: the two conditional panels share one figure-level legend placed
below the panels, so no legend can ever sit on top of the bars.
"""
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle as S  # noqa: E402

J = json.load(open("/mnt/e/论文2/dataset/bench_zerotrain_2x2.json",
                   encoding="utf-8"))
M, E = J["means"], J["effects"]
N = J["n"]

COMBO = [("a_uniform", "uniform", "#BDBDBD"),
         ("t_time", "time", "#7FB2D5"),
         ("c_cloud", "cloud", "#F0A868"),
         ("ct_cloud_time", "time+cloud", "#B2182B")]
STRATA = [("all", "all", "#4D4D4D"), ("hard", "HARD", S.C_HARD),
          ("easy", "EASY", S.C_ACCENT)]

fig = plt.figure(figsize=(7.16, 3.55))
gs = fig.add_gridspec(1, 3, left=0.058, right=0.986, top=0.875, bottom=0.215,
                      wspace=0.30)
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])
axc = fig.add_subplot(gs[2])

# ------------------------------------------------------------------- (a) PSNR
x = np.arange(len(COMBO))
w = 0.25
for k, (sk, sname, scol) in enumerate(STRATA):
    v = [M["%s|%s" % (ck, sk)] for ck, _, _ in COMBO]
    axa.bar(x + (k - 1) * w, v, w, color=scol, edgecolor="#333333",
            lw=0.6, label=sname, zorder=3)
axa.set_xticks(x)
axa.set_xticklabels([c[1] for c in COMBO], fontsize=6.6)
axa.set_ylabel("PSNR (dB)", fontsize=7.4)
axa.set_ylim(0, 33)
axa.tick_params(labelsize=6.6)
lg = axa.legend(fontsize=6.0, loc="upper left", framealpha=0.95, ncol=3,
                columnspacing=0.9, handlelength=1.0, borderpad=0.35)
lg.set_zorder(30)
S.style_axis(axa)
S.panel_tag(axa, "a")
S.panel_cap(axa, "Weighted-average PSNR by stratum", y=-0.10)


def cond(ax, lo, hi, labels):
    """Conditional-effect panel: two conditions x three strata."""
    xx = np.arange(len(STRATA))
    ww = 0.34
    for k, (key, col) in enumerate([(lo, "#7FB2D5"), (hi, "#B2182B")]):
        vals, errs = [], []
        for sk, _, _ in STRATA:
            e = E["%s|%s" % (key, sk)]
            vals.append(e["delta"])
            errs.append([e["delta"] - e["ci"][0], e["ci"][1] - e["delta"]])
        errs = np.array(errs).T
        b = ax.bar(xx + (k - 0.5) * ww, vals, ww, color=col,
                   edgecolor="#333333", lw=0.6, zorder=3)
        ax.errorbar(xx + (k - 0.5) * ww, vals, yerr=errs, fmt="none",
                    ecolor="#333333", elinewidth=0.8, capsize=2.2, zorder=4)
        if labels:
            ax.bar_label(b, fmt="%+.2f", fontsize=5.0, padding=2.0,
                         color="#333333")
    ax.axhline(0, color="#333333", lw=0.9, zorder=5)
    ax.set_xticks(xx)
    ax.set_xticklabels([s[1] for s in STRATA], fontsize=6.6)
    ax.set_ylabel("effect (dB)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    S.style_axis(ax)


# ------------------------------------------------- (b) time weight conditional
cond(axb, "time  | no cloud  (t - a)", "time  | cloud     (ct - c)", True)
axb.set_ylim(-0.34, 0.92)
S.panel_tag(axb, "b")
S.panel_cap(axb, "Effect of the time weight", y=-0.10)

# ------------------------------------------------ (c) cloud weight conditional
cond(axc, "cloud | no time   (c - a)", "cloud | time      (ct - t)", False)
axc.set_ylim(0, 21.5)
S.panel_tag(axc, "c")
S.panel_cap(axc, "Effect of the cloud weight", "y-scale is ~100x that of (b)",
            y=-0.10)

# shared legend for the two conditional panels, below everything
leg = fig.legend(
    handles=[Patch(facecolor="#7FB2D5", edgecolor="#333333", lw=0.6,
                   label="without the other weight"),
             Patch(facecolor="#B2182B", edgecolor="#333333", lw=0.6,
                   label="with the other weight")],
    fontsize=6.2, loc="lower center", ncol=2, framealpha=0.95,
    bbox_to_anchor=(0.5, 0.015), columnspacing=1.6, handlelength=1.2)
leg.set_zorder(60)

fig.text(0.5, 0.990,
         "Zero-training 2 x 2 control: weighted averages of the input frames "
         "(test, n = %d, tau = 30 d, p = 2)" % N,
         ha="center", va="top", fontsize=8.6, fontweight="bold")
S.save(fig, "fig05_zerotrain_2x2.png")
