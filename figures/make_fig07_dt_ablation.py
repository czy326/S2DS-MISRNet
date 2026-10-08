# -*- coding: utf-8 -*-
"""Figure 7 -- how dt is injected matters more than whether it is injected.

The dt main effect, ((C - A) + (D - B)) / 2, computed per scene and averaged,
for each (seed, endpoint) cell of the three injection designs:
    add   dt added straight into the feature map (unnormalised)
    film  dt normalised and applied as a FiLM modulation -- still in features
    gate  dt never touches the features, it only shapes the fusion weights

Source: runs_s2ds/<prefix>_arm{X}_s{seed}/scene_psnr_test_{hard,valid}.json
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
MODES = [("add", "s2dsR", (2026, 2027), "#B2182B"),
         ("film", "s2dsV1", (2026, 2027), "#E08214"),
         ("gate", "s2dsV2", (2026, 2027, 2028), "#2166AC")]
MASKS = [("hard", "HARD", S.C_HARD), ("valid", "valid", S.C_VALID)]


def cell(mode, seed, mask):
    """(mean, lo, hi, n) of the dt main effect over scenes."""
    prefix = dict((m[0], m[1]) for m in MODES)[mode]
    d = {}
    for arm in "ABCD":
        p = os.path.join(RUNS, "%s_arm%s_s%d" % (prefix, arm, seed),
                         "scene_psnr_test_%s.json" % mask)
        if not os.path.exists(p):
            return None
        d[arm] = json.load(open(p, encoding="utf-8"))
    keys = sorted(set(d["A"]) & set(d["B"]) & set(d["C"]) & set(d["D"]))
    a = np.array([d["A"][k] for k in keys], np.float64)
    b = np.array([d["B"][k] for k in keys], np.float64)
    c = np.array([d["C"][k] for k in keys], np.float64)
    e = np.array([d["D"][k] for k in keys], np.float64)
    x = ((c - a) + (e - b)) * 0.5
    x = x[np.isfinite(x)]
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - 1.96 * se), \
        float(x.mean() + 1.96 * se), len(x)


fig, ax = plt.subplots(figsize=(9.6, 4.40))
fig.subplots_adjust(left=0.085, right=0.975, top=0.855, bottom=0.135)

rng = np.random.default_rng(7)
means = {}
for i, (mode, prefix, seeds, col) in enumerate(MODES):
    y0 = len(MODES) - 1 - i
    if i % 2 == 0:
        ax.axhspan(y0 - 0.45, y0 + 0.45, color="#F4F4F4", zorder=0)
    vals = []
    for j, (mk, mkname, mkcol) in enumerate(MASKS):
        ys = y0 + (j - 0.5) * 0.30
        for k, s in enumerate(seeds):
            r = cell(mode, s, mk)
            if r is None:
                continue
            mu, lo, hi, n = r
            vals.append(mu)
            yy = ys + (k - (len(seeds) - 1) / 2.0) * 0.095 \
                + rng.uniform(-0.012, 0.012)
            ax.plot([lo, hi], [yy, yy], color=mkcol, lw=2.2, alpha=0.95,
                    zorder=3, solid_capstyle="butt")
            ax.plot(mu, yy, "o", ms=6.6, color=mkcol, mec="white", mew=1.0,
                    zorder=4)
    if vals:
        m = float(np.mean(vals))
        means[mode] = m
        ax.plot([m, m], [y0 - 0.44, y0 + 0.44], color="#111111", lw=2.6,
                zorder=6)
        ax.text(m, y0 + 0.52, "mean %+.2f" % m, fontsize=11.0, color="#111111",
                ha="center", va="bottom", fontweight="bold")

ax.axvline(0, color="#222222", lw=1.4, ls="--", zorder=2)
ax.set_yticks(np.arange(len(MODES)))
ax.set_yticklabels([m[0] for m in MODES], fontsize=13.0)
ax.set_ylim(-0.62, len(MODES) - 1 + 0.92)
allv = []
for mode, _, seeds, _ in MODES:
    for mk, _, _ in MASKS:
        for s in seeds:
            r = cell(mode, s, mk)
            if r:
                allv += [r[1], r[2]]
ax.set_xlim(min(allv) - 0.06, max(allv) + 0.06)
ax.set_xlabel("dt main effect  ((C - A) + (D - B)) / 2   (dB)", fontsize=13.5,
              labelpad=6)
ax.tick_params(labelsize=12.0, width=1.2, length=4.5)
S.style_axis(ax)
ax.xaxis.grid(True, color="#CFCFCF", lw=0.9, zorder=0)
ax.set_axisbelow(True)

handles = [plt.Line2D([], [], marker="o", ls="none", ms=7.4, color=S.C_HARD,
                      mec="white", mew=0.9, label="HARD endpoint"),
           plt.Line2D([], [], marker="o", ls="none", ms=7.4, color=S.C_VALID,
                      mec="white", mew=0.9, label="valid endpoint"),
           plt.Line2D([], [], color="#111111", lw=2.6,
                      label="mean of the cells in a row")]
leg = ax.legend(handles=handles, fontsize=11.4, loc="upper right",
                framealpha=0.96, edgecolor="#999999", borderpad=0.7,
                labelspacing=0.55, handletextpad=0.8)
leg.get_frame().set_linewidth(1.0)

fig.text(0.5, 0.985, "Ablation over the three ways of injecting dt "
                     "(fixed-iteration endpoint)",
         ha="center", va="top", fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.925, "one point per (seed, endpoint) cell; whiskers are 95 % CIs "
                     "over test scenes",
         ha="center", va="top", fontsize=11.4, color="#333333")

# PNG + PDF + SVG (vector): S.save only writes the PNG
OUT = os.path.join(S.OUT, "fig07_dt_ablation")
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
fig.savefig(OUT + ".svg", bbox_inches="tight", pad_inches=0.05, facecolor="white")
S.audit_text(fig, "fig07_dt_ablation")
for ext in (".png", ".pdf", ".svg"):
    print("saved", OUT + ext)

for mode, _, _, _ in MODES:
    print("  %-5s cells:" % mode, end=" ")
    for mk, _, _ in MASKS:
        for s in dict((m[0], m[2]) for m in MODES)[mode]:
            r = cell(mode, s, mk)
            print("%s/%d=%s" % (mk, s, "  n/a" if r is None else "%+0.3f" % r[0]),
                  end="  ")
    print("| mean %+0.3f" % means.get(mode, float("nan")))
