# -*- coding: utf-8 -*-
"""Standalone bubble chart: PSNR vs #params, bubble area = MACs (S2DS test, HARD
endpoint, fixed 30k checkpoint).  NOT inserted into the manuscript -- delivered
as a standalone image at the user's request.

PSNR sources: scene_psnr_test_hard{,_fixed}.json of each run (3-seed mean;
RAMS 1 seed; zero-training baseline deterministic).
Params/MACs : dataset/model_params_flops.json (rebuilt from each run's 30k ckpt).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/mnt/e/论文2/figures")
import figstyle  # noqa: E402  (rcParams + audit_text)

# ----------------------------------------------------------------- data
pf = json.load(open("/mnt/e/论文2/dataset/model_params_flops.json", encoding="utf-8"))
MACS = {k: v["macs"] for k, v in pf.items()}
PARAMS = {k: v["params"] / 1e3 for k, v in pf.items()}

PSNR = {  # HARD endpoint, fixed-iteration 30k, mean over seeds
    "MISRNet (arm A)": (26.444 + 26.484 + 26.655) / 3,
    "MISRNet (arm B)": (26.743 + 26.864 + 27.015) / 3,
    "MISRNet (arm C)": (26.751 + 26.660 + 26.614) / 3,
    "MISRNet (arm D, ours)": (26.896 + 27.075 + 26.743) / 3,
    "HighRes-net": (25.883 + 25.765 + 26.040) / 3,
    "HighRes-net + cld": (26.029 + 25.817 + 26.085) / 3,
    "RAMS": 24.935,
    "BreizhSR": (25.392 + 25.107 + 25.601) / 3,
}
ZERO_PSNR = 26.523          # cloudaware_mean, 0 params, 0 MACs

# brighter / more saturated palette so the bubbles read clearly at print size
COLORS = {
    "MISRNet (arm A)": "#6FC3EF",
    "MISRNet (arm B)": "#1E93D2",
    "MISRNet (arm C)": "#3FBF8F",
    "MISRNet (arm D, ours)": "#E8172B",
    "HighRes-net": "#FFC971",
    "HighRes-net + cld": "#FF9A2E",
    "RAMS": "#BFA8EE",
    "BreizhSR": "#8AD97A",
}

# label placement: (dx, dy) in points, ha
LABEL = {
    "RAMS":                     (0, -18, "center"),
    "HighRes-net":              (-16, -5, "right"),
    "HighRes-net + cld":        (16, 4, "left"),
    "BreizhSR":                 (-16, -3, "right"),
    "MISRNet (arm A)":          (12, -4, "left"),
    "MISRNet (arm C)":          (12, 0, "left"),
    "MISRNet (arm B)":          (-16, 0, "right"),
    "MISRNet (arm D, ours)":    (16, 3, "left"),
}

fig, ax = plt.subplots(figsize=(7.4, 5.3))

# bubble area ~ MACs (sqrt scaling); reference bubbles 2/5/10/20 G
def area(macs):
    return 110.0 * np.sqrt(np.asarray(macs) / 2e9)

for name, y in PSNR.items():
    x = PARAMS[name]
    m = MACS[name]
    if name == "MISRNet (arm D, ours)":
        ax.scatter(x, y, s=area(m), c=COLORS[name], alpha=0.98, zorder=6,
                   edgecolors="#8A0D18", linewidths=1.8)
        ax.scatter([x], [y], marker="*", s=240, c=COLORS[name],
                   edgecolors="white", linewidths=1.0, zorder=7)
    else:
        ax.scatter(x, y, s=area(m), c=COLORS[name], alpha=0.95, zorder=5,
                   edgecolors="#3A3A3A", linewidths=1.4)

# zero-training baseline: no params, no MACs -> diamond marker on the axis
ax.scatter([0], [ZERO_PSNR], marker="D", s=110, c="#3A3A3A",
           edgecolors="white", linewidths=1.0, zorder=6)
ax.annotate("Zero-training cloud-aware\nbaseline (0 params)",
            xy=(0, ZERO_PSNR), xytext=(8, -30), textcoords="offset points",
            fontsize=9.0, ha="left", va="top", color="#1A1A1A")

# model labels -- all bold now so they survive down-scaling in print
for name, y in PSNR.items():
    dx, dy, ha = LABEL[name]
    is_ours = name.startswith("MISRNet (arm D")
    ax.annotate(name.replace("MISRNet (arm D, ours)", "MISRNet (arm D, Ours)"),
                xy=(PARAMS[name], y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center",
                fontsize=10.0 if not is_ours else 11.2,
                fontweight="bold",
                color=COLORS[name] if is_ours else "#1A1A1A", zorder=8)

# ours callout, in the spirit of the reference figure (upper-left free space)
ax.annotate("+0.38 dB vs zero-training baseline\n62% fewer MACs than HighRes-net",
            xy=(PARAMS["MISRNet (arm D, ours)"], 26.905),
            xytext=(0.015, 0.945), textcoords="axes fraction",
            fontsize=9.4, fontweight="bold", ha="left", va="top", color="#C8102E",
            arrowprops=dict(arrowstyle="-", color="#C8102E", lw=1.2, alpha=0.9,
                            connectionstyle="arc3,rad=0.0",
                            shrinkA=2, shrinkB=4))

# MACs size legend (like the reference figure)
handles = [Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(area(m)) / 1.0,
                  markerfacecolor="#9BD3F5", markeredgecolor="#3A3A3A",
                  markeredgewidth=1.2, alpha=0.95, label=t)
           for m, t in [(2e9, "2 G"), (5e9, "5 G"), (10e9, "10 G"), (20e9, "20 G")]]
leg = ax.legend(handles=handles, title="MACs", loc="center right",
                fontsize=9.0, title_fontsize=10.5, frameon=True, framealpha=0.95,
                borderpad=0.8, labelspacing=1.15, handletextpad=1.3)
leg.get_frame().set_linewidth(1.0)
leg.get_title().set_fontweight("bold")

ax.set_xlabel("Number of Parameters (K)", fontsize=12.5, fontweight="bold")
ax.set_ylabel("PSNR (dB)", fontsize=12.5, fontweight="bold")
ax.set_xlim(-20, 400)
ax.set_ylim(24.6, 27.3)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
ax.tick_params(axis="both", labelsize=11.0, width=1.1, length=4.0)
ax.grid(True, color="#BFBFBF", lw=0.8, zorder=0)
ax.set_axisbelow(True)

# full box frame like the reference figure (figstyle closes top/right by default)
for s in ax.spines.values():
    s.set_visible(True)
    s.set_linewidth(1.3)
    s.set_edgecolor("#333333")

ax.text(0.995, 0.005,
        "S2DS test, HARD endpoint, fixed-iteration 30k checkpoint; PSNR = mean over 3 seeds (RAMS: 1 seed)",
        transform=ax.transAxes, fontsize=7.6, ha="right", va="bottom", color="#555555")

fig.tight_layout()
OUT = "/mnt/e/论文2/figures/model_psnr_bubble"
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06,
            facecolor="white")
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.06, facecolor="white")
# vector output for the manuscript (text converted to paths -> renders
# identically everywhere)
fig.savefig(OUT + ".svg", bbox_inches="tight", pad_inches=0.06, facecolor="white")
figstyle.audit_text(fig, "model_psnr_bubble")
for ext in (".png", ".pdf", ".svg"):
    print("saved", OUT + ext)
