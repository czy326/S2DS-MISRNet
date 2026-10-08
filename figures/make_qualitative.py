# -*- coding: utf-8 -*-
"""Figure: qualitative comparison, rebuilt to journal standard.

Layout (2 x 4, equal square panels, every panel framed and lettered):
    (a) HR target            (b) bicubic nearest   (c) cloudaware mean   (d) arm A
    (e) arm B                (f) arm C             (g) arm D             (h) arm D |error|
Each of the seven image panels carries a x3 zoom inset of the SAME region in its
lower-right corner, and the source region is boxed in white on the full frame -- this is
the standard "zoom inset" device of Remote Sensing / ISPRS comparison figures.
Titles sit ABOVE each panel in two lines (method / PSNR) and never overlap, because the
grid is built with fixed subplotspec spacing instead of tight_layout.
"""
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(0, "/mnt/e/论文2/figures")
sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
import figstyle as S                                    # noqa: E402
from misr.models import build_model                     # noqa: E402
from misr.train_s2ds import ARMS, psnr_masked           # noqa: E402

SHARD = "/home/czy/data/s2ds_built/test_dg_north.npz"
IDX = 84
SEED_RUNS = {"A": "s2dsR_armA_s2026", "B": "s2dsR_armB_s2026",
             "C": "s2dsV2_armC_s2026", "D": "s2dsV2_armD_s2026"}
DEV = "cuda"
ZOOM = (96, 20, 60, 60)          # x0, y0, w, h  (HR grid, the region every inset shows)
ZOOM_FRAC = 3.2                  # inset magnification

z = np.load(SHARD, allow_pickle=True)
i = IDX
cld_i = z["cld"][i].astype(np.float32) / 255.0
hr = z["hr"][i].astype(np.float32) / 10000.0
lr = z["lr"][i].astype(np.float32) / 10000.0
dt = z["dt"][i].astype(np.float32) / 30.0
hrc = z["hr_cloud"][i]
q = 1.0 - cld_i.reshape(12, -1).mean(1)
j0 = int(np.argmin(np.abs(dt * 30.0)))

sample = dict(lr=torch.from_numpy(lr[None]), dt=torch.from_numpy(dt[None]),
              cld=torch.from_numpy(cld_i[None]),
              q=torch.from_numpy(q[None].astype(np.float32)))

valid_full = (hrc < 0.5).astype(np.float32)
c0_full = cv2.resize(cld_i[j0], (192, 192), interpolation=cv2.INTER_NEAREST)
hard_mask = ((valid_full > 0.5) & (c0_full > 0.5)).astype(np.float32)


def rgb(arr):
    return np.stack([arr[2], arr[1], arr[0]], axis=-1).astype(np.float32)


def norm(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def up4(x):
    return np.stack([cv2.resize(x[c], (192, 192), interpolation=cv2.INTER_CUBIC)
                     for c in range(4)])


def cloud_edge_overlay(vis, color=(0.99, 0.92, 0.25), alpha=0.75, thick=True):
    """1-px yellow line on the nearest-frame cloud boundary."""
    m = (c0_full > 0.5).astype(np.uint8)
    k = np.ones((5, 5), np.uint8) if thick else np.ones((3, 3), np.uint8)
    edge = cv2.dilate(m, k) - m
    vis = vis.copy()
    vis[edge > 0] = vis[edge > 0] * (1 - alpha) + np.array(color) * alpha
    return vis


wt_t = np.exp(-np.abs(dt * 30.0) / 30.0).astype(np.float32)
wc = np.clip(1.0 - cld_i, 1e-3, None) ** 2
w = (wt_t[:, None, None, None] * wc[:, None, :, :]).astype(np.float32)
cloudaware = up4((lr * w).sum(0) / w.sum(0).clip(min=1e-6))
nearest = up4(lr[j0])

panels = [("HR target", None),
          ("Bicubic nearest", nearest),
          ("Cloud-aware mean", cloudaware)]
with torch.no_grad():
    preds = {}
    for arm in "ABCD":
        run = SEED_RUNS[arm]
        ck = torch.load("/mnt/e/论文2/runs_s2ds/%s/last.pt" % run,
                        map_location="cpu", weights_only=False)
        a = ck["args"]
        use_dt, use_q = ARMS[a["arm"]]
        model = build_model(cin=4, c=a["c"], scale=4, use_dt=use_dt, use_q=use_q,
                            att_mode=a.get("att_mode", "frame"),
                            base_resid=a.get("base_resid", False),
                            head_init=a.get("head_init", 0.0),
                            dt_mode=a.get("dt_mode", "add")).to(DEV)
        model.load_state_dict(ck["model"])
        model.eval()
        preds[arm] = model(lr=sample["lr"].to(DEV), q=sample["q"].to(DEV),
                           cld=sample["cld"].to(DEV),
                           dt=sample["dt"].to(DEV))[0].cpu().numpy()
LBL = {"A": "Arm A (no factors)", "B": "Arm B (+ cloud gate q)",
       "C": "Arm C (+ time gate \u0394t)", "D": "Arm D (\u0394t + q)"}
for arm in "ABCD":
    panels.append((LBL[arm], preds[arm]))
arm_d = preds["D"]
panels.append(("|Arm D \u2212 HR| error", None))     # placeholder for panel (h)

psnr_of = {}
for name, img in panels[:-1]:
    psnr_of[name] = psnr_masked(img, hr, hard_mask) if img is not None else float("nan")

# --------------------------------------------------------------------- figure grid
fig = plt.figure(figsize=(7.25, 4.15))
gs = fig.add_gridspec(2, 4, left=0.015, right=0.935, top=0.845, bottom=0.015,
                      wspace=0.055, hspace=0.34)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
LET = "abcdefgh"

zx0, zy0, zw, zh = ZOOM
hm = hard_mask > 0.5
err = np.abs(arm_d - hr).mean(0)
err_show = np.where(hm, err, np.nan)
vmax = float(np.nanpercentile(err, 97))

for k, ax in enumerate(axes):
    name, img = panels[k]
    if k == 7:                                     # error map
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("#141414")
        im = ax.imshow(err_show, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        # a cax placed OUTSIDE the axes keeps panel (h) exactly as wide as (a)-(g)
        cb = fig.colorbar(im, cax=ax.inset_axes([1.045, 0.0, 0.055, 1.0]))
        cb.ax.tick_params(labelsize=6, length=2, width=0.6)
        cb.outline.set_linewidth(0.6)
        cb.set_label("|error| (reflectance)", fontsize=6.5, labelpad=2)
        ax.set_title("|Arm D \u2212 HR|\nHARD pixels only", fontsize=7.8, pad=3,
                     linespacing=1.25)
    else:
        if img is None:
            vis = norm(rgb(hr)).copy()
            vis[hm] = vis[hm] * 0.70 + np.array([0.86, 0.16, 0.16]) * 0.30
        else:
            vis = norm(rgb(img))
        vis = cloud_edge_overlay(vis)
        ax.imshow(np.clip(vis, 0, 1), interpolation="bilinear")
        # white box marking the zoom source region
        ax.add_patch(Rectangle((zx0, zy0), zw, zh, fill=False, ec="#FFFFFF",
                               lw=0.9, zorder=20))
        axin = ax.inset_axes([0.60, 0.02, 0.385, 0.385])
        axin.imshow(np.clip(vis[zy0:zy0 + zh, zx0:zx0 + zw], 0, 1),
                    interpolation="nearest")
        axin.set_xticks([])
        axin.set_yticks([])
        for s in axin.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.9)
            s.set_edgecolor("#FFFFFF")
        title = name
        sub = "PSNR(HARD) = %.2f dB" % psnr_of[name] if img is not None \
            else "HARD pixels tinted red"
        ax.set_title(title + "\n" + sub, fontsize=7.8, pad=3, linespacing=1.25)
    S.frame_image(ax, color="#FFFFFF", lw=0.9)
    # panel letter INSIDE the image (top-left badge) so it can never collide with the
    # centred two-line title above the panel
    ax.text(0.018, 0.985, "(%s)" % LET[k], transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", color="#FFFFFF", ha="left", va="top", zorder=40,
            bbox=dict(boxstyle="square,pad=0.22", facecolor="#000000", alpha=0.55,
                      edgecolor="none"))

fig.text(0.5, 0.995,
         "Qualitative comparison on a HARD-heavy test scene "
         "(dg_north, target 2024-12-29, seed 2026, fixed-iteration checkpoints)",
         ha="center", va="top", fontsize=8.6, fontweight="bold")
fig.text(0.5, 0.955,
         "yellow outline = nearest-frame cloud boundary;  white box = region shown at "
         "\u00d73.2 zoom in the lower-right corner of every panel",
         ha="center", va="top", fontsize=7.0, color="#444444")

S.save(fig, "fig_qualitative.png")
