# -*- coding: utf-8 -*-
"""Figure 4 -- qualitative comparison, rebuilt to journal standard.

Layout (2 x 4, equal square panels, every panel framed and lettered):
    (a) HR target   (b) bicubic nearest   (c) cloud-aware mean   (d) arm A
    (e) arm B       (f) arm C             (g) arm D              (h) arm D |error|

Each of the seven image panels carries a zoom inset of the SAME region in its
lower-right corner and the source region is boxed in white on the full frame --
the standard "zoom inset" device of Remote Sensing / ISPRS comparison figures.
The zoom window is chosen automatically: among 64 x 64 candidates it maximises
(HARD-pixel fraction) x (gradient energy), i.e. it lands on a cloudy, textured
area where the methods actually differ -- so the panel always shows something.

Titles sit ABOVE each panel in two lines (method / PSNR) and never overlap,
because the grid is built with fixed subplotspec spacing instead of tight_layout.
"""
import sys

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/mnt/e/论文2/figures")
sys.path.insert(0, "/mnt/e/论文2")
import figstyle as S                                     # noqa: E402
from misr.models import build_model                      # noqa: E402
from misr.train_s2ds import ARMS, psnr_masked            # noqa: E402

import figsample as FS                                   # noqa: E402
SHARD, IDX = FS.SHARD, FS.IDX          # never hard-code: must match figs 1, 2
SEED_RUNS = {"A": "s2dsR_armA_s2026", "B": "s2dsR_armB_s2026",
             "C": "s2dsV2_armC_s2026", "D": "s2dsV2_armD_s2026"}
DEV = "cuda"
ZW = 32                     # zoom window edge (HR grid)
# magnification of the inset relative to the full panel:
#   panel shows 192 image px across 1.00 of the axes width,
#   inset shows ZW image px across INSET_W of it.
INSET_W = 0.50
ZOOM_FRAC = INSET_W * 192.0 / ZW      # = 3.0

z = np.load(SHARD, allow_pickle=True)
i = IDX
date = str(z["date"][i])
cld_i = z["cld"][i].astype(np.float32) / 255.0
hr = z["hr"][i].astype(np.float32) / 10000.0
lr = z["lr"][i].astype(np.float32) / 10000.0
dt = z["dt"][i].astype(np.float32) / 30.0
hrc = z["hr_cloud"][i]
q = 1.0 - cld_i.reshape(12, -1).mean(1)
j0 = int(np.argmin(np.abs(dt * 30.0)))

tgt_clear = (hrc < 0.5)
c0_full = cv2.resize(cld_i[j0], (192, 192), interpolation=cv2.INTER_NEAREST) > 0.5
hard_mask = (tgt_clear & c0_full).astype(np.float32)
hm = hard_mask > 0.5

sample = dict(lr=torch.from_numpy(lr[None]), dt=torch.from_numpy(dt[None]),
              cld=torch.from_numpy(cld_i[None]),
              q=torch.from_numpy(q[None].astype(np.float32)))


def rgb(a):
    return np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32)


def norm(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def up4(x):
    return np.stack([cv2.resize(x[c], (192, 192), interpolation=cv2.INTER_CUBIC)
                     for c in range(4)])


def pick_zoom():
    """64 x 64 window with the mostHARD pixels and the most texture."""
    g = cv2.Laplacian(cv2.cvtColor((norm(rgb(hr)) * 255).astype(np.uint8),
                                   cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    e = np.abs(g)
    k = np.ones((ZW, ZW), np.float32)
    integ = lambda a: cv2.filter2D(a, -1, k / (ZW * ZW), borderType=cv2.BORDER_ISOLATED)  # noqa: E731
    hard_f = integ(hard_mask)
    ener = integ(e)
    lim = 192 - ZW
    sub_h = hard_f[ZW // 2: ZW // 2 + lim + 1, ZW // 2: ZW // 2 + lim + 1]
    sub_e = ener[ZW // 2: ZW // 2 + lim + 1, ZW // 2: ZW // 2 + lim + 1]
    score = (sub_h + 0.02) * (sub_e / (sub_e.max() + 1e-6) + 0.02)
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(c), int(r)


zx, zy = pick_zoom()

wt = np.exp(-np.abs(dt * 30.0) / 30.0).astype(np.float32)
wc = np.clip(1.0 - cld_i, 1e-3, None) ** 2
w = (wt[:, None, None, None] * wc[:, None, :, :]).astype(np.float32)
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
LBL = {"A": "Arm A   no factor", "B": "Arm B   + q",
       "C": "Arm C   + dt", "D": "Arm D   q + dt"}
for arm in "ABCD":
    panels.append((LBL[arm], preds[arm]))
arm_d = preds["D"]
panels.append(("|Arm D - HR| error", None))            # placeholder for (h)

psnr_of = {}
for name, img in panels[:-1]:
    psnr_of[name] = psnr_masked(img, hr, hard_mask) if img is not None else float("nan")

# --------------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.25, 4.28))
gs = fig.add_gridspec(2, 4, left=0.015, right=0.935, top=0.905, bottom=0.075,
                      wspace=0.055, hspace=0.50)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
LET = "abcdefgh"


def cloud_edge_overlay(vis, color=(0.99, 0.92, 0.25), alpha=0.80):
    """1-px yellow line on the nearest-frame cloud boundary."""
    m = c0_full.astype(np.uint8)
    edge = cv2.dilate(m, np.ones((5, 5), np.uint8)) - m
    v = vis.copy()
    v[edge > 0] = v[edge > 0] * (1 - alpha) + np.array(color) * alpha
    return v


err = np.abs(arm_d - hr).mean(0)
err_show = np.where(hm, err, np.nan)
vmax = float(np.nanpercentile(err, 97))

for k, ax in enumerate(axes):
    name, img = panels[k]
    if k == 7:
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("#141414")
        im = ax.imshow(err_show, cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest")
        cb = fig.colorbar(im, cax=ax.inset_axes([1.045, 0.0, 0.055, 1.0]))
        cb.ax.tick_params(labelsize=6, length=2, width=0.6)
        cb.outline.set_linewidth(0.6)
        cb.set_label("|error| (reflectance)", fontsize=6.5, labelpad=2)
    else:
        if img is None:
            vis = norm(rgb(hr)).copy()
            vis[hm] = vis[hm] * 0.70 + np.array([0.86, 0.16, 0.16]) * 0.30
        else:
            vis = norm(rgb(img))
        vis = cloud_edge_overlay(vis)
        ax.imshow(np.clip(vis, 0, 1), interpolation="nearest")
        ax.add_patch(Rectangle((zx, zy), ZW, ZW, fill=False, ec="#FFFFFF",
                               lw=1.0, zorder=20))
        axin = ax.inset_axes([1.0 - INSET_W - 0.015, 0.015, INSET_W, INSET_W])
        axin.imshow(np.clip(vis[zy:zy + ZW, zx:zx + ZW], 0, 1),
                    interpolation="nearest")
        axin.set_xticks([])
        axin.set_yticks([])
        for s in axin.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
            s.set_edgecolor("#FFFFFF")
        if k == 7:
            S.panel_cap(ax, "Arm D absolute error", "HARD pixels only")
        else:
            sub = ("PSNR(HARD) = %.2f dB" % psnr_of[name]) if img is not None \
                else "HARD pixels tinted red"
            S.panel_cap(ax, name, sub)
    S.frame_image(ax, color="#FFFFFF", lw=0.9)
    S.panel_tag(ax, LET[k])

fig.text(0.5, 0.995,
         "Qualitative comparison on a HARD-heavy test scene "
         "(AOI dg_north, target date %s, seed 2026, fixed-iteration checkpoints)"
         % date, ha="center", va="top", fontsize=8.4, fontweight="bold")
S.save(fig, "fig04_qualitative.png")

print("zoom window = (%d, %d, %d, %d)" % (zx, zy, ZW, ZW))
for nm, v in psnr_of.items():
    print("   %-22s %s" % (nm, "  n/a" if np.isnan(v) else "%.3f dB" % v))
print("best arm D - best baseline = %.3f dB"
      % (psnr_of[LBL["D"]] - max(psnr_of["Bicubic nearest"],
                                 psnr_of["Cloud-aware mean"])))
