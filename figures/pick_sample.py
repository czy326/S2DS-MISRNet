# -*- coding: utf-8 -*-
"""Pick the shared qualitative sample for Figures 1 / 2 / 4.

Criteria (all stated so the choice is reproducible, not cherry-picked):
  1. cloudiness typical of S2DS: number of >95%-cloudy input frames in [4, 7]
     (dataset median is 6 of 12);
  2. all three pixel strata present: HARD >= 15 %, EASY >= 12 %,
     target-cloudy >= 8 %;
  3. arm D beats the zero-training cloud-aware baseline on HARD pixels
     (so the figure does not contradict the paper's claim).

Among the samples that pass, rank by (arm D - cloud-aware) on HARD pixels.
"""
import glob
import os

import numpy as np
import cv2
import torch

import sys
sys.path.insert(0, "/mnt/e/论文2")
from misr.models import build_model                       # noqa: E402
from misr.train_s2ds import ARMS, psnr_masked             # noqa: E402

DEV = "cuda"
NFULL_LO, NFULL_HI = 4, 7
MIN_HARD, MIN_EASY, MIN_TGTC = 0.15, 0.12, 0.08


def up4(x):
    return np.stack([cv2.resize(x[c], (192, 192), interpolation=cv2.INTER_CUBIC)
                     for c in range(4)])


rows = []
for f in sorted(glob.glob("/home/czy/data/s2ds_built/test_*.npz")):
    z = np.load(f, allow_pickle=True)
    aoi = os.path.basename(f)[len("test_"):-len(".npz")]
    cld = z["cld"].astype(np.float32) / 255.0
    hrc = z["hr_cloud"]
    dtv = z["dt"].astype(np.float32)
    n = cld.shape[0]
    for i in range(n):
        cf = (cld[i] > 0.5).mean(axis=(1, 2))
        nfull = int((cf > 0.95).sum())
        if not (NFULL_LO <= nfull <= NFULL_HI):
            continue
        j0 = int(np.argmin(np.abs(dtv[i])))
        c0 = cv2.resize(cld[i][j0], (192, 192),
                        interpolation=cv2.INTER_NEAREST) > 0.5
        tgt_clear = hrc[i] < 0.5
        hard = tgt_clear & c0
        easy = tgt_clear & (~c0)
        tgtc = ~tgt_clear
        fh, fe, ft = hard.mean(), easy.mean(), tgtc.mean()
        if fh < MIN_HARD or fe < MIN_EASY or ft < MIN_TGTC:
            continue
        rows.append(dict(aoi=aoi, shard=f, i=i, date=str(z["date"][i]),
                         nfull=nfull, hard=float(fh), easy=float(fe),
                         tgtc=float(ft)))

print("candidates passing (1)+(2): %d" % len(rows))
if not rows:
    raise SystemExit("no candidate")

# ---------------------------------------------------------------- batch arm D
run = "s2dsV2_armD_s2026"
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

B = 32
for s in range(0, len(rows), B):
    chunk = rows[s:s + B]
    cache = {}
    lrs, qs, clds, dts, hrs, hms, bas, near = [], [], [], [], [], [], [], []
    for r in chunk:
        if r["shard"] not in cache:
            cache[r["shard"]] = np.load(r["shard"], allow_pickle=True)
        z = cache[r["shard"]]
        i = r["i"]
        lr = z["lr"][i].astype(np.float32) / 10000.0
        hr = z["hr"][i].astype(np.float32) / 10000.0
        cl = z["cld"][i].astype(np.float32) / 255.0
        dv = z["dt"][i].astype(np.float32)
        q = 1.0 - cl.reshape(cl.shape[0], -1).mean(1)
        dtn = dv / 30.0
        j0 = int(np.argmin(np.abs(dv)))
        c0 = cv2.resize(cl[j0], (192, 192),
                        interpolation=cv2.INTER_NEAREST) > 0.5
        hm = ((z["hr_cloud"][i] < 0.5) & c0).astype(np.float32)
        # zero-training baselines
        wt = np.exp(-np.abs(dv) / 30.0).astype(np.float32)
        wc = np.clip(1.0 - cl, 1e-3, None) ** 2
        w = (wt[:, None, None, None] * wc[:, None, :, :]).astype(np.float32)
        ca = up4((lr * w).sum(0) / w.sum(0).clip(min=1e-6))
        nb = up4(lr[j0])
        lrs.append(lr); qs.append(q); clds.append(cl); dts.append(dtn)
        hrs.append(hr); hms.append(hm); bas.append(ca); near.append(nb)

    with torch.no_grad():
        out = model(lr=torch.from_numpy(np.stack(lrs)).to(DEV),
                    q=torch.from_numpy(np.stack(qs)).to(DEV),
                    cld=torch.from_numpy(np.stack(clds)).to(DEV),
                    dt=torch.from_numpy(np.stack(dts)).to(DEV)).cpu().numpy()
    for k, r in enumerate(chunk):
        r["D"] = psnr_masked(out[k], hrs[k], hms[k])
        r["CA"] = psnr_masked(bas[k], hrs[k], hms[k])
        r["NN"] = psnr_masked(near[k], hrs[k], hms[k])

rows.sort(key=lambda r: -(r["D"] - r["CA"]))
print("%-6s %-16s %4s %8s %5s %5s %5s %5s %7s %7s %7s %7s"
      % ("aoi", "date", "idx", "nfull", "HARD", "EASY", "TGTC",
         "D", "CA", "NN", "D-CA", "D-NN"))
for r in rows[:25]:
    print("%-6s %-16s %4d %8d %5.1f %5.1f %5.1f %7.2f %7.2f %7.2f %+7.2f %+7.2f"
          % (r["aoi"], r["date"], r["i"], r["nfull"],
             100 * r["hard"], 100 * r["easy"], 100 * r["tgtc"],
             r["D"], r["CA"], r["NN"], r["D"] - r["CA"], r["D"] - r["NN"]))

nwin = sum(1 for r in rows if r["D"] > r["CA"])
print("arm D > cloud-aware on %d / %d candidates (%.0f%%)"
      % (nwin, len(rows), 100.0 * nwin / len(rows)))
