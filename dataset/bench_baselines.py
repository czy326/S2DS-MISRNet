"""Baseline benchmark on the built dataset: how much headroom does the task have?

Baselines (all evaluated on the CLEAR pixels of the HR target, i.e. the validity mask):
  1. bicubic_nearest  : bicubic x4 upsample of the single frame closest in time
  2. bicubic_median   : bicubic x4 upsample of the temporal median of the T LR frames
  3. cloudaware_mean  : LR-domain weighted mean, w = exp(-|dt|/tau) * (1 - cloud)^p,
                        then bicubic upsample  (a "smart trivial" fusion)
  4. oracle_bestframe : the clear target pixel taken from the temporally nearest
                        frame's LR upsampled -- reported only as a sanity reference

Reported: mean/median PSNR over samples, paired t / Wilcoxon between baselines, and the
gain of cloudaware_mean over bicubic_median (the number a learned fusion has to beat).

Usage: python bench_baselines.py [--splits val,test] [--tau 30] [--p 2]
"""
import os
import json
import glob
import argparse

import numpy as np
import cv2
from scipy import stats

DST = "/home/czy/data/s2ds_built"
OUT = "/mnt/e/论文2/dataset/bench_baselines.json"
REFL = 10000.0


def up4(x):
    """(C,h,w) -> (C,4h,4w) bicubic.  cv2 treats 3-D input as (H,W,C), so loop channels."""
    return np.stack([cv2.resize(x[c].astype(np.float32),
                                (x.shape[2] * 4, x.shape[1] * 4),
                                interpolation=cv2.INTER_CUBIC) for c in range(x.shape[0])])


def psnr_masked(pred, gt, mask, peak=1.0):
    m = mask.astype(bool)
    if m.sum() < 32:
        return float("nan")
    d = (pred - gt)[:, m]
    mse = float((d ** 2).mean())
    return 10 * np.log10(peak ** 2 / (mse + 1e-12))


def score_sample(hr, lr, dt, cld, hr_cloud, tau, p):
    """hr (4,P,P) u16, lr (T,4,sl,sl) u16, dt (T,), cld (T,sl,sl) u8 0..255."""
    gt = hr.astype(np.float32) / REFL
    lrf = lr.astype(np.float32) / REFL
    valid = (hr_cloud == 0)
    i_near = int(np.argmin(np.abs(dt)))
    wt = np.exp(-np.abs(dt) / tau)[:, None, None]                     # (T,1,1)
    wc = np.maximum(1.0 - cld.astype(np.float32) / 255.0, 1e-3) ** p  # (T,sl,sl)

    def wmean(w):
        s = (lrf * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]
        return s

    out = {}
    out["bicubic_nearest"] = psnr_masked(up4(lrf[i_near]), gt, valid)
    out["bicubic_median"] = psnr_masked(up4(np.median(lrf, 0)), gt, valid)
    out["timeweighted_mean"] = psnr_masked(up4(wmean(wt * np.ones_like(wc))), gt, valid)
    out["cloudweighted_mean"] = psnr_masked(up4(wmean(np.ones_like(wt) * wc)), gt, valid)
    out["cloudaware_mean"] = psnr_masked(up4(wmean(wt * wc)), gt, valid)
    out["bicubic_firstframe"] = psnr_masked(up4(lrf[0]), gt, valid)
    out["_valid_frac"] = float(valid.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    res = {"tau": args.tau, "p": args.p, "splits": {}}
    keys = ["bicubic_firstframe", "bicubic_nearest", "bicubic_median", "timeweighted_mean", "cloudweighted_mean", "cloudaware_mean"]
    for split in args.splits.split(","):
        files = sorted(glob.glob(os.path.join(DST, "%s_*.npz" % split)))
        if not files:
            print("[%s] no shards" % split); continue
        per = {k: [] for k in keys}
        aois = []
        for fp in files:
            z = np.load(fp)
            hr, lr, dt = z["hr"], z["lr"], z["dt"]
            cld, hc = z["cld"], z["hr_cloud"]
            n = hr.shape[0] if not args.limit else min(args.limit, hr.shape[0])
            for i in range(n):
                s = score_sample(hr[i], lr[i], dt[i].astype(np.float32), cld[i], hc[i],
                                 args.tau, args.p)
                for k in keys:
                    per[k].append(s[k])
            aois.append((os.path.basename(fp), n))
        print("=== split=%s  shards=%d  samples=%d ===" % (split, len(files), len(per[keys[0]])))
        row = {}
        for k in keys:
            v = np.array(per[k], dtype=np.float64)
            ok = np.isfinite(v)
            print("  %-18s mean=%.4f  median=%.4f  sd=%.4f  n=%d"
                  % (k, np.nanmean(v), np.nanmedian(v), np.nanstd(v), int(ok.sum())))
            row[k] = dict(mean=float(np.nanmean(v)), median=float(np.nanmedian(v)),
                          sd=float(np.nanstd(v)), n=int(ok.sum()))
        # paired comparisons
        def pair(a, b):
            m = np.isfinite(per[a]) & np.isfinite(per[b])
            d = np.array(per[b])[m] - np.array(per[a])[m]
            if m.sum() < 5:
                return None
            t, p_ = stats.ttest_rel(np.array(per[b])[m], np.array(per[a])[m])
            se = d.std(ddof=1) / np.sqrt(len(d))
            return dict(delta=float(d.mean()), ci=[float(d.mean() - 1.96 * se),
                                                   float(d.mean() + 1.96 * se)],
                        p=float(p_), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                        n=int(m.sum()))
        cmps = {}
        for a, b in [("bicubic_nearest", "bicubic_median"),
                     ("bicubic_median", "cloudaware_mean"),
                     ("bicubic_nearest", "cloudaware_mean")]:
            r = pair(a, b)
            if r:
                cmps["%s -> %s" % (a, b)] = r
                print("  %-34s delta=%+.4f CI[%+.4f,%+.4f] dz=%+.2f p=%.3g n=%d"
                      % ("%s -> %s" % (a, b), r["delta"], r["ci"][0], r["ci"][1],
                         r["dz"], r["p"], r["n"]))
        res["splits"][split] = dict(per_baseline=row, comparisons=cmps,
                                    shards=[a[0] for a in aois])
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)
    if "test" in res["splits"]:
        c = res["splits"]["test"]["comparisons"].get("bicubic_median -> cloudaware_mean")
        if c:
            print("\nHEADROOM CHECK: best trivial fusion beats temporal median by "
                  "%+.4f dB (95%% CI [%+.4f,%+.4f])" % (c["delta"], c["ci"][0], c["ci"][1]))


if __name__ == "__main__":
    main()
