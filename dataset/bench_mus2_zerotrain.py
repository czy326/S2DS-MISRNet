"""Checklist item 5: external validity of the factor pattern on MuS2 (zero training).

Runs the SAME parameter-free 2x2 that `bench_zerotrain_2x2.py` runs on S2DS, but on
the 91 cached MuS2 scenes (real S2 time series -> WV-2 HR, 3x). No training, no
learned parameters, no endpoint definition borrowed from S2DS, so a replication
here cannot be an artifact of the S2DS pipeline.

    a  = uniform mean over the T frames          (no time, no cloud)
    t  = time only        exp(-|dt|/tau)
    c  = cloud only       (1-cld)^p
    ct = cloud x time                            (the full rule)

Differences from S2DS that must be stated in the paper:
  * MuS2 has no HR-side cloud mask and its HR comes from a different sensor/date,
    so ABSOLUTE PSNR is not comparable with S2DS; only within-MuS2 contrasts are.
  * Strata are therefore defined operationally on the LR cloud masks:
        clean    = no frame cloudy at this pixel
        contam   = >=1 frame cloudy AND >=1 frame clear at this pixel
  * MuS2 lr is uint16 reflectance (/10000) while hr_resized is uint8 (/255), so a
    single global gain gamma is fitted ONCE on the control arm `a` (least squares
    over all scenes) and reused unchanged for every other arm.  gamma is therefore
    arm-independent and cannot manufacture a contrast.

Usage:  /home/czy/miniconda3/envs/emssm/bin/python bench_mus2_zerotrain.py [--p 2]
"""
import os
import glob
import json
import argparse

import numpy as np
import cv2
from scipy import stats

CACHE = "/home/czy/data/mus2/cache"
S = 3
REFL = 10000.0
ARMS = ["a_uniform", "t_time", "c_cloud", "ct_cloud_time"]
STRATA = ["all", "clean", "contam"]


def up3(x):
    """(C,h,w) -> (C,3h,3w) bicubic."""
    return np.stack([cv2.resize(x[c].astype(np.float32),
                                (x.shape[2] * S, x.shape[1] * S),
                                interpolation=cv2.INTER_CUBIC)
                     for c in range(x.shape[0])])


def psnr(pred, gt, mask, peak=1.0):
    m = mask.astype(bool)
    if int(m.sum()) < 500:
        return float("nan")
    return 10 * np.log10(peak ** 2 / (float(((pred - gt)[:, m] ** 2).mean()) + 1e-12))


def pair(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return None
    d = b[m] - a[m]
    t, p = stats.ttest_rel(b[m], a[m])
    try:
        w = stats.wilcoxon(d).pvalue
    except Exception:
        w = float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), wilcoxon=float(w),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                win=float((d > 0).mean()), n=int(m.sum()))


def build(fp, tau, p):
    z = np.load(fp)
    lr = z["lr"].astype(np.float32) / REFL            # T,3,H,W
    hr = z["hr"].astype(np.float32) / 255.0           # 3,3H,3W
    cld = z["cld"].astype(np.float32)                 # T,H,W in {0,1}
    dt = z["dt"].astype(np.float32)                   # T,  in years
    T = lr.shape[0]

    wt = np.exp(-np.abs(dt) / tau)[:, None, None].astype(np.float32)
    wc = np.maximum(1.0 - cld, 1e-3) ** p
    ones = np.ones_like(wc)

    def wmean(w):
        return (lr * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]

    syn = {"a_uniform": wmean(ones),
           "t_time": wmean(wt * ones),
           "c_cloud": wmean(ones * wc),
           "ct_cloud_time": wmean(wt * wc)}

    ncl = cld.sum(0)                                   # number of cloudy frames per pixel
    clean = (ncl == 0)
    contam = (ncl > 0) & (ncl < T)
    Hh, Wh = hr.shape[-2:]
    masks = {}
    for k, m in (("all", np.ones((Hh, Wh), bool)), ("clean", clean), ("contam", contam)):
        masks[k] = cv2.resize(m.astype(np.uint8), (Wh, Hh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    out = {a: up3(syn[a]) for a in ARMS}
    return out, hr, masks, float(cld.mean()), float(clean.mean()), float(contam.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--taus", default="0.08,0.25,0.50")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
    print("scenes:", len(files))

    report = {}
    for tau in [float(x) for x in args.taus.split(",")]:
        preds, gts, mks = [], [], {s: [] for s in STRATA}
        cf, cl, ct = [], [], []
        for fp in files:
            o, hr, masks, c, a1, a2 = build(fp, tau, args.p)
            preds.append(o)
            gts.append(hr)
            for s in STRATA:
                mks[s].append(masks[s])
            cf.append(c); cl.append(a1); ct.append(a2)

        # ---- single global gain fitted on the CONTROL arm only ----
        num = 0.0
        den = 0.0
        for o, hr in zip(preds, gts):
            num += float((o["a_uniform"] * hr).sum())
            den += float((o["a_uniform"] ** 2).sum())
        gamma = num / max(den, 1e-12)
        print("tau=%.2f  global gamma (fitted on a_uniform) = %.4f" % (tau, gamma))
        print("  mean cloud fraction %.4f | clean pixels %.3f | contaminated pixels %.3f"
              % (np.mean(cf), np.mean(cl), np.mean(ct)))

        per = {}
        for a in ARMS:
            for s in STRATA:
                per[(a, s)] = [psnr(gamma * o[a], hr, m)
                               for o, hr, m in zip(preds, gts, mks[s])]

        print("  %-16s %10s %10s %10s" % ("arm", "ALL", "CLEAN", "CONTAM"))
        means = {}
        for a in ARMS:
            row = []
            for s in STRATA:
                v = np.array(per[(a, s)], dtype=np.float64)
                v = v[np.isfinite(v)]
                row.append(float(v.mean())); means["%s|%s" % (a, s)] = float(v.mean())
            print("  %-16s %10.3f %10.3f %10.3f" % (a, row[0], row[1], row[2]))
        print()

        eff = {}
        for s in STRATA:
            print("  --- stratum=%s ---" % s)
            defs = {
                "time  | no cloud  (t - a)": ("a_uniform", "t_time"),
                "time  | cloud     (ct - c)": ("c_cloud", "ct_cloud_time"),
                "cloud | no time   (c - a)": ("a_uniform", "c_cloud"),
                "cloud | time      (ct - t)": ("t_time", "ct_cloud_time"),
                "joint      (ct - a)": ("a_uniform", "ct_cloud_time"),
            }
            for lab, pr in defs.items():
                r = pair(per[(pr[0], s)], per[(pr[1], s)])
                if r is None:
                    print("    %-30s n<5" % lab); continue
                eff["%s|%s" % (lab, s)] = r
                print("    %-30s %+8.3f CI[%+7.3f,%+7.3f] dz=%+6.2f p=%.3g win=%.2f n=%d"
                      % (lab, r["delta"], r["ci"][0], r["ci"][1], r["dz"], r["p"], r["win"], r["n"]))
            r = pair(list(np.array(per[("ct_cloud_time", s)]) - np.array(per[("c_cloud", s)])),
                     list(np.array(per[("t_time", s)]) - np.array(per[("a_uniform", s)])))
            if r is not None:
                eff["interaction (ct-c)-(t-a)|%s" % s] = r
                print("    %-30s %+8.3f CI[%+7.3f,%+7.3f] dz=%+6.2f p=%.3g win=%.2f n=%d"
                      % ("interaction (ct-c)-(t-a)", r["delta"], r["ci"][0], r["ci"][1],
                         r["dz"], r["p"], r["win"], r["n"]))
            print()

        report["tau=%.2f" % tau] = dict(
            gamma=float(gamma), p=args.p, n=int(len(files)),
            cloud_fraction=float(np.mean(cf)),
            clean_pixel_fraction=float(np.mean(cl)),
            contam_pixel_fraction=float(np.mean(ct)),
            means=means, effects=eff)

    dst = "/mnt/e/论文2/dataset/bench_mus2_zerotrain.json"
    json.dump(report, open(dst, "w"), indent=1)
    print("saved", dst)


if __name__ == "__main__":
    main()
