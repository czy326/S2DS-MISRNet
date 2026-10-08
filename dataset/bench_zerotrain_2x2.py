"""Zero-training 2x2 counterpart of the learned factorial design.

Motivation (reviewer concern M4 / MINOR-2):  the HARD endpoint is defined as
"target clear AND temporally-nearest frame cloudy", which could in principle be
circular -- it selects exactly the pixels where a *cloud-aware* rule should win.
A cheap way to test whether the endpoint itself manufactures the result is to run
the SAME 2x2 in a fusion rule that has no parameters, no training and no access
to the endpoint definition at all:

    F1' = temporal weight  exp(-|dt|/tau)      (the zero-training analogue of dt)
    F2' = per-pixel cloud weight (1-cld)^p     (the zero-training analogue of q)

    a0 = uniform mean over the T frames        (no time, no cloud)
    t0 = time only
    c0 = cloud only
    ct = cloud x time                          == `cloudaware_mean`, the gatekeeper

If the zero-training 2x2 reproduces the learned pattern (cloud term large and
positive, time term ~ 0) on BOTH the HARD and the EASY stratum, then the endpoint
is not manufacturing the conclusion: the pattern is a property of the data, not of
the endpoint definition or of the learned model.

Usage:  python bench_zerotrain_2x2.py [--split test] [--tau 30] [--p 2]
"""
import os
import glob
import json
import argparse

import numpy as np
import cv2
from scipy import stats

DST = "/home/czy/data/s2ds_built"
REFL = 10000.0
S = 4  # upscaling factor


def up4(x):
    """(C,h,w) -> (C,4h,4w) bicubic."""
    return np.stack([cv2.resize(x[c].astype(np.float32),
                                (x.shape[2] * S, x.shape[1] * S),
                                interpolation=cv2.INTER_CUBIC)
                     for c in range(x.shape[0])])


def psnr(pred, gt, mask, peak=1.0):
    m = mask.astype(bool)
    if int(m.sum()) < 32:
        return float("nan")
    return 10 * np.log10(peak ** 2 / (float(((pred - gt)[:, m] ** 2).mean()) + 1e-12))


def pair(a, b):
    """ paired delta b - a, t-test, dz, normal-approx 95% CI """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return None
    d = b[m] - a[m]
    t, p = stats.ttest_rel(b[m], a[m])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--p", type=float, default=2.0)
    args = ap.parse_args()

    arms = ["a_uniform", "t_time", "c_cloud", "ct_cloud_time"]
    strata = ["all", "hard", "easy"]
    per = {(a, s): [] for a in arms for s in strata}
    meta = {"vf": [], "hf": [], "ef": []}

    for fp in sorted(glob.glob(os.path.join(DST, "%s_*.npz" % args.split))):
        z = np.load(fp)
        hr, lr, dt, cld, hc = z["hr"], z["lr"], z["dt"], z["cld"], z["hr_cloud"]
        for i in range(hr.shape[0]):
            gt = hr[i].astype(np.float32) / REFL
            lrf = lr[i].astype(np.float32) / REFL
            valid = (hc[i] == 0)
            j0 = int(np.argmin(np.abs(dt[i])))
            c0 = cld[i][j0].astype(np.float32) / 255.0
            c0_hr = cv2.resize(c0, (gt.shape[-1], gt.shape[-2]),
                               interpolation=cv2.INTER_NEAREST)
            hard = valid & (c0_hr > 0.5)
            easy = valid & (c0_hr <= 0.5)

            wt = np.exp(-np.abs(dt[i].astype(np.float32)) / args.tau)[:, None, None]
            wc = np.maximum(1.0 - cld[i].astype(np.float32) / 255.0, 1e-3) ** args.p
            ones = np.ones_like(wc)

            def wmean(w):
                return (lrf * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]

            syn = {"a_uniform": wmean(ones),
                   "t_time": wmean(wt * ones),
                   "c_cloud": wmean(ones * wc),
                   "ct_cloud_time": wmean(wt * wc)}

            masks = {"all": valid, "hard": hard, "easy": easy}
            for a in arms:
                pred = up4(syn[a])
                for s in strata:
                    per[(a, s)].append(psnr(pred, gt, masks[s]))
            meta["vf"].append(float(valid.mean()))
            meta["hf"].append(float(hard.sum()))
            meta["ef"].append(float(easy.sum()))

    n = len(per[(arms[0], "all")])
    print("split=%s  samples=%d  tau=%.1f  p=%.1f" % (args.split, n, args.tau, args.p))
    print("  mean hard pixels / sample : %.0f      mean easy pixels / sample : %.0f"
          % (np.mean(meta["hf"]), np.mean(meta["ef"])))
    print()

    print("  %-16s %10s %10s %10s" % ("arm", "ALL", "HARD", "EASY"))
    means = {}
    for a in arms:
        row = []
        for s in strata:
            v = np.array(per[(a, s)], dtype=np.float64)
            v = v[np.isfinite(v)]
            row.append(float(v.mean()))
            means[(a, s)] = float(v.mean())
        print("  %-16s %10.3f %10.3f %10.3f" % (a, row[0], row[1], row[2]))
    print()

    # zero-training main effects and interaction, per stratum
    eff = {}
    for s in strata:
        print("  --- stratum=%s ---" % s)
        defs = {
            "time  | no cloud  (t - a)": ("a_uniform", "t_time"),
            "time  | cloud     (ct - c)": ("c_cloud", "ct_cloud_time"),
            "cloud | no time   (c - a)": ("a_uniform", "c_cloud"),
            "cloud | time      (ct - t)": ("t_time", "ct_cloud_time"),
            "joint      (ct - a)": ("a_uniform", "ct_cloud_time"),
        }
        for lab, pr in defs.items():
            if pr is None:
                continue
            r = pair(per[(pr[0], s)], per[(pr[1], s)])
            if r is None:
                print("    %-34s n<5" % lab)
                continue
            eff["%s|%s" % (lab.strip(), s)] = r
            print("    %-34s %+8.3f  CI[%+7.3f,%+7.3f]  dz=%+5.2f  p=%.3g  n=%d"
                  % (lab, r["delta"], r["ci"][0], r["ci"][1], r["dz"], r["p"], r["n"]))
        # interaction: (ct - c) - (t - a)
        r = pair(list(np.array(per[("ct_cloud_time", s)]) - np.array(per[("c_cloud", s)])),
                 list(np.array(per[("t_time", s)]) - np.array(per[("a_uniform", s)])))
        if r is not None:
            eff["interaction (ct-c)-(t-a)|%s" % s] = r
            print("    %-34s %+8.3f  CI[%+7.3f,%+7.3f]  dz=%+5.2f  p=%.3g  n=%d"
                  % ("interaction (ct-c)-(t-a)", r["delta"], r["ci"][0], r["ci"][1],
                     r["dz"], r["p"], r["n"]))
        print()

    out = {"split": args.split, "n": int(n), "tau": args.tau, "p": args.p,
           "meta": {k: float(np.mean(v)) for k, v in meta.items()},
           "means": {"%s|%s" % (a, s): means[(a, s)] for a in arms for s in strata},
           "effects": eff}
    dst = "/mnt/e/论文2/dataset/bench_zerotrain_2x2.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("saved", dst)


if __name__ == "__main__":
    main()
