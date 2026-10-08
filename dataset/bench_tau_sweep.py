"""E4: sensitivity of the zero-training time weight to its shape (tau, p).

The paper's claim is "the time term is fully shadowed by the cloud term".  A reviewer
can object that this is an artefact of one arbitrary choice
    w = exp(-|dt|/tau) * (1-cld)^p      with tau = 30 d, p = 2.
This sweep re-runs the zero-training 2x2 over tau in {7, 15, 30, 60, 180} and p in
{1, 2, 4} and reports, for every (tau, p):
    ct - c : the marginal value of the time weight ON TOP OF the cloud weight  <- key
    t  - a : the value of the time weight when no cloud weight is present
    c  - a : the value of the cloud weight
Pre-registered expectation: |ct - c| <= 0.1 dB for every (tau, p).
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
S = 4


def up4(x):
    return np.stack([cv2.resize(x[c].astype(np.float32), (x.shape[2] * S, x.shape[1] * S),
                                interpolation=cv2.INTER_CUBIC) for c in range(x.shape[0])])


def psnr(pred, gt, mask, peak=1.0):
    m = mask.astype(bool)
    if int(m.sum()) < 32:
        return float("nan")
    return 10 * np.log10(peak ** 2 / (float(((pred - gt)[:, m] ** 2).mean()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--taus", default="7,15,30,60,180")
    ap.add_argument("--ps", default="1,2,4")
    args = ap.parse_args()

    shards = []
    for fp in sorted(glob.glob(os.path.join(DST, "%s_*.npz" % args.split))):
        shards.append(np.load(fp))
    print("split=%s shards=%d" % (args.split, len(shards)))

    # pre-extract per-sample tensors once (the sweep is the expensive part, not I/O)
    samples = []
    for z in shards:
        hr, lr, dt, cld, hc = z["hr"], z["lr"], z["dt"], z["cld"], z["hr_cloud"]
        for i in range(hr.shape[0]):
            gt = hr[i].astype(np.float32) / REFL
            lrf = lr[i].astype(np.float32) / REFL
            valid = (hc[i] == 0)
            j0 = int(np.argmin(np.abs(dt[i])))
            c0 = cv2.resize(cld[i][j0].astype(np.float32) / 255.0,
                            (gt.shape[-1], gt.shape[-2]), interpolation=cv2.INTER_NEAREST)
            samples.append(dict(
                gt=gt, lrf=lrf, valid=valid,
                hard=valid & (c0 > 0.5), easy=valid & (c0 <= 0.5),
                dta=np.abs(dt[i].astype(np.float32)),
                cldf=cld[i].astype(np.float32) / 255.0))
    print("samples=%d" % len(samples))

    out = {"split": args.split, "n": len(samples), "grid": []}
    for tau in [float(x) for x in args.taus.split(",")]:
        for p in [float(x) for x in args.ps.split(",")]:
            per = {k: [] for k in ["a", "t", "c", "ct"]}
            for s in samples:
                wt = np.exp(-s["dta"] / tau)[:, None, None]
                wc = np.maximum(1.0 - s["cldf"], 1e-3) ** p
                ones = np.ones_like(wc)

                def wm(w):
                    return (s["lrf"] * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]

                for k, w in [("a", ones), ("t", wt * ones), ("c", ones * wc), ("ct", wt * wc)]:
                    pred = up4(wm(w))
                    per[k].append([psnr(pred, s["gt"], s["valid"]),
                                   psnr(pred, s["gt"], s["hard"]),
                                   psnr(pred, s["gt"], s["easy"])])
            per = {k: np.array(v, dtype=np.float64) for k, v in per.items()}

            def eff(x, y, j):
                a_, b_ = per[x][:, j], per[y][:, j]
                m = np.isfinite(a_) & np.isfinite(b_)
                d = b_[m] - a_[m]
                t, pv = stats.ttest_rel(b_[m], a_[m])
                se = d.std(ddof=1) / np.sqrt(len(d))
                return dict(delta=float(d.mean()),
                            ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                            p=float(pv), n=int(m.sum()))

            row = {"tau": tau, "p": p,
                   "means": {k: [float(np.nanmean(per[k][:, j])) for j in range(3)]
                             for k in per}}
            for j, lab in enumerate(["all", "hard", "easy"]):
                row["ct-c_" + lab] = eff("c", "ct", j)
                row["t-a_" + lab] = eff("a", "t", j)
                row["c-a_" + lab] = eff("a", "c", j)
            out["grid"].append(row)
            print("tau=%5.0f p=%.0f | ct-c: all %+0.3f hard %+0.3f easy %+0.3f | "
                  "t-a: all %+0.3f hard %+0.3f easy %+0.3f"
                  % (tau, p,
                     row["ct-c_all"]["delta"], row["ct-c_hard"]["delta"], row["ct-c_easy"]["delta"],
                     row["t-a_all"]["delta"], row["t-a_hard"]["delta"], row["t-a_easy"]["delta"]))
            print("                  | p(ct-c): %.2f %.2f %.2f"
                  % (row["ct-c_all"]["p"], row["ct-c_hard"]["p"], row["ct-c_easy"]["p"]))

    dst = "/mnt/e/论文2/dataset/bench_tau_sweep.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("saved", dst)

    mx = max(abs(r[k]["delta"]) for r in out["grid"]
             for k in ("ct-c_all", "ct-c_hard", "ct-c_easy"))
    verdict = "PASS" if mx <= 0.10 else "CHECK"
    print("\nmax |ct-c| over the whole (tau,p) grid = %.3f dB  ->  %s"
          " (pre-registered expectation: <= 0.10 dB)" % (mx, verdict))


if __name__ == "__main__":
    main()
