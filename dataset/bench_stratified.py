"""Stratified baseline check: where (if anywhere) does temporal fusion actually matter?

The current metric averages PSNR over all clear pixels of the target.  Most such pixels
are also clear in the temporally nearest frame, so copying that frame already works -- and
indeed the trivial `cloudaware_mean` fusion and every learned arm land at ~27.0 dB, i.e.
the learned network adds nothing over a weighted average.

This script splits the target's clear pixels into
  easy : clear in the nearest frame (|dt| minimal)      -> one frame suffices
  hard : cloudy in the nearest frame                    -> only other dates can help
and reports the cloud-aware weighted-mean PSNR on each subset.  If `hard` is where the
baseline collapses, that is the endpoint on which a multi-temporal model should be judged.
"""
import os
import glob
import json
import argparse

import numpy as np
import cv2

DST = "/home/czy/data/s2ds_built"
REFL = 10000.0


def up4(x):
    return np.stack([cv2.resize(x[c].astype(np.float32), (x.shape[2] * 4, x.shape[1] * 4),
                                interpolation=cv2.INTER_CUBIC) for c in range(x.shape[0])])


def psnr(pred, gt, mask, peak=1.0):
    m = mask.astype(bool)
    if int(m.sum()) < 32:
        return float("nan")
    return 10 * np.log10(peak ** 2 / (float(((pred - gt)[:, m] ** 2).mean()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--p", type=float, default=2.0)
    args = ap.parse_args()
    rows = []
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
                               interpolation=cv2.INTER_NEAREST)     # LR -> HR mask grid
            hard = valid & (c0_hr > 0.5)
            easy = valid & (c0_hr <= 0.5)
            wt = np.exp(-np.abs(dt[i].astype(np.float32)) / args.tau)[:, None, None]
            wc = np.maximum(1.0 - cld[i].astype(np.float32) / 255.0, 1e-3) ** args.p
            w = wt * wc
            syn = (lrf * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]
            pred = up4(syn)
            near = up4(lrf[j0])
            rows.append(dict(
                vf=float(valid.mean()), hf=float(hard.sum()), ef=float(easy.sum()),
                all_ca=psnr(pred, gt, valid), hard_ca=psnr(pred, gt, hard),
                hard_near=psnr(near, gt, hard), easy_ca=psnr(pred, gt, easy),
            ))
    d = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    def m(x):
        v = x[np.isfinite(x)]
        return (float(v.mean()), len(v)) if len(v) else (float("nan"), 0)
    print("split=%s  samples=%d" % (args.split, len(rows)))
    print("  target clear fraction (mean)      : %.3f" % d["vf"].mean())
    print("  hard pixel count per sample (mean) : %.0f  (%.1f%% of clear)"
          % (d["hf"].mean(), 100 * d["hf"].mean() / max(d["vf"].mean() * 192 * 192, 1)))
    for k, lab in [("all_ca", "cloudaware on ALL clear px"),
                   ("easy_ca", "cloudaware on EASY px (nearest frame clear)"),
                   ("hard_ca", "cloudaware on HARD px (nearest frame cloudy)"),
                   ("hard_near", "nearest-frame copy on HARD px")]:
        v, n = m(d[k])
        print("  %-42s mean=%7.3f  (n=%d)" % (lab, v, n))
    out = {"split": args.split, "n": len(rows),
           "means": {k: m(d[k])[0] for k in d}, "hard_frac_of_clear": float(
               d["hf"].mean() / max(d["vf"].mean() * 192 * 192, 1))}
    json.dump(out, open("/mnt/e/论文2/dataset/bench_stratified.json", "w"), indent=1)
    print("saved dataset/bench_stratified.json")


if __name__ == "__main__":
    main()
