# -*- coding: utf-8 -*-
"""MAJOR (v4 review): rule out the misregistration confound for MuS2's negative
time effect.

Mechanism under test: with cross-sensor misregistration, time-decay weighting
concentrates weight on a few frames and amplifies misalignment, while uniform
averaging dilutes it.  If that (not "temporal nearness is a bad reliability
proxy") drives t - a < 0, then within WELL-REGISTERED scenes the effect should
return to 0 or turn positive.

Per scene we estimate the sub-pixel shift between the bicubic-upscaled nearest
low-cloud LR frame and the WV-2 HR reference (phase correlation with Hann window
on a high-pass filtered green band), then stratify scenes by |shift| (median
split) and recompute the zero-training t - a contrast within each stratum.
A continuous Spearman correlation between |shift| and per-scene (t - a) is also
reported.

Reuses the exact arm definitions of bench_mus2_zerotrain.py (gamma fitted once
on the control arm, reused unchanged).
"""
import os, glob, json, sys
import numpy as np
import cv2
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

CACHE = "/home/czy/data/mus2/cache"
S = 3
REFL = 10000.0
TAUS = [0.08, 0.25, 0.50]
P = 2.0
OUT = "/mnt/e/论文2/dataset/bench_mus2_regstrat.json"


def up3(x):
    return np.stack([cv2.resize(x[c].astype(np.float32),
                                (x.shape[2] * S, x.shape[1] * S),
                                interpolation=cv2.INTER_CUBIC)
                     for c in range(x.shape[0])])


def highpass(img, k=31):
    return img - cv2.GaussianBlur(img, (k, k), 0)


def reg_metrics(fp):
    """Per-scene registration proxy: phase correlation between the nearest
    low-cloud LR frame (upscaled) and HR, on the green band."""
    z = np.load(fp)
    lr = z["lr"].astype(np.float32) / REFL
    hr = z["hr"].astype(np.float32) / 255.0
    cld = z["cld"].astype(np.float32)
    dt = z["dt"].astype(np.float32)
    T = lr.shape[0]
    order = np.argsort(np.abs(dt))
    # nearest frame with mean cloud < 0.2, else the least cloudy of all
    pick = None
    for i in order:
        if cld[i].mean() < 0.2:
            pick = int(i)
            break
    if pick is None:
        pick = int(order[int(np.argmin(cld.mean(axis=(1, 2))))])
    ref = up3(lr[pick])
    g_lr = highpass(ref[1])
    g_hr = highpass(hr[1])
    win = cv2.createHanningWindow(g_lr.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(g_hr, g_lr, win)
    return dict(file=os.path.basename(fp), pick_frame=int(pick),
                pick_cloud=float(cld[pick].mean()),
                shift=float(np.hypot(dx, dy)), dx=float(dx), dy=float(dy),
                resp=float(resp))


def psnr_full(pred, gt, gamma, peak=1.0):
    d = (gamma * pred - gt)
    return 10 * np.log10(peak ** 2 / (float((d ** 2).mean()) + 1e-12))


def main():
    files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
    print("scenes:", len(files))

    # ---- per-scene arms (a and t only) ----
    preds, gts = [], []
    for fp in files:
        z = np.load(fp)
        lr = z["lr"].astype(np.float32) / REFL
        hr = z["hr"].astype(np.float32) / 255.0
        cld = z["cld"].astype(np.float32)
        dt = z["dt"].astype(np.float32)
        wc = np.maximum(1.0 - cld, 1e-3) ** P

        def wmean(w):
            return (lr * w[:, None, :, :]).sum(0) / w.sum(0)[None, :, :]

        syn_a = wmean(np.ones_like(wc))
        tt = {t: wmean(np.exp(-np.abs(dt) / t)[:, None, None] * np.ones_like(wc))
              for t in TAUS}
        preds.append((fp, {t: up3(v) for t, v in tt.items()}, up3(syn_a)))
        gts.append(hr)

    num = den = 0.0
    for _, o, _ in preds:
        pass
    # gamma on a_uniform over all scenes
    # gamma on a_uniform over all scenes
    for (fp, tt, sa), hr in zip(preds, gts):
        num += float((sa * hr).sum())
        den += float((sa ** 2).sum())
    gamma = num / max(den, 1e-12)
    print("global gamma (fitted on a_uniform) = %.4f" % gamma)

    reg = [reg_metrics(fp) for fp in files]
    shifts = np.array([r["shift"] for r in reg])
    resps = np.array([r["resp"] for r in reg])
    print("|shift| px: median %.3f  IQR %.3f~%.3f  max %.3f"
          % (np.median(shifts), np.percentile(shifts, 25), np.percentile(shifts, 75), shifts.max()))
    print("resp      : median %.3f  IQR %.3f~%.3f"
          % (np.median(resps), np.percentile(resps, 25), np.percentile(resps, 75)))

    med = float(np.median(shifts))
    good = shifts <= med          # well-registered half
    # stricter split as robustness check
    good_strict = shifts <= 0.5

    report = {"gamma": float(gamma), "n": len(files),
              "shift_median": med,
              "reg": reg, "taus": {}}

    for tau in TAUS:
        pa, pt = [], []
        for (fp, tt, sa), hr in zip(preds, gts):
            pa.append(psnr_full(sa, hr, gamma))
            pt.append(psnr_full(tt[tau], hr, gamma))
        pa = np.array(pa)
        pt = np.array(pt)
        d = pt - pa

        def blk(mask, name):
            m = mask & np.isfinite(d)
            if m.sum() < 5:
                return dict(name=name, n=int(m.sum()))
            t, p = stats.ttest_rel(pt[m], pa[m])
            se = d[m].std(ddof=1) / np.sqrt(int(m.sum()))
            return dict(name=name, n=int(m.sum()), delta=float(d[m].mean()),
                        ci=[float(d[m].mean() - 1.96 * se), float(d[m].mean() + 1.96 * se)],
                        p=float(p), win=float((d[m] > 0).mean()))

        rho, rho_p = stats.spearmanr(shifts, d)
        rho_r, rho_r_p = stats.spearmanr(resps, d)
        ok = np.isfinite(d) & (shifts < 20.0)      # drop proxy failures
        t_ex, p_ex = stats.ttest_rel(pt[ok], pa[ok])
        se_ex = d[ok].std(ddof=1) / np.sqrt(int(ok.sum()))
        out = dict(all=blk(np.ones(len(d), bool), "all"),
                   good=blk(good, "good(half)"),
                   poor=blk(~good, "poor(half)"),
                   good_strict=blk(good_strict, "good(shift<=0.5px)"),
                   all_excl_outlier=dict(n=int(ok.sum()), delta=float(d[ok].mean()),
                                         ci=[float(d[ok].mean() - 1.96 * se_ex),
                                             float(d[ok].mean() + 1.96 * se_ex)],
                                         p=float(p_ex)),
                   per_scene_delta=[float(x) for x in d],
                   spearman_shift=dict(rho=float(rho), p=float(rho_p)),
                   spearman_resp=dict(rho=float(rho_r), p=float(rho_r_p)))
        report["taus"]["%.2f" % tau] = out
        print("\n=== tau = %.2f yr ===" % tau)
        for k in ["all", "good", "poor", "good_strict"]:
            b = out[k]
            if "delta" in b:
                print("  %-22s n=%3d  t-a %+7.3f CI[%+7.3f,%+7.3f] p=%.3g win=%.2f"
                      % (b["name"], b["n"], b["delta"], b["ci"][0], b["ci"][1], b["p"], b["win"]))
            else:
                print("  %-22s n=%d (too few)" % (b["name"], b["n"]))
        b = out["all_excl_outlier"]
        print("  %-22s n=%3d  t-a %+7.3f CI[%+7.3f,%+7.3f] p=%.3g"
              % ("all excl |shift|>20", b["n"], b["delta"], b["ci"][0], b["ci"][1], b["p"]))
        print("  Spearman(|shift|, t-a): rho=%+.3f p=%.3g | Spearman(resp, t-a): rho=%+.3f p=%.3g"
              % (rho, rho_p, rho_r, rho_r_p))

    json.dump(report, open(OUT, "w"), indent=1)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
