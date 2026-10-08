# -*- coding: utf-8 -*-
"""E5 (train-time arm): how much of q's gain survives a spatially meaningless cloud mask?

Three arm-B variants per seed, all at the same 30k fixed-iteration budget, all paired
per scene:

    A       s2dsV2_armA_s{seed}   never consumes cld              -> reference
    B_true  s2dsV2_armB_s{seed}   true per-pixel SCL mask         -> gain_intact
    B_perm  s2dsP_armB_s{seed}    8x8 block-permuted SCL mask     -> gain_scrambled

    retained = (B_perm - A) / (B_true - A)

The 8x8 block permutation is exact on a 48x48 mask: it preserves the frame-level clear
fraction q = 1 - mean(cld) bit-for-bit, and destroys only the spatial correspondence.
So `retained` is the share of q's gain carried by the frame-level q_mlp route, and
1 - retained is the share carried by the two per-pixel routes (feature suppression and
the cld channel of the per-pixel attention conv).

Pre-registered expectation: mean retained fraction <= 0.5.
"""
import os
import json
import numpy as np
from scipy import stats

RUNS = r"/mnt/e/论文2/runs_s2ds"
OUT = r"/mnt/e/论文2/dataset/permcld_train_result.json"
SEEDS = [2026, 2027, 2028]
MASKS = ["hard", "valid"]


def load(run, mask, tag=""):
    fn = "scene_psnr_test_%s%s.json" % (mask, ("_%s" % tag) if tag else "")
    p = os.path.join(RUNS, run, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def mean(d):
    v = np.array(list(d.values()), dtype=np.float64)
    return float(np.nanmean(v)), int(np.isfinite(v).sum())


def paired(a, b):
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = y - x
    if len(d) < 2:
        return None
    t, p = stats.ttest_rel(y, x)
    w = stats.wilcoxon(y, x).pvalue if len(d) > 10 else float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()), ci=[float(d.mean() - 1.96 * se),
                                           float(d.mean() + 1.96 * se)],
                p=float(p), wilcoxon_p=float(w),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), n=int(m.sum()))


def endpoint_check():
    """Confirm that scene_psnr_test_*.json in the V2 runs is the FIXED (30k) endpoint."""
    print("--- endpoint check: file means vs archived arm_means ---")
    snap = os.path.join(RUNS, "_snap_dtfix")
    for s in SEEDS:
        for m in MASKS:
            d = load("s2dsV2_armA_s%d" % s, m)
            if d is None:
                continue
            fm, n = mean(d)
            row = ["s%d/%s file=%0.3f(n=%d)" % (s, m, fm, n)]
            for tag in ["analysis_V2", "analysis_V2_fixed"]:
                p = os.path.join(snap, "%s_s%d_test_%s.json" % (tag, s, m))
                if os.path.exists(p):
                    a = json.load(open(p, encoding="utf-8"))
                    row.append("%s A=%0.3f" % (tag.replace("analysis_", ""),
                                               a["arm_means"]["A"]))
            print("   " + "  ".join(row))


def main():
    endpoint_check()
    res = {}
    for m in MASKS:
        print("\n=== mask=%s ===" % m)
        rows = []
        for s in SEEDS:
            A = load("s2dsV2_armA_s%d" % s, m)
            Bt = load("s2dsV2_armB_s%d" % s, m)
            Bp = load("s2dsP_armB_s%d" % s, m, "fixed") or load("s2dsP_armB_s%d" % s, m)
            if A is None or Bt is None or Bp is None:
                print("   s%d MISSING (A=%s Bt=%s Bp=%s)" % (s, A is not None,
                                                             Bt is not None, Bp is not None))
                continue
            gi = paired(A, Bt)
            gs = paired(A, Bp)
            loss = paired(Bp, Bt)
            fr = gs["delta"] / gi["delta"] if abs(gi["delta"]) > 1e-6 else float("nan")
            rows.append(dict(seed=s, gain_intact=gi, gain_scrambled=gs,
                             B_true_minus_B_perm=loss, retained_frac=fr))
            print("   s%d  intact %+0.3f [%+0.3f,%+0.3f] p=%.2g | scrambled %+0.3f "
                  "[%+0.3f,%+0.3f] p=%.2g | retained %0.0f%% | Btrue-Bperm %+0.3f p=%.2g"
                  % (s, gi["delta"], gi["ci"][0], gi["ci"][1], gi["p"],
                     gs["delta"], gs["ci"][0], gs["ci"][1], gs["p"], 100 * fr,
                     loss["delta"], loss["p"]))
        if not rows:
            continue
        fr = np.array([r["retained_frac"] for r in rows], float)
        ok = bool(np.nanmean(fr) <= 0.5)
        seed_t = stats.ttest_1samp(fr, 0.5) if len(fr) > 1 else None
        res[m] = dict(rows=rows,
                      mean_retained=float(np.nanmean(fr)),
                      per_seed_retained=[float(x) for x in fr],
                      prediction_holds=ok,
                      seed_level_t_vs_0p5=(float(seed_t.pvalue)
                                           if seed_t is not None else None))
        print("   -> mean retained %.2f  (per seed %s)  prediction(<=0.5) holds: %s"
              % (np.nanmean(fr), [round(float(x), 2) for x in fr], ok))
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
