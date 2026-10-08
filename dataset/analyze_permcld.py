# -*- coding: utf-8 -*-
"""E5: does q exploit per-pixel cloud structure, or only frame-level statistics?

arm B is evaluated twice with the SAME weights: once with the true SCL mask, once with the
mask block-permuted inside every frame (frame-level clear fraction preserved, spatial
correspondence destroyed).  arm A never sees cld, so it is the reference.

    gain_intact   = B_true      - A
    gain_scrambled= B_permuted  - A
Pre-registered expectation: gain_scrambled <= 0.5 * gain_intact  (q needs the pixels).
Secondary read-out: the part of the gain that SURVIVES scrambling is attributable to the
frame-level clear-fraction embedding (q component ii), which the permutation preserves.
"""
import os
import json
import numpy as np
from scipy import stats

RUNS = r"/mnt/e/论文2/runs_s2ds"
OUT = r"/mnt/e/论文2/dataset/permcld_result.json"
SEEDS = [2026, 2027, 2028]
MASKS = ["hard", "valid"]


def load(run, mask, tag=""):
    fn = "scene_psnr_test_%s%s.json" % (mask, ("_%s" % tag) if tag else "")
    p = os.path.join(RUNS, run, fn)
    return json.load(open(p, encoding="utf-8"))


def paired(a, b):
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    d = y - x
    t, p = stats.ttest_rel(y, x)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()), ci=[float(d.mean() - 1.96 * se),
                                          float(d.mean() + 1.96 * se)],
                p=float(p), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), n=len(d))


def main():
    res = {}
    for m in MASKS:
        rows = []
        for s in SEEDS:
            A = load("s2dsV2_armA_s%d" % s, m)
            Bt = load("s2dsV2_armB_s%d" % s, m)
            Bp = load("s2dsV2_armB_s%d" % s, m, "permcld")
            gi = paired(A, Bt)
            gs = paired(A, Bp)
            # B_true - B_permuted : how much of q's value the scrambling destroys
            loss = paired(Bp, Bt)
            rows.append(dict(seed=s, gain_intact=gi, gain_scrambled=gs,
                             B_true_minus_B_perm=loss,
                             retained_frac=(gs["delta"] / gi["delta"]
                                            if abs(gi["delta"]) > 1e-6 else float("nan"))))
            print("[%s s%d] gain intact %+0.3f | scrambled %+0.3f | retained %.0f%% | "
                  "B_true-B_perm %+0.3f (p=%.3g)"
                  % (m, s, gi["delta"], gs["delta"], 100 * rows[-1]["retained_frac"],
                     loss["delta"], loss["p"]))
        res[m] = rows
        fr = np.array([r["retained_frac"] for r in rows], float)
        ok = bool(np.nanmean(fr) <= 0.5)
        res[m + "_verdict"] = dict(mean_retained=float(np.nanmean(fr)),
                                   prediction_holds=ok)
        print("  -> mean retained fraction %.2f  prediction(<=0.5) holds: %s"
              % (np.nanmean(fr), ok))
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
