# -*- coding: utf-8 -*-
"""E5, evaluation-time variant: feed an ALREADY TRAINED arm B a scrambled cloud mask.

This is the cheaper half of E5 and it needs the trained weights, so it only runs for the
seeds whose last.pt still exists (seed 2028).  The train-time variant
(analyze_permcld_train.py) covers all three seeds.

    gain_intact    B_true      - A
    gain_scrambled B_permuted  - A
    retained       gain_scrambled / gain_intact
"""
import os
import json
import numpy as np
from scipy import stats

RUNS = r"/mnt/e/论文2/runs_s2ds"
OUT = r"/mnt/e/论文2/dataset/permcld_evaltime_result.json"
SEEDS = [2026, 2027, 2028]
MASKS = ["hard", "valid"]


def load(run, mask, tag=""):
    fn = "scene_psnr_test_%s%s.json" % (mask, ("_%s" % tag) if tag else "")
    p = os.path.join(RUNS, run, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def paired(a, b):
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = y - x
    t, p = stats.ttest_rel(y, x)
    w = stats.wilcoxon(y, x).pvalue if len(d) > 10 else float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), wilcoxon_p=float(w),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), n=int(m.sum()))


def main():
    res = {}
    for m in MASKS:
        rows = []
        for s in SEEDS:
            A = load("s2dsV2_armA_s%d" % s, m)
            Bt = load("s2dsV2_armB_s%d" % s, m)
            Bp = load("s2dsV2_armB_s%d" % s, m, "permcld")
            if None in (A, Bt, Bp):
                print("[%s s%d] skipped (weights or permuted eval missing)" % (m, s))
                continue
            gi, gs = paired(A, Bt), paired(A, Bp)
            lo = paired(Bp, Bt)
            fr = gs["delta"] / gi["delta"] if abs(gi["delta"]) > 1e-6 else float("nan")
            rows.append(dict(seed=s, gain_intact=gi, gain_scrambled=gs,
                             B_true_minus_B_perm=lo, retained_frac=fr))
            print("[%s s%d] intact %+0.3f [%+0.3f,%+0.3f] p=%.2g | scrambled %+0.3f "
                  "[%+0.3f,%+0.3f] p=%.2g | retained %0.0f%% | Btrue-Bperm %+0.3f p=%.2g"
                  % (m, s, gi["delta"], gi["ci"][0], gi["ci"][1], gi["p"],
                     gs["delta"], gs["ci"][0], gs["ci"][1], gs["p"], 100 * fr,
                     lo["delta"], lo["p"]))
        if rows:
            fr = np.array([r["retained_frac"] for r in rows], float)
            res[m] = dict(rows=rows, mean_retained=float(np.nanmean(fr)),
                          prediction_holds=bool(np.nanmean(fr) <= 0.5))
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
