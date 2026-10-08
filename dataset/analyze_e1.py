# -*- coding: utf-8 -*-
"""E1 -- attribute arm B's cloud gain to a route.

Reference arms (already trained, fixed 30k endpoint, verified in
check_e1_equivalence.py to be the same architecture):
    A  no cloud input at all
    B  all three q routes  (identical to main 2x2 arm B, 208,038 params)
New arms (this batch):
    B1 per-pixel routes only      206,918 params  (-1,120, the q_mlp)
    B2 frame-level route only     208,005 params  (-33,   the q_gate scalar)

Everything is paired per scene, per seed, seeds never pooled.
"""
import os
import json
import numpy as np
from scipy import stats

RUNS = r"/mnt/e/论文2/runs_s2ds"
OUT = r"/mnt/e/论文2/dataset/e1_result.json"
SEEDS = [2026, 2027, 2028]
MASKS = ["hard", "valid"]
PARAMS = {"A": 206885, "B": 208038, "B1": 206918, "B2": 208005}


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
        print("\n=== mask=%s ===" % m)
        A = {s: load("s2dsV2_armA_s%d" % s, m) for s in SEEDS}
        B = {s: load("s2dsV2_armB_s%d" % s, m) for s in SEEDS}
        B1 = {s: load("s2dsE1_armB1_s%d" % s, m, "fixed") for s in SEEDS}
        B2 = {s: load("s2dsE1_armB2_s%d" % s, m, "fixed") for s in SEEDS}
        rows = []
        for s in SEEDS:
            if None in (A[s], B[s], B1[s], B2[s]):
                print("   s%d MISSING, skipped" % s)
                continue
            r = dict(seed=s,
                     B_minus_A=paired(A[s], B[s]),
                     B1_minus_A=paired(A[s], B1[s]),
                     B2_minus_A=paired(A[s], B2[s]),
                     B_minus_B1=paired(B1[s], B[s]),
                     B_minus_B2=paired(B2[s], B[s]))
            rows.append(r)
            print("   s%d  B-A %+0.3f | B1(px)-A %+0.3f [p=%.2g] | B2(fr)-A %+0.3f [p=%.2g] "
                  "| B-B1 %+0.3f | B-B2 %+0.3f"
                  % (s, r["B_minus_A"]["delta"], r["B1_minus_A"]["delta"],
                     r["B1_minus_A"]["p"], r["B2_minus_A"]["delta"],
                     r["B2_minus_A"]["p"], r["B_minus_B1"]["delta"],
                     r["B_minus_B2"]["delta"]))
        if not rows:
            continue
        summ = {}
        for key in ["B_minus_A", "B1_minus_A", "B2_minus_A", "B_minus_B1", "B_minus_B2"]:
            v = np.array([r[key]["delta"] for r in rows], float)
            tt = stats.ttest_1samp(v, 0.0) if len(v) > 1 else None
            summ[key] = dict(per_seed=[round(float(x), 4) for x in v],
                             mean=float(v.mean()), sd=float(v.std(ddof=1)),
                             seed_level_p=(float(tt.pvalue) if tt is not None else None),
                             n_pos=int((v > 0).sum()), n=len(v))
            print("   %-11s mean %+0.3f (sd %0.3f)  per-seed %s  seed-level p=%.3f  "
                  "positive %d/%d"
                  % (key, v.mean(), v.std(ddof=1), [round(float(x), 3) for x in v],
                     summ[key]["seed_level_p"] or float("nan"),
                     summ[key]["n_pos"], len(v)))
        res[m] = dict(rows=rows, summary=summ, params=PARAMS)

    # pre-registered prediction: on HARD both routes are positive and B1 (per-pixel) > B2
    h = res.get("hard", {}).get("summary", {})
    if h:
        pos = bool(h["B1_minus_A"]["mean"] > 0 and h["B2_minus_A"]["mean"] > 0)
        order = bool(h["B1_minus_A"]["mean"] > h["B2_minus_A"]["mean"])
        res["hard_verdict"] = dict(both_positive=pos, pixel_larger_than_frame=order)
        print("\nHARD verdict: both routes positive=%s ; per-pixel > frame=%s" % (pos, order))
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
