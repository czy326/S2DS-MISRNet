# -*- coding: utf-8 -*-
"""E7 -- BreizhSR / MISR-S2 on S2DS, under this paper's fixed-iteration protocol.

Endpoint conventions (they differ between run families, so they are spelled out):
  * the paper's four main arms  -> s2dsR_* / s2dsV2_*  : scene_psnr_test_<mask>.json
    (for those runs the UNTAGGED file already holds the fixed-iteration endpoint)
  * the E2 baselines and the E7 runs -> scene_psnr_test_<mask>_fixed.json

Contrasts, all paired per sample, seeds never pooled:

    breizhsr      - A                a published time-aware method vs our factor-free arm
    breizhsr      - C                MATCHED FACTORS (both get dt, neither gets q): the
                                     cleanest test of "does a dedicated temporal encoder
                                     extract more from the dates than our dt factor does?"
    breizhsr      - breizhsr_nope    the paper's central claim: the target-relative
                                     temporal positional encoding
    breizhsr      - highresnet       same encoder, L-TAE fusion instead of recursive
    breizhsr      - D                is cloud gating worth more than a time encoder
"""
import json
import os

import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/e7_result.json"
MASKS = ["hard", "valid"]
SEEDS = [2026, 2027, 2028]


def load(run, mask, fixed):
    fn = "scene_psnr_test_%s%s.json" % (mask, "_fixed" if fixed else "")
    p = os.path.join(RUNS, run, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def paired(ref, cur):
    ks = sorted(set(ref) & set(cur))
    x = np.array([ref[k] for k in ks], float)
    y = np.array([cur[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = y - x
    if len(d) < 2:
        return None
    t, p = stats.ttest_rel(y, x)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                win=float((d > 0).mean()), n=int(m.sum()),
                median=float(np.median(d)))


def arm_of(run, seed, arm, mask):
    pre = "s2dsR" if (seed in (2026, 2027) and arm in "AB") else "s2dsV2"
    return load("%s_arm%s_s%d" % (pre, arm, seed), mask, fixed=False)


def main():
    res = {"protocol": "fixed-iteration (last.pt, 30k), paired per sample, "
                       "seeds never pooled",
           "models": "breizhsr = HighRes-net encoder + L-TAE + target-relative temporal PE "
                     "(3 seeds); breizhsr_nope = same weights, PE off (3 seeds)"}
    for mask in MASKS:
        print("\n=== mask=%s (fixed 30k endpoint) ===" % mask)
        per_seed = {}
        rows = []
        for s in SEEDS:
            br = load("s2dsE7_breizhsr_s%d" % s, mask, fixed=True)
            npn = load("s2dsE7_breizhsr_nope_s%d" % s, mask, fixed=True)
            hn = load("s2dsE2_highresnet_s%d" % s, mask, fixed=True)
            A = arm_of(None, s, "A", mask)
            C = arm_of(None, s, "C", mask)
            D = arm_of(None, s, "D", mask)
            if br is None:
                print("   seed %d: breizhsr MISSING" % s)
                continue
            r = dict(seed=s, psnr=float(np.nanmean(list(br.values()))))
            pr = {}
            for nm, ref in [("A", A), ("C", C), ("D", D), ("nope", npn),
                            ("highresnet", hn)]:
                if ref is None:
                    continue
                d = paired(ref, br)
                if d:
                    r["minus_" + nm] = d
                    pr[nm] = d["delta"]
            rows.append(r)
            per_seed[s] = pr
            line = "   seed %d  breizhsr PSNR %6.3f " % (s, r["psnr"])
            for nm in ["A", "C", "D", "nope", "highresnet"]:
                if nm in pr:
                    line += "| -%s %+0.3f " % (nm, pr[nm])
            print(line)
            extra = "        detail:"
            for nm in ["A", "C", "nope"]:
                if nm in r:
                    d = r["minus_" + nm]
                    extra += ("  [%s p=%.3g win=%.0f%% n=%d]" %
                              (nm, d["p"], 100 * d["win"], d["n"]))
            print(extra)
            if npn is not None:
                print("        nope PSNR %6.3f" % float(np.nanmean(list(npn.values()))))

        seed_level = {}
        for nm in ["A", "C", "D", "nope", "highresnet"]:
            v = np.array([per_seed[s][nm] for s in per_seed if nm in per_seed[s]], float)
            if len(v) < 2:
                continue
            t, p = stats.ttest_1samp(v, 0.0)
            same = bool(np.all(v > 0) or np.all(v < 0))
            seed_level[nm] = dict(per_seed=[round(float(x), 4) for x in v],
                                  mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                  p=float(p), n=len(v), all_same_sign=same)
            print("   seed-level breizhsr - %-11s %s  mean %+0.3f sd %0.3f p=%.3f n=%d %s"
                  % (nm, ["%+0.3f" % x for x in v], v.mean(), v.std(ddof=1), p, len(v),
                     "same sign" if same else "SIGN FLIPS"))
        res[mask] = dict(rows=rows, seed_level=seed_level)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
