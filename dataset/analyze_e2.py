# -*- coding: utf-8 -*-
"""E2 -- place the published baselines against our arms on the same test set.

Every comparison is paired per scene, within a seed; seeds are never pooled.  The
baselines are cloud-blind and date-blind, so the informative contrasts are

    baseline - A   is a published fusion network better than our factor-free arm?
    baseline - B   does the published network reach our arm that only adds cloud gating?
    baseline - D   ... our arm with both factors?

Reported separately for the HARD and valid strata, at the fixed 30k endpoint.
"""
import os
import json
import numpy as np
from scipy import stats

RUNS = r"/mnt/e/论文2/runs_s2ds"
OUT = r"/mnt/e/论文2/dataset/e2_result.json"
MASKS = ["hard", "valid"]
BASELINES = ["s2dsE2_highresnet_s2026", "s2dsE2_highresnet_s2027",
             "s2dsE2_highresnet_s2028", "s2dsE2_rams_s2026"]


def load(run, mask, tag="fixed"):
    fn = "scene_psnr_test_%s%s.json" % (mask, ("_%s" % tag) if tag else "")
    p = os.path.join(RUNS, run, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mmean(d):
    v = np.array(list(d.values()), float)
    return float(np.nanmean(v))


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
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                n=int(m.sum()))


def seed_of(run):
    return int(run.rsplit("_s", 1)[1])


def main():
    res = {}
    for m in MASKS:
        print("\n=== mask=%s (fixed 30k endpoint) ===" % m)
        rows = []
        for b in BASELINES:
            bs = load(b, m)
            if bs is None:
                print("   %-26s MISSING" % b)
                continue
            s = seed_of(b)
            # NOTE on filename convention (this used to be wrong here): for the paper's
            # four main arms (s2dsR_* / s2dsV2_*) the UNTAGGED `scene_psnr_test_<mask>.json`
            # already holds the fixed-iteration endpoint, whereas the E2 baselines store
            # best as untagged and the fixed endpoint as `..._fixed.json`.  Passing
            # tag="fixed" for the arms therefore always returned None and this script
            # silently produced a JSON with no paired contrasts at all.
            A = load("s2dsV2_armA_s%d" % s, m, tag=None)
            B = load("s2dsV2_armB_s%d" % s, m, tag=None)
            D = load("s2dsV2_armD_s%d" % s, m, tag=None)
            r = dict(run=b, seed=s, psnr=mmean(bs))
            for nm, ref in [("A", A), ("B", B), ("D", D)]:
                if ref is not None:
                    r["minus_" + nm] = paired(ref, bs)
            rows.append(r)
            line = "   %-26s PSNR %6.3f " % (b, r["psnr"])
            for nm in ["A", "B", "D"]:
                if "minus_" + nm in r:
                    d = r["minus_" + nm]
                    line += "| -%s %+0.3f [%+0.3f,%+0.3f] p=%.2g " % (
                        nm, d["delta"], d["ci"][0], d["ci"][1], d["p"])
            print(line)
        res[m] = rows
        # seed-level summary for whichever baseline has more than one seed
        for tag, names in [("highresnet", [b for b in BASELINES if "highresnet" in b])]:
            ds = [r["minus_A"]["delta"] for r in rows
                  if r["run"] in names and "minus_A" in r]
            if len(ds) > 1:
                v = np.array(ds, float)
                tt = stats.ttest_1samp(v, 0.0)
                res[m + "_" + tag + "_vs_A"] = dict(
                    per_seed=[round(float(x), 4) for x in v], mean=float(v.mean()),
                    sd=float(v.std(ddof=1)), seed_level_p=float(tt.pvalue), n=len(v))
                print("   %s vs A over seeds: mean %+0.3f (sd %0.3f) p=%.3f n=%d"
                      % (tag, v.mean(), v.std(ddof=1), tt.pvalue, len(v)))
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
