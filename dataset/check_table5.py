# -*- coding: utf-8 -*-
"""Audit 表 5: recompute every run's paired delta vs the zero-training
cloud-aware baseline, for both the HARD and valid strata, and for BOTH
checkpoint endpoints that exist on disk.

Prints a compact grid so the manuscript's numbers can be checked cell by cell.
"""
import os, json, sys, io
import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUNS = "/mnt/e/论文2/runs_s2ds"
BASE_P = "/mnt/e/论文2/dataset/zero_baseline_scene.json"
BASE = json.load(open(BASE_P, encoding="utf-8"))

RUN_LIST = [
    "s2dsE2_highresnet_s2026", "s2dsE2_highresnet_s2027", "s2dsE2_highresnet_s2028",
    "s2dsE2_rams_s2026",
    "s2dsV2_armA_s2026", "s2dsV2_armA_s2027", "s2dsV2_armA_s2028",
    "s2dsV2_armD_s2026", "s2dsV2_armD_s2027", "s2dsV2_armD_s2028",
    "s2dsE9_highresnet_cld_s2026", "s2dsE9_highresnet_cld_s2027", "s2dsE9_highresnet_cld_s2028",
]


def files(run, mask):
    """return list of (label, path) for every endpoint json that exists"""
    out = []
    for tag, fn in [("best", "scene_psnr_test_%s.json" % mask),
                    ("fixed", "scene_psnr_test_%s_fixed.json" % mask)]:
        p = os.path.join(RUNS, run, fn)
        if os.path.exists(p):
            out.append((tag, p))
    return out


def stats_vs_base(d, b):
    ks = sorted(set(d) & set(b))
    x = np.array([b[k] for k in ks], float)
    y = np.array([d[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return None
    diff = y - x
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t, p = stats.ttest_rel(y, x)
    return dict(delta=float(diff.mean()),
                ci=[float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                p=float(p), n=int(m.sum()), win=float((diff > 0).mean()),
                psnr=float(y.mean()))


def main():
    for mask in ["hard", "valid"]:
        b = BASE[mask]
        bv = np.array(list(b.values()), float)
        print("\n" + "=" * 96)
        print("mask = %s   baseline mean = %.3f   n = %d"
              % (mask, np.nanmean(bv), np.isfinite(bv).sum()))
        print("=" * 96)
        print("%-30s %-6s %8s %8s %18s %6s %6s" %
              ("run", "tag", "PSNR", "delta", "95% CI", "win", "n"))
        for run in RUN_LIST:
            if not os.path.isdir(os.path.join(RUNS, run)):
                continue
            fs = files(run, mask)
            if not fs:
                print("%-30s %-6s   MISSING" % (run, "-"))
                continue
            for tag, p in fs:
                d = json.load(open(p, encoding="utf-8"))
                r = stats_vs_base(d, b)
                if r is None:
                    print("%-30s %-6s   n<2" % (run, tag))
                    continue
                print("%-30s %-6s %8.3f %+8.3f [%+6.3f,%+6.3f] %5.1f%% %6d"
                      % (run, tag, r["psnr"], r["delta"], r["ci"][0], r["ci"][1],
                         r["win"] * 100, r["n"]))


if __name__ == "__main__":
    main()
