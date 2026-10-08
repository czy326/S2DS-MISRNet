# -*- coding: utf-8 -*-
"""E2 -- paired comparison of the two learned MISR baselines against the
zero-training cloud-aware mean and against arm D, per sample, per seed.

analyze_e2.py only prints means.  A mean gap is not enough here: we want the
paired CI, the p-value, and the per-sample win rate, on the same footing as
everything else in the paper (fixed-iteration endpoint, HARD / valid masks).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/e/论文2/figures")
from winrate_vs_baseline import baselines, runof        # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
E2 = [("s2dsE2_highresnet_s2026", "HighRes-net", 2026),
      ("s2dsE2_highresnet_s2027", "HighRes-net", 2027),
      ("s2dsE2_highresnet_s2028", "HighRes-net", 2028),
      ("s2dsE2_rams_s2026", "RAMS", 2026)]


def paired(run, mask, base):
    d = json.load(open(os.path.join(RUNS, run, "scene_psnr_test_%s_fixed.json"
                                    % mask)))
    keys = [k for k in d if k in base and d[k] == d[k]]
    a = np.array([d[k] for k in keys])
    b = np.array([base[k][0] for k in keys])
    x = a - b
    se = x.std(ddof=1) / np.sqrt(x.size)
    return (x.mean(), x.mean() - 1.96 * se, x.mean() + 1.96 * se,
            int((x > 0).sum()), x.size)


def main():
    print("per-sample paired vs cloudaware_mean (fixed 30k endpoint)")
    for mask in ("hard", "valid"):
        base = baselines()
        print("  --- mask=%s  (n baseline keys %d)" % (mask, len(base)))
        for run, name, seed in E2:
            mu, lo, hi, w, n = paired(run, mask, base)
            print("    %-24s s%d  mean %+.3f dB  CI [%+.3f, %+.3f]  win %d/%d"
                  " (%.0f%%)" % (run, seed, mu, lo, hi, w, n, 100.0 * w / n))
        # reference points: arm A / arm D vs the same baseline
        for seed in (2026, 2027, 2028):
            for arm in "AD":
                d = json.load(open(os.path.join(
                    RUNS, runof(seed, arm), "scene_psnr_test_%s.json" % mask)))
                keys = [k for k in d if k in base and d[k] == d[k]]
                a = np.array([d[k] for k in keys])
                b = np.array([base[k][0] for k in keys])
                x = a - b
                se = x.std(ddof=1) / np.sqrt(x.size)
                print("    arm %s                  s%d  mean %+.3f dB  "
                      "CI [%+.3f, %+.3f]  win %d/%d (%.0f%%)"
                      % (arm, seed, x.mean(), x.mean() - 1.96 * se,
                         x.mean() + 1.96 * se, int((x > 0).sum()), x.size,
                         100.0 * (x > 0).mean()))


if __name__ == "__main__":
    main()
