"""Paired 2x2 analysis: dt main effect, q main effect, and the dt x q interaction."""
import os, sys, json, argparse, glob
import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_misr"


def load(prefix, split):
    out = {}
    for arm in "ABCD":
        p = os.path.join(RUNS, prefix % arm, "scene_psnr_%s.json" % split)
        if not os.path.exists(p):
            return None
        out[arm] = json.load(open(p))
    keys = set.intersection(*[set(v) for v in out.values()])
    keys = sorted(keys, key=lambda k: (len(k), k))
    return {a: np.array([out[a][k] for k in keys]) for a in out}, keys


def report(name, d, keys):
    d = np.asarray(d)
    n = len(d)
    if n < 3:
        print("  %-28s n=%d (too few)" % (name, n)); return
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t, p = stats.ttest_rel(d, np.zeros_like(d))
    dz = m / sd if sd > 0 else float("nan")
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print("  %-28s n=%2d  mean=%+.4f  sd=%.4f  95%%CI[%+.4f,%+.4f]  d_z=%+.2f  t=%+.2f  p=%.3g"
          % (name, n, m, sd, lo, hi, dz, t, p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mus2_arm%s")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    for split in args.splits.split(","):
        ld = load(args.prefix, split)
        if ld is None:
            print("[%s] missing per-scene json -> run eval_scene.py first" % split); continue
        d, keys = ld
        A, B, C, D = d["A"], d["B"], d["C"], d["D"]
        print("=== split=%s  n_scenes=%d ===" % (split, len(keys)))
        print("  arm means: A=%.4f B=%.4f C=%.4f D=%.4f" % (A.mean(), B.mean(), C.mean(), D.mean()))
        report("q effect @ no-dt  (B-A)", B - A, keys)
        report("q effect @ dt     (D-C)", D - C, keys)
        report("dt effect @ no-q  (C-A)", C - A, keys)
        report("dt effect @ q     (D-B)", D - B, keys)
        report("INTERACTION (D-C)-(B-A)", (D - C) - (B - A), keys)
        report("joint (D-A)", D - A, keys)
