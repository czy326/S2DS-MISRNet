"""Paired A vs B analysis for a single factor (e.g. q gating on PROBA-V)."""
import os, json, argparse
import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_misr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    for split in args.splits.split(","):
        pa = os.path.join(RUNS, args.a, "scene_psnr_%s.json" % split)
        pb = os.path.join(RUNS, args.b, "scene_psnr_%s.json" % split)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("[%s] missing json: %s / %s" % (split, os.path.exists(pa), os.path.exists(pb)))
            continue
        A, B = json.load(open(pa)), json.load(open(pb))
        keys = sorted(set(A) & set(B))
        a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
        d = b - a
        n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
        t, p = stats.ttest_rel(b, a)
        w, pw = stats.wilcoxon(b, a) if n >= 10 else (float("nan"), float("nan"))
        dz = m / sd if sd > 0 else float("nan")
        print("=== %s vs %s | split=%s n=%d ===" % (args.b, args.a, split, n))
        print("  A mean=%.4f  B mean=%.4f  diff(B-A)=%+.4f  sd=%.4f  SE=%.4f" % (a.mean(), b.mean(), m, sd, se))
        print("  95%%CI=[%+.4f, %+.4f]  d_z=%+.3f  paired t=%+.2f p=%.3g  wilcoxon p=%.3g" % (m - 1.96 * se, m + 1.96 * se, dz, t, p, pw))
        print("  min/max diff: %+.3f / %+.3f ; #scenes B>A: %d/%d" % (d.min(), d.max(), int((d > 0).sum()), n))


if __name__ == "__main__":
    main()
