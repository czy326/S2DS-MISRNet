"""2x2 interaction analysis on the self-built S2 dataset (pre-registered rules).

Primary endpoint: I = (D - C) - (B - A), paired.
Decision rule (written down in dataset/EXPERIMENT_PROTOCOL.md BEFORE running):
  pass     : I >= +0.10 dB and paired 95% CI lower bound > 0
  partial  : 0 < I < +0.10 and CI lower bound > 0
  fail     : CI contains 0, or I <= 0
Reported twice: sample-level (n = all samples) and spatial-unit level (average the target
dates inside each (AOI, patch) unit, n = 45) -- samples inside a unit share input frames,
so the unit level is the conservative one.  Both must agree in direction.
"""
import os
import re
import sys
import json
import argparse

import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_s2ds"


def unit_of(scene):
    m = re.match(r"^(.*)_(\d{8})_(\d+)_(\d+)$", scene)
    return "%s_%s_%s" % (m.group(1), m.group(3), m.group(4)) if m else scene


def load(prefix, split, seed, mask="valid"):
    out = {}
    for arm in "ABCD":
        fname = ("scene_psnr_%s.json" % split) if mask == "valid" \
            else ("scene_psnr_%s_%s.json" % (split, mask))
        p = os.path.join(RUNS, prefix % (arm, seed), fname)
        if not os.path.exists(p):
            return None
        out[arm] = json.load(open(p))
    keys = sorted(set.intersection(*[set(v) for v in out.values()]))
    return {a: np.array([out[a][k] for k in keys]) for a in out}, keys


def agg_units(vals, keys):
    d = {}
    for v, k in zip(vals, keys):
        d.setdefault(unit_of(k), []).append(v)
    return np.array([np.nanmean(d[k]) for k in sorted(d)])


def report(tag, d):
    d = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(d)
    d = d[ok]
    n = len(d)
    if n < 5:
        return dict(n=n, mean=float("nan"))
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t, p = stats.ttest_rel(d, np.zeros_like(d))
    try:
        w, pw = stats.wilcoxon(d)
    except Exception:
        pw = float("nan")
    r = dict(n=n, mean=m, sd=sd, se=se, ci=[m - 1.96 * se, m + 1.96 * se],
             dz=m / sd if sd > 0 else float("nan"), t=float(t), p=float(p),
             wilcoxon_p=float(pw), frac_pos=float((d > 0).mean()))
    print("  %-30s n=%4d  mean=%+.4f  95%%CI[%+.4f,%+.4f]  d_z=%+.2f  t=%+.2f  p=%.3g  pos=%.0f%%"
          % (tag, n, m, r["ci"][0], r["ci"][1], r["dz"], t, p, 100 * r["frac_pos"]))
    return r


def verdict(I):
    if not np.isfinite(I.get("mean", float("nan"))):
        return "no-data"
    if I["mean"] >= 0.10 and I["ci"][0] > 0:
        return "PASS"
    if I["mean"] > 0 and I["ci"][0] > 0:
        return "PARTIAL (direction positive, magnitude < +0.10 dB)"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split", default="test")
    ap.add_argument("--prefix", default="s2ds_arm%s_s%d")
    ap.add_argument("--mask", default="valid", choices=["valid", "hard", "easy"])
    args = ap.parse_args()
    ld = load(args.prefix, args.split, args.seed, args.mask)
    if ld is None:
        print("missing per-sample json for some arm (run eval_s2ds.py first)")
        return
    d, keys = ld
    A, B, C, D = d["A"], d["B"], d["C"], d["D"]
    res = {"seed": args.seed, "split": args.split, "mask": args.mask, "n_samples": len(keys),
           "arm_means": {a: float(np.nanmean(d[a])) for a in "ABCD"},
           "sample_level": {}, "unit_level": {}}
    print("=== seed=%d split=%s mask=%s  n_samples=%d  n_units=%d ==="
          % (args.seed, args.split, args.mask, len(keys), len(set(unit_of(k) for k in keys))))
    print("  arm means: " + "  ".join("%s=%.4f" % (a, np.nanmean(d[a])) for a in "ABCD"))

    eff = {
        "F2_q@no_dt (B-A)": B - A,
        "F2_q@dt    (D-C)": D - C,
        "F1_dt@no_q (C-A)": C - A,
        "F1_dt@q    (D-B)": D - B,
        "INTERACTION (D-C)-(B-A)": (D - C) - (B - A),
        "JOINT       (D-A)": D - A,
        "F1 main effect": (C + D) / 2 - (A + B) / 2,
        "F2 main effect": (B + D) / 2 - (A + C) / 2,
    }
    print("-- sample level (paired, n=%d) --" % len(keys))
    for k, v in eff.items():
        res["sample_level"][k] = report(k, v)
    print("-- spatial-unit level (avg over target dates in each AOI x patch, n=45) --")
    for k, v in eff.items():
        res["unit_level"][k] = report(k, agg_units(v, keys))

    key = "INTERACTION (D-C)-(B-A)"
    res["verdict_sample"] = verdict(res["sample_level"][key])
    res["verdict_unit"] = verdict(res["unit_level"][key])
    res["verdict"] = res["verdict_sample"] if res["verdict_sample"] == res["verdict_unit"] \
        else "MIXED (sample vs unit level disagree -> treat as not established)"
    print("\nINTERACTION verdict (pre-registered rule: >= +0.10 dB and CI lower > 0):")
    print("  sample level : %s" % res["verdict_sample"])
    print("  unit level   : %s" % res["verdict_unit"])
    print("  -> FINAL     : %s" % res["verdict"])
    out = os.path.join(RUNS, "analysis_s%d_%s_%s.json" % (args.seed, args.split, args.mask))
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
