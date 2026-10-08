"""E9 analysis: cloud-fed HighRes-net vs cloud-blind HighRes-net vs our arms.

Comparisons (per seed, per-sample paired on test, fixed last.pt protocol):
  1. highresnet_cld - highresnet      (does mask access alone rescue the baseline?)
  2. arm A - highresnet_cld           (do our arms still beat a mask-aware baseline?)
  3. arm D - highresnet_cld

Inputs:
  runs_s2ds/s2dsE9_highresnet_cld_s*/scene_psnr_test_{hard,valid}_fixed.json
  runs_s2ds/s2dsE2_highresnet_s*/scene_psnr_test_{hard,valid}_fixed.json
  main-arm fixed last.pt jsons (s2dsR_* for A/B s2026/2027, s2dsV2_* otherwise)

Output: dataset/analysis_e9.json + printed table.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/analysis_e9.json"
SEEDS = [2026, 2027, 2028]

MAIN = {
    ("A", 2026): "s2dsR_armA_s2026", ("A", 2027): "s2dsR_armA_s2027",
    ("A", 2028): "s2dsV2_armA_s2028",
    ("B", 2026): "s2dsR_armB_s2026", ("B", 2027): "s2dsR_armB_s2027",
    ("B", 2028): "s2dsV2_armB_s2028",
    ("C", 2026): "s2dsV2_armC_s2026", ("C", 2027): "s2dsV2_armC_s2027",
    ("C", 2028): "s2dsV2_armC_s2028",
    ("D", 2026): "s2dsV2_armD_s2026", ("D", 2027): "s2dsV2_armD_s2027",
    ("D", 2028): "s2dsV2_armD_s2028",
}


def _load_base(endpoint):
    """Per-scene PSNR of the deterministic zero-training gatekeeper (cloudaware_mean)."""
    p = "/mnt/e/论文2/dataset/zero_baseline_scene.json"
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))[endpoint]
    return {k: v for k, v in d.items() if v is not None and np.isfinite(float(v))}


def load(run, endpoint):
    p = os.path.join(RUNS, run, "scene_psnr_test_%s_fixed.json" % endpoint)
    if not os.path.exists(p):           # main arms keep the fixed protocol untagged
        p = os.path.join(RUNS, run, "scene_psnr_test_%s.json" % endpoint)
    with open(p) as f:
        d = json.load(f)
    # scenes with no pixel of this endpoint carry NaN/None and must be dropped,
    # otherwise np.mean over the dict values returns NaN for the whole run.
    return {k: v for k, v in d.items()
            if v is not None and np.isfinite(float(v))}


def paired(a, b):
    """b - a over common scenes, both finite."""
    ks = [k for k in sorted(set(a) & set(b))
          if np.isfinite(float(a[k])) and np.isfinite(float(b[k]))]
    x = np.array([a[k] for k in ks], dtype=np.float64)
    y = np.array([b[k] for k in ks], dtype=np.float64)
    d = y - x
    n = len(d)
    t, p = stats.ttest_rel(y, x)
    w = float(np.mean(d > 0))
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12))
    se = float(d.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                ci95=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), win=w, dz=dz)


res = {"seeds": SEEDS, "endpoints": {}}
for ep in ["hard", "valid"]:
    out = {"means": {}, "effects": {}}
    BASE = _load_base(ep)
    for s in SEEDS:
        hr_blind = load("s2dsE2_highresnet_s%d" % s, ep)
        hr_cld = load("s2dsE9_highresnet_cld_s%d" % s, ep)
        armA = load(MAIN[("A", s)], ep)
        armD = load(MAIN[("D", s)], ep)
        out["means"]["s%d" % s] = dict(
            baseline=float(np.mean(list(BASE.values()))) if BASE else float("nan"),
            highresnet=float(np.mean(list(hr_blind.values()))),
            highresnet_cld=float(np.mean(list(hr_cld.values()))),
            armA=float(np.mean(list(armA.values()))),
            armD=float(np.mean(list(armD.values()))))
        out["effects"]["s%d" % s] = dict(
            cld_minus_blind=paired(hr_blind, hr_cld),
            blind_minus_base=paired(BASE, hr_blind),
            cld_minus_base=paired(BASE, hr_cld),
            armA_minus_cld=paired(hr_cld, armA),
            armD_minus_cld=paired(hr_cld, armD))
    # seed-level aggregation (n=3)
    agg = {}
    for eff in ["cld_minus_blind", "blind_minus_base", "cld_minus_base",
                "armA_minus_cld", "armD_minus_cld"]:
        ms = np.array([out["effects"]["s%d" % s][eff]["mean"] for s in SEEDS])
        t, p = stats.ttest_1samp(ms, 0.0)
        agg[eff] = dict(per_seed=[float(m) for m in ms], mean=float(ms.mean()),
                        sd=float(ms.std(ddof=1)), p_seed=float(p),
                        signs=[int(np.sign(m)) for m in ms])
    out["seed_level"] = agg
    res["endpoints"][ep] = out

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

for ep in ["hard", "valid"]:
    o = res["endpoints"][ep]
    print("== endpoint %s ==" % ep)
    for s in SEEDS:
        m = o["means"]["s%d" % s]
        print("  s%d  blind=%.3f cld=%.3f armA=%.3f armD=%.3f"
              % (s, m["highresnet"], m["highresnet_cld"], m["armA"], m["armD"]))
    for eff, a in o["seed_level"].items():
        print("  %-16s per-seed %s  mean=%.3f sd=%.3f p_seed=%.4f"
              % (eff, ["%+.3f" % m for m in a["per_seed"]], a["mean"], a["sd"], a["p_seed"]))
print("WROTE", OUT)
