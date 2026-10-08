"""EASY-stratum factorial analysis of the 12 learned arms (reviewer item 2).

The claim "temporal information still helps in clear scenes" previously rested on the
zero-training 2x2 only.  This script re-scores the same 12 checkpoints (fixed last.pt)
on the EASY stratum (target clear AND nearest frame clear) and computes the same
factorial contrasts that the manuscript reports on HARD/valid:

  dt effect     = ((C - A) + (D - B)) / 2      (simple effect averaged over q levels)
  q effect      = ((B - A) + (D - C)) / 2
  interaction   = (D - C) - (B - A) = (D - B) - (C - A)

Statistics: per-seed per-sample paired t (n=423 scenes) + seed-level n=3.
Output: dataset/analysis_e9easy.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/analysis_e9easy.json"
SEEDS = [2026, 2027, 2028]

RUN = {
    ("A", 2026): "s2dsR_armA_s2026", ("A", 2027): "s2dsR_armA_s2027",
    ("A", 2028): "s2dsV2_armA_s2028",
    ("B", 2026): "s2dsR_armB_s2026", ("B", 2027): "s2dsR_armB_s2027",
    ("B", 2028): "s2dsV2_armB_s2028",
    ("C", 2026): "s2dsV2_armC_s2026", ("C", 2027): "s2dsV2_armC_s2027",
    ("C", 2028): "s2dsV2_armC_s2028",
    ("D", 2026): "s2dsV2_armD_s2026", ("D", 2027): "s2dsV2_armD_s2027",
    ("D", 2028): "s2dsV2_armD_s2028",
}


def load(run):
    # a scene whose EASY mask is empty yields NaN; drop it from every comparison
    with open(os.path.join(RUNS, run, "scene_psnr_test_easy.json")) as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if v is not None and np.isfinite(v)}


def diff(a, b):
    """b - a over common scenes."""
    ks = sorted(set(a) & set(b))
    ks = [k for k in ks if np.isfinite(a[k]) and np.isfinite(b[k])]
    x = np.array([a[k] for k in ks])
    y = np.array([b[k] for k in ks])
    d = y - x
    n = len(d)
    t, p = stats.ttest_rel(y, x)
    se = float(d.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                ci95=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), win=float(np.mean(d > 0)),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)))


def interaction(A, B, C, D):
    """(D - C) - (B - A) per sample = does dt change the worth of q?"""
    ks = sorted(set(A) & set(B) & set(C) & set(D))
    ks = [k for k in ks if all(np.isfinite(d[k]) for d in (A, B, C, D))]
    v = (np.array([D[k] for k in ks]) - np.array([C[k] for k in ks])) - \
        (np.array([B[k] for k in ks]) - np.array([A[k] for k in ks]))
    n = len(v)
    t, p = stats.ttest_1samp(v, 0.0)
    se = float(v.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                ci95=[float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
                p=float(p), win=float(np.mean(v > 0)))


def contrast(a1, a2, b1, b2):
    """((a2-a1) + (b2-b1)) / 2 per sample."""
    ks = sorted(set(a1) & set(a2) & set(b1) & set(b2))
    ks = [k for k in ks if all(np.isfinite(d[k]) for d in (a1, a2, b1, b2))]
    v1 = np.array([a1[k] for k in ks])
    v2 = np.array([a2[k] for k in ks])
    w1 = np.array([b1[k] for k in ks])
    w2 = np.array([b2[k] for k in ks])
    d = 0.5 * ((v2 - v1) + (w2 - w1))
    n = len(d)
    t, p = stats.ttest_1samp(d, 0.0)
    se = float(d.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                ci95=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), win=float(np.mean(d > 0)))


res = {"stratum": "easy", "seeds": SEEDS, "per_seed": {}}
for s in SEEDS:
    A, B = load(RUN[("A", s)]), load(RUN[("B", s)])
    C, D = load(RUN[("C", s)]), load(RUN[("D", s)])
    means = {k: float(np.mean([x for x in v.values() if np.isfinite(x)]))
             for k, v in [("A", A), ("B", B), ("C", C), ("D", D)]}
    means["n_scenes"] = len(A)
    res["per_seed"]["s%d" % s] = dict(
        means=means,
        C_minus_A=diff(A, C), D_minus_B=diff(B, D),
        B_minus_A=diff(A, B), D_minus_C=diff(C, D),
        dt_effect=contrast(A, C, B, D),
        q_effect=contrast(A, B, C, D),
        interaction=interaction(A, B, C, D))

# seed level
for name in ["C_minus_A", "D_minus_B", "B_minus_A", "D_minus_C",
             "dt_effect", "q_effect", "interaction"]:
    ms = np.array([res["per_seed"]["s%d" % s][name]["mean"] for s in SEEDS])
    t, p = stats.ttest_1samp(ms, 0.0)
    res["seed_level_" + name] = dict(per_seed=[float(m) for m in ms],
                                     mean=float(ms.mean()), sd=float(ms.std(ddof=1)),
                                     p_seed=float(p),
                                     signs=[int(np.sign(m)) for m in ms])

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print("== EASY stratum, learned arms (fixed last.pt), n=%d scenes/seed =="
      % res["per_seed"]["s2026"]["C_minus_A"]["n"])
for s in SEEDS:
    m = res["per_seed"]["s%d" % s]["means"]
    print("  s%d  A=%.3f B=%.3f C=%.3f D=%.3f" % (s, m["A"], m["B"], m["C"], m["D"]))
for name in ["dt_effect", "q_effect", "C_minus_A", "D_minus_B",
             "B_minus_A", "D_minus_C", "interaction"]:
    a = res["seed_level_" + name]
    print("  %-12s per-seed %s | mean=%+.3f sd=%.3f p_seed=%.4f"
          % (name, " ".join("%+.3f" % m for m in a["per_seed"]),
             a["mean"], a["sd"], a["p_seed"]))
print("WROTE", OUT)
