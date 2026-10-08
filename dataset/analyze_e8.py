# -*- coding: utf-8 -*-
"""E8 -- hyper-parameter sensitivity of the q gate (checklist item 6, sub-items 1 & 2).

Runs (all arm B, T=12, 30k iters, batch 8, cosine lr, last.pt endpoint):

    s2dsE8_gm2_sS   gamma0 = -2.0  (initial suppression sigmoid(-2) = 0.12, i.e. weak)
    s2dsE8_gp2_sS   gamma0 = +2.0  (initial suppression sigmoid(+2) = 0.88, i.e. strong)
    s2dsE8_w16_sS   gate width c = 16  (half of the main 32)
    s2dsE8_w48_sS   gate width c = 48  (1.5x the main 32)

Reference: the MAIN arm-B run at gamma0 = 0 (sigmoid 0.5) and c = 32, same seed.
Secondary context: the main arm-A run (no q) at the same seed.

Why this design.  The main runs initialise the learnable suppression scalar at
gamma = 0, and on the trained checkpoints it barely moves (sigmoid(gamma) =
0.5015 .. 0.5188, see dataset/analyze_gamma.py).  That raises a specific
objection: is the q result an artefact of that initial half-suppression point?
If yes, moving gamma0 to either extreme should move accuracy materially.
Likewise the gate network width is tied to the model width and has never been
varied.  Both scans are therefore read as EQUIVALENCE questions, not as
improvement questions: the finding we want to protect is "q helps", and it is
protected only if accuracy does not depend on these two knobs.

PRE-REGISTERED VERDICT RULE (fixed before the runs are read; not adjusted after):
per scanned setting, using the SEED-LEVEL mean of (scanned - main arm B):
    ROBUST      |mean| <= 0.10 dB, or seeds do not share a sign
    SENSITIVE   |mean| >  0.30 dB and all three seeds share that sign
    BORDERLINE  anything in between
The 0.10 / 0.30 dB thresholds are the same order as the effect the paper claims
for q (+0.15 .. +0.55 dB), so a knob that moves accuracy by more than 0.30 dB
would be a knob the claimed effect is contingent on.

Endpoint: scene_psnr_test_<mask>_fixed.json (fixed-iteration last.pt, 30k).
Seeds are never pooled; per-sample tests are paired on the sample id.
"""
import json
import os

import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/e8_result.json"
MASKS = ["hard", "valid"]
SEEDS = [2026, 2027, 2028]

SCANS = [
    ("gm2", "gamma0=-2 (weak initial suppression)"),
    ("gp2", "gamma0=+2 (strong initial suppression)"),
    ("w16", "gate width c=16 (0.5x)"),
    ("w48", "gate width c=48 (1.5x)"),
]

# The width scan changes the model width globally, so for w16/w48 the arm-A
# comparison is confounded (arm A exists only at c=32).  Flagged in the output.
CONFOUNDED_WITH_A = {"w16", "w48"}


def load(run, mask, fixed=True):
    fn = "scene_psnr_test_%s%s.json" % (mask, "_fixed" if fixed else "")
    p = os.path.join(RUNS, run, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def meanof(d):
    return float(np.nanmean(list(d.values()))) if d else float("nan")


def paired(ref, cur):
    """cur - ref, paired per sample id."""
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


def arm_of(seed, arm, mask):
    pre = "s2dsR" if (seed in (2026, 2027) and arm in "AB") else "s2dsV2"
    return load("%s_arm%s_s%d" % (pre, arm, seed), mask, fixed=False)


def verdict(v):
    mean = float(v.mean())
    same = bool(np.all(v > 0) or np.all(v < 0))
    if abs(mean) <= 0.10 or not same:
        return "ROBUST"
    if abs(mean) > 0.30 and same:
        return "SENSITIVE"
    return "BORDERLINE"


def main():
    res = {
        "protocol": "fixed-iteration (last.pt, 30k), paired per sample, seeds never pooled",
        "endpoint": "scene_psnr_test_<mask>_fixed.json",
        "rule": "ROBUST if |seed-level mean|<=0.10 dB or signs differ; "
                "SENSITIVE if |mean|>0.30 dB and all seeds share the sign; "
                "BORDERLINE otherwise",
        "scans": {k: v for k, v in SCANS},
    }

    for mask in MASKS:
        print("\n=== mask=%s (fixed 30k endpoint) ===" % mask)
        block = {}
        for key, desc in SCANS:
            per_seed_B, per_seed_A, rows = {}, {}, []
            for s in SEEDS:
                cur = load("s2dsE8_%s_s%d" % (key, s), mask)
                B = arm_of(s, "B", mask)
                A = arm_of(s, "A", mask)
                if cur is None or B is None:
                    print("   %-4s seed %d: MISSING (scan=%s ref=%s)"
                          % (key, s, cur is not None, B is not None))
                    continue
                d = paired(B, cur)
                r = dict(seed=s, psnr=meanof(cur), refB=meanof(B), minus_B=d)
                rows.append(r)
                per_seed_B[s] = d["delta"]
                if A is not None:
                    da = paired(A, cur)
                    r["minus_A"] = da
                    per_seed_A[s] = da["delta"]
                print("   %-4s seed %d  scan %6.3f | refB %6.3f | -B %+0.3f "
                      "(p=%.3g win=%2.0f%%) | -A %+0.3f"
                      % (key, s, r["psnr"], r["refB"], d["delta"], d["p"],
                         100 * d["win"],
                         (r["minus_A"]["delta"] if "minus_A" in r else float("nan"))))
            if not per_seed_B:
                continue
            v = np.array([per_seed_B[s] for s in sorted(per_seed_B)], float)
            t, p = stats.ttest_1samp(v, 0.0)
            sl = dict(per_seed=[round(float(x), 4) for x in v],
                      mean=float(v.mean()), sd=float(v.std(ddof=1)),
                      p=float(p), n=len(v),
                      all_same_sign=bool(np.all(v > 0) or np.all(v < 0)),
                      verdict=verdict(v))
            print("   %-4s SEED-LEVEL - mainB: %s  mean %+0.3f sd %0.3f p=%.3f "
                  "n=%d -> %s" % (key, ["%+0.3f" % x for x in v], sl["mean"],
                                  sl["sd"], sl["p"], sl["n"], sl["verdict"]))
            if per_seed_A:
                va = np.array([per_seed_A[s] for s in sorted(per_seed_A)], float)
                ta, pa = stats.ttest_1samp(va, 0.0)
                sl["minus_A"] = dict(per_seed=[round(float(x), 4) for x in va],
                                     mean=float(va.mean()), sd=float(va.std(ddof=1)),
                                     p=float(pa), n=len(va),
                                     all_same_sign=bool(np.all(va > 0) or np.all(va < 0)),
                                     confounded=key in CONFOUNDED_WITH_A)
                print("        (context) - mainA: %s mean %+0.3f p=%.3f%s"
                      % (["%+0.3f" % x for x in va], va.mean(), pa,
                         "   [CONFOUNDED: width changed model-wide]" if key in
                         CONFOUNDED_WITH_A else ""))
            block[key] = dict(desc=desc, rows=rows, seed_level=sl)
        res[mask] = block

    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
