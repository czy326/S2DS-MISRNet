# -*- coding: utf-8 -*-
"""One-screen status of every supplement batch, and a compact headline per finished one.

Run this instead of tailing five different logs: it reports which artefacts exist, how far
the training lanes are, and -- for the batches that are done -- the numbers that will be
pasted into the manuscript.
"""
import os
import glob
import json

R = r"/mnt/e/论文2/runs_s2ds"
D = r"/mnt/e/论文2/dataset"
SEEDS = [2026, 2027, 2028]


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def have(run, mask, tag=""):
    fn = "scene_psnr_test_%s%s.json" % (mask, ("_%s" % tag) if tag else "")
    return os.path.exists(os.path.join(R, run, fn))


def marker(log, text):
    p = os.path.join(D, log)
    if not os.path.exists(p):
        return False
    return text in open(p, encoding="utf-8", errors="ignore").read()


def sec(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    sec("training lanes")
    for tag, pat in [("E3  frame-shuffle  s2dsS_*", "s2dsS_*"),
                     ("E5  permuted-cloud s2dsP_armB_*", "s2dsP_armB_*"),
                     ("E1  q-route        s2dsE1_*", "s2dsE1_*"),
                     ("E2  baselines      s2dsE2_*", "s2dsE2_*")]:
        dirs = sorted(glob.glob(os.path.join(R, pat)))
        print("%-32s %d run dirs" % (tag, len(dirs)))
    for log, mark, name in [("pipeline_shuffle.log", "E3 DONE", "E3"),
                            ("pipeline_permcld_train.log", "E5-train DONE", "E5-train"),
                            ("pipeline_e1.log", "E1 DONE", "E1"),
                            ("pipeline_e2.log", "E2 DONE", "E2")]:
        print("   %-9s marker %-14s %s" % (name, mark,
                                           "DONE" if marker(log, mark) else "running"))

    sec("E4  tau/p sweep (CPU) -- done")
    d = jload(os.path.join(D, "bench_tau_sweep.json"))
    if d:
        print("   key: |ct-c| max over the grid = see bench_tau_sweep.json")

    sec("E5  evaluation-time permuted cloud (seed 2028 only)")
    d = jload(os.path.join(D, "permcld_evaltime_result.json"))
    if d:
        for m in ["hard", "valid"]:
            if m in d:
                gi = d[m]["rows"][0]["gain_intact"]
                gs = d[m]["rows"][0]["gain_scrambled"]
                print("   %-6s intact %+0.3f | scrambled %+0.3f | retained %0.0f%%"
                      % (m, gi["delta"], gs["delta"],
                         100 * d[m]["rows"][0]["retained_frac"]))

    sec("E5  train-time permuted cloud")
    d = jload(os.path.join(D, "permcld_train_result.json"))
    if d:
        for m in ["hard", "valid"]:
            if m in d:
                print("   %-6s mean retained %0.2f  per-seed %s  prediction<=0.5: %s"
                      % (m, d[m]["mean_retained"], d[m]["per_seed_retained"],
                         d[m]["prediction_holds"]))
    else:
        print("   not finished yet (%d/3 runs have per-scene output)"
              % sum(have("s2dsP_armB_s%d" % s, "hard", "fixed") for s in SEEDS))

    sec("E6  full statistics")
    d = jload(os.path.join(D, "appendix_stats.json"))
    if isinstance(d, dict):
        print("   %s" % d.get("summary", "see appendix_stats.json"))
    elif isinstance(d, list):
        sig = sum(1 for r in d if isinstance(r, dict) and r.get("q_fdr", 1) < 0.05)
        print("   %d effect cells, %d significant at BH-FDR q<0.05" % (len(d), sig))
    elif d is not None:
        print("   %d entries" % len(d))

    sec("E1  q-route ablation")
    d = jload(os.path.join(D, "e1_result.json"))
    if d:
        for m in ["hard", "valid"]:
            if m in d:
                s = d[m]["summary"]
                print("   %-6s B-A %+0.3f | B1(px)-A %+0.3f | B2(fr)-A %+0.3f"
                      % (m, s["B_minus_A"]["mean"], s["B1_minus_A"]["mean"],
                         s["B2_minus_A"]["mean"]))
    else:
        print("   not finished yet (%d/6 runs have per-scene output)"
              % sum(have("s2dsE1_arm%s_s%d" % (a, s), "hard", "fixed")
                    for a in ["B1", "B2"] for s in SEEDS))

    sec("E2  published baselines")
    d = jload(os.path.join(D, "e2_result.json"))
    if d:
        for m in ["hard", "valid"]:
            for r in d.get(m, []):
                a = r.get("minus_A")
                if a:
                    print("   %-6s %-26s PSNR %6.3f  -A %+0.3f (p=%.2g)"
                          % (m, r["run"], r["psnr"], a["delta"], a["p"]))
    else:
        n = sum(have("s2dsE2_%s_s%d" % (b, s), "hard", "fixed")
                for b in ["highresnet", "rams"] for s in SEEDS)
        print("   not finished yet (%d/4 runs have per-scene output)" % n)

    sec("E3  frame-shuffle verdict")
    d = jload(os.path.join(D, "shuffle_result.json"))
    if d:
        print("   %s" % json.dumps(d.get("verdict", d), ensure_ascii=False)[:400])
    else:
        print("   not finished yet (%d/12 runs have fixed-endpoint output)"
              % sum(have("s2dsS_arm%s_s%d" % (a, s), "hard", "fixed")
                    for a in "ABCD" for s in SEEDS))


if __name__ == "__main__":
    main()
