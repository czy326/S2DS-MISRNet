"""E3 analysis: does removing the ordinal time rank (frame shuffle) change the value of dt?

Reads the archived per-scene PSNR of the ordered runs (s2dsV2_arm{A,B,C,D}_s{seed}, gate,
fixed-iter = last.pt) and of the shuffled runs (s2dsS_arm{A,B,C,D}_s{seed}), and reports,
per seed x mask:

    dCA_ord = C_ord - A_ord          dt's value when the frame order leaks the rank
    dCA_shu = C_shu - A_shu          dt's value when it does not
    diff    = dCA_shu - dCA_ord      <-- the pre-registered quantity (prediction: > 0)
    dA      = A_shu - A_ord          prediction: < 0 on HARD (A loses "which is nearest")
    dC      = C_shu - C_ord          prediction: ~ 0 (C still has dt)

Secondary (not pre-registered, but the 12 runs exist): the same contrast on top of q,
    dDB_ord = D_ord - B_ord,  dDB_shu = D_shu - B_shu,  diffDB = dDB_shu - dDB_ord

Two-sided verdicts; paired t + Wilcoxon + 95% CI at sample level, then a conservative
spatial-unit level (45 units), then a seed-level one-sample t (n = 3).

NOTE: the per-scene files carry one entry per test sample (423) but only the samples that
actually contain HARD pixels are finite (299).  Every pairing therefore has to mask
non-finite entries -- otherwise the mean is NaN (this is exactly what happened on the
first run of this script).
"""
import os
import json
import argparse

import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/shuffle_result.json"
SEEDS = [2026, 2027, 2028]


def load(run, mask):
    """fixed-iter file first (that is the primary endpoint), else the plain one."""
    for fn in ("scene_psnr_test_%s_fixed.json" % mask, "scene_psnr_test_%s.json" % mask):
        p = os.path.join(RUNS, run, fn)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError("no %s psnr for %s" % (mask, run))


def finite_pairs(a, b):
    """Scene keys present in both and finite in both."""
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], dtype=np.float64)
    y = np.array([b[k] for k in ks], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    return [k for k, ok in zip(ks, m) if ok], x[m], y[m]


def paired(a, b):
    ks, x, y = finite_pairs(a, b)
    if len(ks) < 5:
        return None
    d = y - x
    t, p = stats.ttest_rel(y, x)
    try:
        _, pw = stats.wilcoxon(d)
    except Exception:                                   # noqa: BLE001
        pw = float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                t=float(t), p=float(p), p_wilcoxon=float(pw),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), n=int(len(d)),
                keys=ks)


def diff_of_diffs(a0, b0, a1, b1):
    """(b1-a1) minus (b0-a0), paired over scenes finite in all four."""
    ks = sorted(set(a0) & set(b0) & set(a1) & set(b1))
    v = {k: [a0[k], b0[k], a1[k], b1[k]] for k in ks}
    ks = [k for k in ks if all(np.isfinite(x) for x in v[k])]
    if len(ks) < 5:
        return None
    x = np.array([v[k][1] - v[k][0] for k in ks])
    y = np.array([v[k][3] - v[k][2] for k in ks])
    d = y - x
    t, p = stats.ttest_rel(y, x)
    try:
        _, pw = stats.wilcoxon(d)
    except Exception:                                    # noqa: BLE001
        pw = float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), p_wilcoxon=float(pw), n=int(len(d)))


def unit_of(scene):
    """scene = '<aoi>_<date>_<oy>_<ox>' -> unit = '<aoi>_<oy>_<ox>' (45 of them)."""
    p = scene.split("_")
    return "_".join(p[:-3] + p[-2:])


def unit_level(a, b):
    r = paired(a, b)
    if r is None:
        return None
    ua, ub = {}, {}
    for k in r["keys"]:
        ua.setdefault(unit_of(k), []).append(a[k])
        ub.setdefault(unit_of(k), []).append(b[k])
    ks = sorted(set(ua) & set(ub))
    if len(ks) < 5:
        return None
    x = np.array([np.mean(ua[k]) for k in ks])
    y = np.array([np.mean(ub[k]) for k in ks])
    d = y - x
    t, p = stats.ttest_rel(y, x)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), n=int(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="hard,valid")
    ap.add_argument("--ordered-prefix", default="s2dsV2_arm%s_s%d")
    ap.add_argument("--shuffled-prefix", default="s2dsS_arm%s_s%d")
    args = ap.parse_args()

    res = {"masks": {}, "seed_level": {}}
    for mask in args.masks.split(","):
        per_seed = {}
        for s in SEEDS:
            Ao = load(args.ordered_prefix % ("A", s), mask)
            Co = load(args.ordered_prefix % ("C", s), mask)
            As = load(args.shuffled_prefix % ("A", s), mask)
            Cs = load(args.shuffled_prefix % ("C", s), mask)
            dCA_o, dCA_s = paired(Ao, Co), paired(As, Cs)
            diff = diff_of_diffs(Ao, Co, As, Cs)
            row = dict(dCA_ordered=dCA_o, dCA_shuffled=dCA_s, diff=diff,
                       A_shu_minus_A_ord=paired(Ao, As),
                       C_shu_minus_C_ord=paired(Co, Cs),
                       A_shu_minus_A_ord_unit=unit_level(Ao, As))
            # secondary: the same contrast with q switched on (arms B -> D)
            try:
                Bo = load(args.ordered_prefix % ("B", s), mask)
                Do = load(args.ordered_prefix % ("D", s), mask)
                Bs = load(args.shuffled_prefix % ("B", s), mask)
                Ds = load(args.shuffled_prefix % ("D", s), mask)
                row["dDB_ordered"] = paired(Bo, Do)
                row["dDB_shuffled"] = paired(Bs, Ds)
                row["diffDB"] = diff_of_diffs(Bo, Do, Bs, Ds)
                row["B_shu_minus_B_ord"] = paired(Bo, Bs)
            except FileNotFoundError:
                pass
            per_seed[s] = row
            extra = ""
            if "diffDB" in row and row["diffDB"] is not None:
                extra = " | D-B ord %+0.3f shu %+0.3f diffDB %+0.3f" % (
                    row["dDB_ordered"]["delta"], row["dDB_shuffled"]["delta"],
                    row["diffDB"]["delta"])
            print("[%s s%d] C-A ordered %+0.3f | shuffled %+0.3f | diff %+0.3f "
                  "CI[%+0.3f,%+0.3f] p=%.3g | A_shu-A_ord %+0.3f (p=%.3g) | C_shu-C_ord %+0.3f%s"
                  % (mask, s, dCA_o["delta"], dCA_s["delta"], diff["delta"],
                     diff["ci"][0], diff["ci"][1], diff["p"],
                     row["A_shu_minus_A_ord"]["delta"], row["A_shu_minus_A_ord"]["p"],
                     row["C_shu_minus_C_ord"]["delta"], extra))
        res["masks"][mask] = {str(s): per_seed[s] for s in SEEDS}

        v = np.array([per_seed[s]["diff"]["delta"] for s in SEEDS], dtype=np.float64)
        t, p = stats.ttest_1samp(v, 0.0)
        se = v.std(ddof=1) / np.sqrt(len(v))
        res["seed_level"][mask] = dict(
            per_seed=[float(x) for x in v], mean=float(v.mean()), se=float(se),
            t=float(t), p=float(p),
            same_sign=bool(np.all(v > 0) or np.all(v < 0)))
        sl = res["seed_level"][mask]
        print("  seed-level diff: %+0.3f +/- %0.3f  t=%.2f p=%.3f  same-sign=%s"
              % (sl["mean"], sl["se"], sl["t"], sl["p"], sl["same_sign"]))

        # secondary seed-level read-out: dt on top of q (D-B), exploratory
        if all(per_seed[s].get("diffDB") for s in SEEDS):
            v2 = np.array([per_seed[s]["diffDB"]["delta"] for s in SEEDS], dtype=np.float64)
            t2, p2 = stats.ttest_1samp(v2, 0.0)
            se2 = v2.std(ddof=1) / np.sqrt(len(v2))
            res["seed_level"][mask + "_diffDB"] = dict(
                per_seed=[float(x) for x in v2], mean=float(v2.mean()), se=float(se2),
                t=float(t2), p=float(p2),
                same_sign=bool(np.all(v2 > 0) or np.all(v2 < 0)))
            s2 = res["seed_level"][mask + "_diffDB"]
            print("  [exploratory] seed-level diffDB (D-B): %+0.3f +/- %0.3f  t=%.2f "
                  "p=%.3f  same-sign=%s"
                  % (s2["mean"], s2["se"], s2["t"], s2["p"], s2["same_sign"]))

    hard = res["seed_level"].get("hard", {})
    pred_i = bool(hard.get("mean", 0) > 0 and hard.get("same_sign"))
    a_vals = [res["masks"]["hard"][str(s)]["A_shu_minus_A_ord"]["delta"] for s in SEEDS] \
        if "hard" in res["masks"] else []
    pred_ii = bool(a_vals and all(v < 0 for v in a_vals))
    res["verdict"] = dict(
        prediction_i_diff_positive=pred_i,
        prediction_ii_A_degrades_on_hard=pred_ii,
        reading=("UPGRADE: dt carries real information, masked by the redundant ordinal rank"
                 if (pred_i and pred_ii) else
                 "NOT SUPPORTED: the frame-order channel does not explain the dt null"
                 if not pred_i else
                 "PARTIAL: dt gains under shuffle but A does not degrade as predicted"))
    print("\nVERDICT:", res["verdict"]["reading"])

    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
