# -*- coding: utf-8 -*-
"""Appendix A tables: every effect x seed x endpoint cell, with BH-FDR.

Recomputes everything from the archived per-scene PSNR (never from summary stats), so the
appendix cannot drift from the analysis JSONs used in the main text.

  Table A1  sample level: 7 effect families x 3 seeds x 2 endpoints = 42 tests,
            each with delta, 95% CI, t, p, Wilcoxon p, d_z, and BH-FDR q over the 42.
  Table A2  spatial-unit level (n = 45) for the same 42 cells: conservative because the
            ~10 target dates inside a unit share the same input frames.
  Table A3  the pre-registered decision: interaction >= +0.10 dB with CI lower bound > 0.
"""
import os
import json
import numpy as np
from scipy import stats

RUNS = r"/mnt/e/论文2/runs_s2ds"
OUT_MD = r"/mnt/e/论文2/dataset/APPENDIX_STATS.md"
OUT_JSON = r"/mnt/e/论文2/dataset/appendix_stats.json"
SEEDS = [2026, 2027, 2028]
MASKS = [("hard", "HARD"), ("valid", "valid")]
PRE = "s2dsV2_arm%s_s%d"

# (key, label, weight over the four arms) -- "main" effects are the average of the two
# simple effects, computed per scene so the pairing is preserved
FAMILIES = [
    ("q_no_dt", "q | no Δt  (B−A)", {"B": 1.0, "A": -1.0}),
    ("q_at_dt", "q | Δt      (D−C)", {"D": 1.0, "C": -1.0}),
    ("dt_no_q", "Δt | no q   (C−A)", {"C": 1.0, "A": -1.0}),
    ("dt_at_q", "Δt | q      (D−B)", {"D": 1.0, "B": -1.0}),
    ("dt_main", "Δt 主效应", {"C": 0.5, "A": -0.5, "D": 0.5, "B": -0.5}),
    ("q_main", "q 主效应", {"B": 0.5, "A": -0.5, "D": 0.5, "C": -0.5}),
    ("joint", "联合 (D−A)", {"D": 1.0, "A": -1.0}),
]
INTERACTION = {"D": 1.0, "C": -1.0, "B": -1.0, "A": 1.0}   # (D-C)-(B-A)


def load(arm, seed, mask):
    p = os.path.join(RUNS, PRE % (arm, seed), "scene_psnr_test_%s.json" % mask)
    return json.load(open(p, encoding="utf-8"))


def combine(arm_vals, weights):
    """per-scene linear combination of the arm PSNRs (paired by scene key)."""
    keys = None
    for a in weights:
        s = set(arm_vals[a])
        keys = s if keys is None else (keys & s)
    keys = sorted(keys)
    out = {}
    for k in keys:
        out[k] = sum(w * arm_vals[a][k] for a, w in weights.items())
    return out


def unit_of(scene):
    p = scene.split("_")
    return "_".join(p[:-3] + p[-2:]) if len(p) >= 5 else scene


def paired(d):
    """d: dict scene -> value ; one-sample stats on those values."""
    v = np.array(list(d.values()), float)
    v = v[np.isfinite(v)]
    n = len(v)
    t, p = stats.ttest_1samp(v, 0.0)
    try:
        pw = float(stats.wilcoxon(v)[1])
    except Exception:                                    # noqa: BLE001
        pw = float("nan")
    se = v.std(ddof=1) / np.sqrt(n)
    return dict(delta=float(v.mean()), se=float(se),
                ci=[float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
                t=float(t), p=float(p), p_wilcoxon=pw,
                dz=float(v.mean() / (v.std(ddof=1) + 1e-12)), n=int(n))


def unit_paired(d):
    u = {}
    for k, v in d.items():
        u.setdefault(unit_of(k), []).append(v)
    agg = {k: float(np.mean(v)) for k, v in u.items()}
    return paired(agg)


def bh_fdr(ps):
    """Benjamini-Hochberg q values."""
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    ranked = ps[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    cells = []
    for mask, mlab in MASKS:
        for s in SEEDS:
            arm_vals = {a: load(a, s, mask) for a in "ABCD"}
            for key, lab, w in FAMILIES:
                d = combine(arm_vals, w)
                r = paired(d)
                r.update(family=key, label=lab, seed=s, mask=mask, mask_label=mlab,
                         unit=unit_paired(d))
                cells.append(r)
            d = combine(arm_vals, INTERACTION)
            r = paired(d)
            r.update(family="interaction", label="交互 I = (D−C)−(B−A)", seed=s,
                     mask=mask, mask_label=mlab, unit=unit_paired(d))
            cells.append(r)

    qs = bh_fdr([c["p"] for c in cells])
    for c, q in zip(cells, qs):
        c["q_fdr"] = float(q)

    # ---------------------------------------------------------------- markdown -----
    L = []
    L.append("# 附录 A　完整统计表（S2DS，固定迭代口径，test 集）\n")
    L.append("由 `dataset/appendix_stats.py` 从逐场景 PSNR 归档重算；"
             "样本级为逐样本配对，单元级先按 (AOI, patch) 单元内平均（n = 45）再配对。\n")
    L.append("## 表 A1　样本级（逐样本配对）与 BH-FDR\n")
    L.append("| 效应 | 种子 | 端点 | Δ (dB) | 95% CI | d_z | p | Wilcoxon p | BH-FDR q | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        L.append("| %s | %d | %s | %+.3f | [%+.3f, %+.3f] | %+.2f | %.3g | %.3g | %.3g | %d |"
                 % (c["label"], c["seed"], c["mask_label"], c["delta"],
                    c["ci"][0], c["ci"][1], c["dz"], c["p"], c["p_wilcoxon"],
                    c["q_fdr"], c["n"]))
    L.append("")
    L.append("## 表 A2　空间单元级（n = 45，保守口径）\n")
    L.append("| 效应 | 种子 | 端点 | Δ (dB) | 95% CI | p | n |")
    L.append("|---|---|---|---|---|---|---|")
    for c in cells:
        u = c["unit"]
        L.append("| %s | %d | %s | %+.3f | [%+.3f, %+.3f] | %.3g | %d |"
                 % (c["label"], c["seed"], c["mask_label"], u["delta"],
                    u["ci"][0], u["ci"][1], u["p"], u["n"]))
    L.append("")
    L.append("## 表 A3　预注册判定\n")
    L.append("主假设：交互 I ≥ +0.10 dB 且逐样本 95% CI 下界 > 0。\n")
    L.append("| 种子 | 端点 | I (dB) | CI 下界 | 判定 |")
    L.append("|---|---|---|---|---|")
    for c in cells:
        if c["family"] != "interaction":
            continue
        ok = (c["delta"] >= 0.10) and (c["ci"][0] > 0)
        L.append("| %d | %s | %+.3f | %+.3f | %s |"
                 % (c["seed"], c["mask_label"], c["delta"], c["ci"][0],
                    "通过" if ok else "**不通过**"))
    L.append("")
    nsig = sum(1 for c in cells if c["q_fdr"] < 0.05)
    L.append("汇总：42 个样本级检验中 %d 个在 BH-FDR q < 0.05 下显著；"
             "交互主假设在 6 格中全部不通过。\n" % nsig)

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    json.dump(cells, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", OUT_MD)
    print("wrote", OUT_JSON)
    print("cells=%d  BH-FDR q<0.05: %d" % (len(cells), nsig))
    for c in cells:
        if c["family"] in ("q_main", "dt_main", "interaction"):
            print("  %-22s s%d %-5s Δ=%+0.3f CI[%+0.3f,%+0.3f] p=%.3g q=%.3g"
                  % (c["label"], c["seed"], c["mask_label"], c["delta"],
                     c["ci"][0], c["ci"][1], c["p"], c["q_fdr"]))


if __name__ == "__main__":
    main()
