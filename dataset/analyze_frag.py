# -*- coding: utf-8 -*-
"""大修清单 第 4 项 —— 云破碎度分层下重新计量 q 的三条通路。

背景：6.6 节的路线分解显示 q 的收益约 87% 由帧级通路（B2）实现、逐像素通路（B1）
只贡献约 0.05 dB，并给出可检验推论"云越破碎，逐像素通路份额越高"。该推论此前只是
假说。本脚本在**已有 3 种子检查点**的逐样本 PSNR 上按云破碎度分层重算，不需要重训。

破碎度指标（逐样本，在 LR 网格 48×48 上计算）：
  cf_t        第 t 帧的云占比（cld > 0.5）
  partly      0.02 < cf_t < 0.98 的帧：只有这类帧上"逐像素选帧"才可能有意义
  ncomp       部分云帧的 8-连通云斑块数量的中位数
  patch       部分云帧的云斑块平均面积（像素）的中位数
  edge        部分云帧的云边界像素占比（4-邻域内存在异类）的中位数

分组：先按"是否存在部分云帧"分成整帧云主导 / 混合两类，再在混合类内按 patch 中位数
分成"云大块"与"云细碎"两半。逐种子逐样本配对，种子不合并。
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy import stats

sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
from s2ds_dataset import S2DS  # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/frag_result.json"
SEEDS = [2026, 2027, 2028]
MASK = "hard"

# A/B 取自主 2×2（s2dsV2_*），B1/B2 取自 E1（s2dsE1_*，用 _fixed 后缀）
PREFIX = {"A": "s2dsV2_armA_s", "B": "s2dsV2_armB_s",
          "B1": "s2dsE1_armB1_s", "B2": "s2dsE1_armB2_s"}


def frag_metrics(cld):
    """cld: (T,48,48) in [0,1] -> dict of fragmentation indices."""
    T = cld.shape[0]
    b = cld > 0.5
    cf = b.reshape(T, -1).mean(1)
    partly = (cf > 0.02) & (cf < 0.98)
    ncomp, patch, edge = [], [], []
    for t in range(T):
        if not partly[t]:
            continue
        lab, n = ndimage.label(b[t], structure=np.ones((3, 3), int))
        if n == 0:
            continue
        sizes = ndimage.sum(b[t], lab, range(1, n + 1))
        ncomp.append(float(n))
        patch.append(float(np.mean(sizes)))
        # 云边界像素占云像素的比例：4-邻域里有晴空像素
        eroded = ndimage.binary_erosion(b[t], structure=np.ones((3, 3), int))
        edge.append(float((b[t] & ~eroded).sum() / max(1.0, b[t].sum())))
    return {
        "frac_partly": float(partly.mean()),
        "n_partly": int(partly.sum()),
        "ncomp": float(np.median(ncomp)) if ncomp else 0.0,
        "patch": float(np.median(patch)) if patch else 0.0,
        "patch_frac": float(np.median(patch) / (48 * 48)) if patch else 0.0,
        "edge": float(np.median(edge)) if edge else 0.0,
        "cf_mean": float(cf.mean()),
    }


def load(run, mask, fixed):
    fn = "scene_psnr_test_%s%s.json" % (mask, "_fixed" if fixed else "")
    p = os.path.join(RUNS, run, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def paired(a, b, keys):
    x = np.array([a[k] for k in keys], float)
    y = np.array([b[k] for k in keys], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = y - x
    if len(d) < 8:
        return None
    t, p = stats.ttest_rel(y, x)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                n=int(len(d)))


def main():
    ds = S2DS("test")
    print("test samples:", len(ds))
    frag = {}
    for j in range(len(ds)):
        s = ds[j]
        frag[str(s["scene"])] = frag_metrics(s["cld"].numpy())
    print("fragmentation computed:", len(frag))

    fp = np.array([v["frac_partly"] for v in frag.values()])
    pa = np.array([v["patch"] for v in frag.values()])
    nc = np.array([v["ncomp"] for v in frag.values()])
    print("\n破碎度分布（n=%d 样本）" % len(frag))
    for nm, v in (("部分云帧占比 frac_partly", fp), ("云斑块平均面积 patch(px)", pa),
                  ("云斑块数量 ncomp", nc)):
        print("   %-24s min %.3f  p25 %.3f  中位 %.3f  p75 %.3f  max %.3f"
              % (nm, v.min(), np.percentile(v, 25), np.median(v), np.percentile(v, 75), v.max()))
    print("   完全无部分云帧的样本（整帧云主导）: %d/%d"
          % (int((fp == 0).sum()), len(fp)))

    scene_psnr = {}
    for arm, pre in PREFIX.items():
        scene_psnr[arm] = {}
        for s in SEEDS:
            scene_psnr[arm][s] = load("%s%d" % (pre, s), MASK, arm in ("B1", "B2"))
        miss = [s for s in SEEDS if scene_psnr[arm][s] is None]
        if miss:
            print("   !! arm %s 缺种子 %s" % (arm, miss))

    # 分组：整帧云主导 vs 混合；混合内按 patch 中位数再分两半
    keys_all = sorted(frag)
    mixed = [k for k in keys_all if frag[k]["frac_partly"] > 0]
    blocky0 = [k for k in keys_all if frag[k]["frac_partly"] == 0]
    if mixed:
        thr = float(np.median([frag[k]["patch"] for k in mixed]))
        frag_half = [k for k in mixed if frag[k]["patch"] <= thr]   # 斑块小 = 破碎
        block_half = [k for k in mixed if frag[k]["patch"] > thr]   # 斑块大 = 大块
    else:
        thr, frag_half, block_half = 0.0, [], []

    groups = {
        "全部样本": keys_all,
        "整帧云主导（无部分云帧）": blocky0,
        "云细碎（patch ≤ 中位 %.0f px）" % thr: frag_half,
        "云大块（patch > 中位 %.0f px）" % thr: block_half,
    }

    res = {"thr_patch": thr, "groups": {}, "frag": frag}
    print("\n=== %s 端点，逐种子配对 Δ dB ===" % MASK.upper())
    for gname, gkeys in groups.items():
        if len(gkeys) < 8:
            print("   %-28s n=%d 太小，跳过" % (gname, len(gkeys)))
            continue
        row = {"n_samples": len(gkeys), "per_seed": {}, "seed_level": {}}
        print("   %-28s n=%d" % (gname, len(gkeys)))
        for eff, (a_arm, b_arm) in (("B−A", ("A", "B")),
                                    ("B1−A", ("A", "B1")),
                                    ("B2−A", ("A", "B2")),
                                    ("B−B2", ("B2", "B"))):
            vals, ps = [], []
            for s in SEEDS:
                A, B = scene_psnr[a_arm][s], scene_psnr[b_arm][s]
                if A is None or B is None:
                    continue
                ks = sorted(set(A) & set(B) & set(gkeys))
                r = paired(A, B, ks)
                if r is None:
                    continue
                row["per_seed"].setdefault(eff, {})[s] = r
                vals.append(r["delta"])
                ps.append(r["p"])
            if not vals:
                continue
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            t, p = stats.ttest_1samp(vals, 0.0) if len(vals) > 1 else (float("nan"), float("nan"))
            row["seed_level"][eff] = {"mean": m, "se": se, "p": float(p),
                                      "per_seed": [float(v) for v in vals],
                                      "n_seeds": len(vals)}
            print("      %-6s 逐种子 %s  种子级 %+.3f ± %.3f (p=%.3f)"
                  % (eff, " ".join("%+.3f" % v for v in vals), m, se, p))
        res["groups"][gname] = row

    # 破碎度与逐像素通路增益的相关（探索性）
    print("\n探索性：逐样本 (B1−A) − (B2−A) 与破碎度的相关（seed 2026）")
    A, B1, B2 = scene_psnr["A"][2026], scene_psnr["B1"][2026], scene_psnr["B2"][2026]
    ks = sorted(set(A) & set(B1) & set(B2) & set(frag))
    dpx = np.array([B1[k] - A[k] for k in ks])
    dfr = np.array([B2[k] - A[k] for k in ks])
    ptc = np.array([frag[k]["patch"] for k in ks])
    ncm = np.array([frag[k]["ncomp"] for k in ks])
    ok = np.isfinite(dpx) & np.isfinite(dfr)
    for nm, v in (("patch", ptc), ("ncomp", ncm)):
        r1 = stats.spearmanr(v[ok], (dpx - dfr)[ok])
        r2 = stats.spearmanr(v[ok], dpx[ok])
        res.setdefault("corr", {})[nm] = {
            "rho_dpx_minus_dfr": float(r1.statistic), "p": float(r1.pvalue),
            "rho_dpx": float(r2.statistic), "p_dpx": float(r2.pvalue)}
        print("   %-6s vs (B1−A)−(B2−A): rho %+.3f p=%.3f | vs (B1−A): rho %+.3f p=%.3f"
              % (nm, r1.statistic, r1.pvalue, r2.statistic, r2.pvalue))

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
