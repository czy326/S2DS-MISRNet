# -*- coding: utf-8 -*-
"""大修清单 第 5 项 —— 云掩膜翻转噪声下的性能衰减曲线。

与已有 8×8 块置换的区别：块置换保留帧级统计量、只破坏空间对应；像素随机翻转同时污染
逐像素掩膜与由它派生的帧级晴空率 q，模拟真实云检测算法的漏检与虚警，是更严苛的退化。

对照口径（保守）：零训练云感知基线 cloudaware_mean 的掩膜**不加噪声**（HARD 端点
26.523 dB），而 arm B / arm D 的掩膜加噪声。若 arm 在该口径下仍高于基线，说明方法在
掩膜劣化时依然可用；这是比"让基线一起劣化"更严苛的判据。
"""
import json
import os

import numpy as np
from scipy import stats

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/flip_result.json"
SEEDS = [2026, 2027, 2028]
RATES = [("000", 0.00), ("050", 0.05), ("100", 0.10), ("150", 0.15)]
BASE_HARD = 26.523          # cloudaware_mean，掩膜无噪声
# arm B 的 2026/2027 检查点未保留在 s2dsV2_ 下，改用 s2dsR_（已验证逐样本 PSNR 完全一致）
RUN = {"B": {2026: "s2dsR_armB_s2026", 2027: "s2dsR_armB_s2027", 2028: "s2dsV2_armB_s2028"},
       "D": {s: "s2dsV2_armD_s%d" % s for s in SEEDS}}


def load(run, tag):
    p = os.path.join(RUNS, run, "scene_psnr_test_hard_%s.json" % tag)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def paired(a, b):
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = y - x
    t, p = stats.ttest_rel(y, x)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(d.mean()),
                ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                p=float(p), n=int(len(d)))


def main():
    res = {"rates": [r for _, r in RATES], "arms": {}}
    for arm in ("B", "D"):
        res["arms"][arm] = {}
        print("\n===== arm %s（HARD 端点，掩膜翻转噪声） =====" % arm)
        print("  %-6s | %s" % ("噪声", " ".join("seed%d" % s for s in SEEDS)) + " | 种子级均值 ± SE | 相对无噪声 | 高于基线")
        rows = {}
        for s in SEEDS:
            rows[s] = {tag: load(RUN[arm][s], "flip" + tag) for tag, _ in RATES}
            if rows[s]["000"] is None:
                print("   seed%d 缺 rate=0 参考，跳过该种子" % s)
        base = {s: rows[s]["000"] for s in SEEDS if rows[s]["000"]}
        for tag, rate in RATES:
            means, decays = [], []
            for s in SEEDS:
                if rows[s][tag] is None or s not in base:
                    continue
                ks = sorted(rows[s][tag])
                v = np.array([rows[s][tag][k] for k in ks], float)
                v = v[np.isfinite(v)]
                means.append(float(v.mean()))
                if rate > 0:
                    decays.append(paired(base[s], rows[s][tag])["delta"])
            if not means:
                continue
            m = float(np.mean(means))
            se = float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
            dec = float(np.mean(decays)) if decays else 0.0
            res["arms"][arm]["rate_%.2f" % rate] = {
                "per_seed": [float(v) for v in means], "mean": m, "se": se,
                "decay_vs_clean": [float(v) for v in decays], "decay_mean": dec,
                "margin_over_baseline": m - BASE_HARD,
                "above_baseline": bool(m > BASE_HARD)}
            print("  %-6s | %s | %6.3f ± %.3f | %+7.3f | %+7.3f %s"
                  % ("%.0f%%" % (rate * 100),
                     " ".join("%6.3f" % v for v in means), m, se, dec,
                     m - BASE_HARD, "是" if m > BASE_HARD else "否"))
        # 交叉点：arm 均值跌破基线的噪声水平（线性插值）
        xs = [r for _, r in RATES]
        ys = [res["arms"][arm].get("rate_%.2f" % r, {}).get("mean") for r in xs]
        if all(y is not None for y in ys):
            cross = None
            for i in range(len(xs) - 1):
                if (ys[i] - BASE_HARD) * (ys[i + 1] - BASE_HARD) < 0:
                    f = (ys[i] - BASE_HARD) / (ys[i] - ys[i + 1])
                    cross = xs[i] + f * (xs[i + 1] - xs[i])
                    break
            res["arms"][arm]["crossing_rate"] = cross
            print("  跌破零训练基线的噪声水平：%s"
                  % ("约 %.1f%%" % (cross * 100) if cross is not None
                     else "在 0~15%% 内未跌破" if ys[-1] > BASE_HARD else "已在 0%% 以下"))

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
