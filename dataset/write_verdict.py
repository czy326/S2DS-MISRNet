"""Write dataset/INTERACTION_RESULT.md from the 2x2 analysis JSONs (all masks).

Primary endpoint for this round is the HARD stratum (target clear, nearest input frame
cloudy) -- the only pixels where a single frame cannot help.  The `valid` stratum is kept
for comparison.  Rules are pre-registered in dataset/EXPERIMENT_PROTOCOL.md.
"""
import os
import glob
import json
import datetime

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/INTERACTION_RESULT.md"

ORDER = ["F2_q@no_dt (B-A)", "F2_q@dt    (D-C)", "F1_dt@no_q (C-A)", "F1_dt@q    (D-B)",
         "INTERACTION (D-C)-(B-A)", "JOINT       (D-A)",
         "F1 main effect", "F2 main effect"]


def table(r):
    L = ["| 效应 | 样本级 Δ (dB) | 95% CI | d_z | p | 单元级 Δ (dB) | 95% CI | p |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        s = r["sample_level"].get(k, {})
        u = r["unit_level"].get(k, {})
        if "mean" not in s:
            continue
        L.append("| %s | %+.4f | [%+.4f, %+.4f] | %+.2f | %.3g | %+.4f | [%+.4f, %+.4f] | %.3g |"
                 % (k.strip(), s["mean"], s["ci"][0], s["ci"][1], s["dz"], s["p"],
                    u.get("mean", float("nan")), u.get("ci", [float("nan")] * 2)[0],
                    u.get("ci", [float("nan")] * 2)[1], u.get("p", float("nan"))))
    return L


def main():
    files = sorted(glob.glob(os.path.join(RUNS, "analysis_s*_*.json")))
    R = {}
    for f in files:
        r = json.load(open(f))
        R[(r["seed"], r["split"], r.get("mask", "valid"))] = r
    L = ["# 2×2 交互效应验证结果（Δt × 云可靠性 q）", "",
         "生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
         "本轮做了两处协议修正：",
         "1. **检查点按 val 最优选择**（在 `ckpt_*.pt` 中挑 val 最高的存为 `best.pt`），消除 `last.pt` 上 ±0.5 dB 的选择噪声；",
         "2. **主端点改为 HARD 像素**（目标晴空、但时间上最近的那一帧被云遮蔽）——只有那里单帧无能为力、跨时相融合才被真正需要。",
         "", "判定规则（预先写死）：交互量 I=(D−C)−(B−A) 需 **≥ +0.10 dB 且配对 95% CI 下界 > 0**；",
         "样本级与空间单元级必须同向；两个种子方向必须一致。", ""]
    for mask, title in [("hard", "主端点：HARD 像素（最近帧有云）"),
                        ("valid", "对照：全部晴空像素")]:
        L += ["## %s" % title, ""]
        keys = sorted(k for k in R if k[2] == mask)
        if not keys:
            L += ["（尚无结果）", ""]
            continue
        for (seed, split, _) in keys:
            r = R[(seed, split, mask)]
            L += ["### seed = %s, split = %s（n_samples=%s）" % (seed, split, r["n_samples"]), "",
                  "arm 均值：" + "，".join("%s %.4f" % (a, v)
                                          for a, v in sorted(r["arm_means"].items())), ""]
            L += table(r)
            L += ["", "**交互判定**：样本级 = %s；单元级 = %s ⇒ **%s**"
                  % (r["verdict_sample"], r["verdict_unit"], r["verdict"]), ""]
        L += ["### 跨种子一致性（split=test）", "",
              "| seed | 交互 Δ | 95% CI | 判定 |", "|---|---|---|---|"]
        signs = []
        for (seed, split, _) in sorted(k for k in keys if k[1] == "test"):
            v = R[(seed, split, mask)]["sample_level"]["INTERACTION (D-C)-(B-A)"]
            signs.append(v["mean"] > 0 and v["ci"][0] > 0)
            L.append("| %s | %+.4f | [%+.4f, %+.4f] | %s |"
                     % (seed, v["mean"], v["ci"][0], v["ci"][1],
                        R[(seed, split, mask)]["verdict"]))
        L += ["", "**多种子结论**：%s" % ("所有种子方向一致且显著 ⇒ 交互效应可复现"
                                          if signs and all(signs) else
                                          "种子间方向/显著性不一致 ⇒ 交互效应未被复现"), ""]
    L += ["## 基线（零训练）参考", "",
          "- `cloudaware_mean`（云感知加权平均）test = 27.046 dB，`cloudweighted_mean` = 27.050 dB；",
          "- 分层：EASY 像素 28.05 / HARD 像素 26.52；直接抄最近帧在 HARD 像素上只有 11.41 dB。",
          "- ⇒ 学习型模型必须超过 `cloudaware_mean` 才有讨论条件化因子的余地。", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT, "(%d lines)" % len(L))


if __name__ == "__main__":
    main()
