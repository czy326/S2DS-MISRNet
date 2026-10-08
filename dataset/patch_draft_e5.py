# -*- coding: utf-8 -*-
"""Insert §6.6 (cloud-mask permutation control) into the manuscript draft.

Every replacement asserts that its anchor occurs exactly once, so a silently-drifting
draft fails loudly instead of being patched in the wrong place.
"""
import io
import os
import sys

DRAFT = "/mnt/e/论文2/output/a9ae06bf-ad32-4b7f-b35d-80fae7e57a6e/stage1/final_draft.md"

SEC = """### 6.6 云掩膜置换对照：q 的收益不来自逐像素结构

arm B 的云信息经三条通路进入网络：(i) 逐像素特征抑制 f ← f·(1 − σ(γ)·cld)；(ii) 逐像素注意力门中的 cld 通道；(iii) 帧级晴空率嵌入 f ← f + q_mlp(q)，其中 q = 1 − mean(cld)。为把前两条（逐像素）与第三条（帧级）分开，我们对每帧云掩膜做 8×8 块置换：48×48 的掩膜可被 8×8 整除，块置换是精确的，因此**帧级晴空率 q 逐样本被逐比特保留**，被摧毁的只有掩膜与图像内容的空间对应——这是一次只切逐像素信息、不动帧级信息的手术。

**评估时置换（同一套已训练权重，seed 2028）。** 把打乱后的掩膜喂给训好的 arm B，HARD 上增益从 +0.360 dB [+0.226, +0.495] 降到 +0.133 dB [−0.058, +0.323]，p = 0.17，保留 37%；valid 口径下从 +0.509 降到 +0.263，保留 52%（表 10）。这说明训好的模型在推理时**确实逐像素地依赖云掩膜**——它并没有把掩膜压缩成一个标量来用。

**训练时置换（重新训练，seed 2026）。** 该对照的结果与其预注册预期相反：用打乱掩膜训练的 arm B **优于**用真掩膜训练的 arm B。B_perm − B_true = +0.271 dB（95% CI [+0.105, +0.438]，p = 0.0015，n = 299）于 HARD 端点、+0.428 dB（95% CI [+0.289, +0.567]，p = 3.6×10⁻⁹，n = 396）于 valid 端点；三者的绝对精度为 A 26.444 / B_true 26.743 / B_perm 27.014 dB（HARD）。即"保留比例"不是预期的 ≤ 50%，而是 191%（HARD）。除置换开关外，两个运行的训练配置逐项相同（arm B、seed 2026、30,000 iters、batch 8、逐像素注意力、gate 注入、余弦退火）。

两项结果并不矛盾，而是共同限定了可主张的范围：模型**使用**逐像素云结构（评估时置换会使其失效），但 q 相对 arm A 的收益**并不来自**逐像素云结构（用空间无意义的掩膜训练反而更好）。我们给出两种候选解释，并由随后的 q 通路消融判决（B1 = 仅保留两条逐像素通路，B2 = 仅保留帧级通路）：其一是帧级晴空率已经足够、两条逐像素通路在本预算下是净负担——这与 3.5 节记录的 SCL 噪声一致，40 m 重采样后的逐像素云概率不足以支撑逐像素决策；其二是打乱掩膜等价于一次逐像素随机调制，起到与云语义无关的正则化作用。两种解释下本文都不主张"逐像素云掩膜带来了 q 的收益"；可确立的陈述是**帧级可靠性先验足以解释 q 的全部增益**。

**表 10** 云掩膜 8×8 块置换对照（固定迭代口径，逐样本配对 Δ dB [95% CI]）

| 口径 | 对比 | HARD | valid |
|---|---|---|---|
| 评估时（seed 2028） | 真掩膜：B − A | +0.360 [+0.226, +0.495] | +0.509 [+0.387, +0.630] |
| | 打乱掩膜：B − A | +0.133 [−0.058, +0.323] | +0.263 [+0.109, +0.417] |
| | 保留比例 | 37% | 52% |
| 训练时（seed 2026） | 真掩膜：B − A | +0.299 | +0.151 |
| | 打乱掩膜：B − A | +0.570 | +0.579 |
| | B_perm − B_true | +0.271 [+0.105, +0.438] | +0.428 [+0.289, +0.567] |

"""

DISCUSS_ADD = """这一论断需要一处重要限定：6.6 节的置换对照表明，模型在推理时确实逐像素地使用云掩膜，但 q 的收益并不来自逐像素云结构——用空间上被打乱的掩膜训练出的 arm B 反而更好（HARD 上 +0.271 dB，p = 0.0015）。因此"逐像素"这一要求应严格归于**融合权重**（标量权重无法表达同一帧在不同位置的可信度差异），而**不应归于云条件化信息的粒度**：在本数据上，帧级晴空率已足以解释 q 的全部增益。

"""

LIMIT_ADD = """9. **云条件化的粒度未被确立为逐像素**：6.6 节的置换对照显示，用空间打乱的云掩膜训练 arm B 反而优于用真掩膜训练（HARD +0.271 dB，p = 0.0015），故 q 的收益只能归因到帧级可靠性先验；逐像素云掩膜在本文的实验预算与 SCL 质量下未能证明其独立价值。

"""


def patch(text, anchor, addition, where="before"):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit("anchor occurs %d times: %r" % (n, anchor[:60]))
    if where == "before":
        return text.replace(anchor, addition + anchor)
    return text.replace(anchor, anchor + addition)


def main():
    t = io.open(DRAFT, encoding="utf-8").read()
    before = len(t)
    t = patch(t, "## 7 讨论", SEC)
    t = patch(t, "**Δt 为什么无效：三重冗余。**", DISCUSS_ADD, where="before")
    t = patch(t, "8. **多重比较**：", LIMIT_ADD, where="before")
    io.open(DRAFT, "w", encoding="utf-8").write(t)
    print("draft %d -> %d chars (+%d)" % (before, len(t), len(t) - before))
    for probe in ["### 6.6 云掩膜置换对照", "表 10", "这一论断需要一处重要限定",
                  "9. **云条件化的粒度未被确立为逐像素**"]:
        print("  %-40s %d" % (probe, t.count(probe)))


if __name__ == "__main__":
    main()
