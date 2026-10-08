# -*- coding: utf-8 -*-
"""Carry the 6.6 granularity caveat into the abstract and the conclusion.

The permutation control changes what can be claimed about q, so the two places that state
the q headline must state the caveat too (anchors asserted unique).
"""
import io

DRAFT = "/mnt/e/论文2/output/a9ae06bf-ad32-4b7f-b35d-80fae7e57a6e/stage1/final_draft.md"

ABS_ANCHOR = "（+0.15 ~ +0.84 dB）；(2) 配合逐像素注意力"
ABS_ADD = ("（+0.15 ~ +0.84 dB）。置换对照进一步限定了它的归因：将每帧云掩膜做 8×8 块置换后"
           "（帧级晴空率逐样本保留、逐像素空间对应被摧毁），用打乱掩膜重新训练的 arm B 反而"
           "优于用真掩膜训练的版本（HARD 上 +0.271 dB，p = 0.0015），故该增益应归因于帧级"
           "可靠性先验，而非逐像素掩膜结构；(2) 配合逐像素注意力")

CONC_ANCHOR = "（B/D 臂 +0.22 ~ +0.56 dB）；"
CONC_ADD = ("（B/D 臂 +0.22 ~ +0.56 dB），但按 6.6 节的置换对照，q 的增益只能归因到帧级晴空率"
            "先验，逐像素云掩膜的独立价值在本预算与 SCL 质量下未获支持；")


def patch(text, anchor, addition):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit("anchor occurs %d times: %r" % (n, anchor[:60]))
    return text.replace(anchor, addition)


def main():
    t = io.open(DRAFT, encoding="utf-8").read()
    b = len(t)
    t = patch(t, ABS_ANCHOR, ABS_ADD)
    t = patch(t, CONC_ANCHOR, CONC_ADD)
    io.open(DRAFT, "w", encoding="utf-8").write(t)
    print("draft %d -> %d chars (+%d)" % (b, len(t), len(t) - b))
    for p in ["置换对照进一步限定了它的归因", "q 的增益只能归因到帧级晴空率"]:
        print("  %-30s %d" % (p, t.count(p)))


if __name__ == "__main__":
    main()
