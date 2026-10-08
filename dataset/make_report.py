"""Render dataset/RESULTS.md from stats.json + bench_baselines.json + manifest_built.json."""
import os
import json
import datetime

D = "/mnt/e/论文2/dataset"


def load(name, default=None):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    st = load("stats.json") or {}
    bn = load("bench_baselines.json") or {}
    mf = load("manifest_built.json") or []
    L = []
    A = L.append
    A("# 自建多时相 Sentinel-2 SR 数据集 · 结果汇总")
    A("")
    A("生成时间：%s（自动生成，勿手工编辑）" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    A("")
    A("## 1. 规模")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| AOI 数 | %s |" % st.get("n_aois"))
    A("| 总样本 | %s |" % st.get("total_samples"))
    A("| test 空间单元（AOI×patch 位置） | %s |" % st.get("n_spatial_units_test"))
    A("| HR patch / LR patch | %s×%s (10 m) / %s×%s (40 m) |" % (
        st.get("patch_hr"), st.get("patch_hr"), st.get("patch_lr"), st.get("patch_lr")))
    A("| 输入帧数 T | %s |" % st.get("T"))
    A("| 波段 | %s |" % ", ".join(st.get("bands", [])))
    A("")
    A("各划分：")
    A("")
    A("| split | AOI | 样本 |")
    A("|---|---|---|")
    for k, v in (st.get("per_split") or {}).items():
        A("| %s | %s | %s |" % (k, v.get("n_aois"), v.get("n_samples")))
    A("")
    A("## 2. 时间结构（真实获取日期）")
    A("")
    t = st.get("temporal") or {}
    A("| 指标 | 值 |")
    A("|---|---|")
    for k, lab in [("abs_dt_median_days", "|Δt| 中位（天）"),
                   ("abs_dt_p90_days", "|Δt| p90"),
                   ("abs_dt_max_days", "|Δt| 最大"),
                   ("frame_gap_median_days", "相邻输入帧间隔中位"),
                   ("frame_gap_p90_days", "相邻输入帧间隔 p90"),
                   ("frame_frac_gap_gt10", "输入帧间隔 >10 天占比"),
                   ("usable_gap_median_days", "**有效（晴空）观测**间隔中位"),
                   ("usable_gap_p90_days", "**有效观测**间隔 p90"),
                   ("usable_gap_max_days", "**有效观测**间隔最大"),
                   ("usable_frac_gap_gt10", "有效观测间隔 >10 天占比"),
                   ("usable_frac_gap_gt20", "有效观测间隔 >20 天占比")]:
        if k in t:
            A("| %s | %.3g |" % (lab, t[k]))
    A("")
    A("## 3. 云 / 可靠性信号")
    A("")
    c = st.get("cloud") or {}
    A("| 指标 | 值 |")
    A("|---|---|")
    for k, lab in [("frame_clear_frac_mean", "输入帧晴空率均值"),
                   ("frame_clear_frac_p10", "输入帧晴空率 p10"),
                   ("frame_frac_below_30pct_clear", "晴空率 <30% 的帧占比"),
                   ("lr_pixel_cloudfrac_mean", "LR 像素平均云占比"),
                   ("lr_pixel_frac_clear", "完全无云的 LR 像素占比"),
                   ("hr_target_clear_frac_mean", "HR 目标晴空率均值"),
                   ("hr_target_frac_clear_gt70", "HR 目标晴空率 >70% 的样本占比")]:
        if k in c:
            A("| %s | %.3g |" % (lab, c[k]))
    A("")
    A("## 4. 每个 AOI")
    A("")
    A("| AOI | split | 样本 | 日期数 | 目标数 | 可用日期 | 有效间隔中位 | 有效间隔p90 |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in (st.get("per_aoi") or {}).items():
        A("| %s | %s | %s | %s | %s | %s | %.0f | %.0f |"
          % (k, v.get("split"), v.get("n"), v.get("n_dates"), v.get("n_targets"),
             v.get("n_usable"), v.get("gap_median", 0), v.get("gap_p90", 0)))
    A("")
    A("## 5. 基线benchmark（在 HR 目标晴空像素上评 PSNR）")
    A("")
    if bn.get("splits"):
        for split, sv in bn["splits"].items():
            A("### split = %s" % split)
            A("")
            A("| 基线 | 均值 PSNR | 中位 | sd | n |")
            A("|---|---|---|---|---|")
            for k, v in (sv.get("per_baseline") or {}).items():
                A("| %s | %.4f | %.4f | %.4f | %d |" % (k, v["mean"], v["median"], v["sd"], v["n"]))
            A("")
            for k, v in (sv.get("comparisons") or {}).items():
                A("- `%s`：Δ=%+.4f dB，95%%CI [%+.4f, %+.4f]，d_z=%+.2f，p=%.3g（n=%d）"
                  % (k, v["delta"], v["ci"][0], v["ci"][1], v["dz"], v["p"], v["n"]))
            A("")
    else:
        A("（尚未产出）")
    A("")
    A("## 6. 文件位置")
    A("")
    A("- 原始时序 npz：`/home/czy/data/s2ds/<aoi>_2024.npz`（WSL 内）")
    A("- 成型 shard：`/home/czy/data/s2ds_built/<split>_<aoi>.npz`")
    A("- 清单：`dataset/manifest_built.json`；统计：`dataset/stats.json`")
    A("- 预览图：`dataset/preview_<aoi>.png`")
    A("- 采集脚本：`dataset/collect_s2.py`；成型：`dataset/build_dataset.py`；基线：`dataset/bench_baselines.py`")
    open(os.path.join(D, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote RESULTS.md (%d lines)" % len(L))


if __name__ == "__main__":
    main()
