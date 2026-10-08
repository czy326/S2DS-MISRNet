# 自建多时相 Sentinel-2 SR 数据集 · 结果汇总

生成时间：2026-09-26 08:21（自动生成，勿手工编辑）

## 1. 规模

| 项 | 值 |
|---|---|
| AOI 数 | 18 |
| 总样本 | 1476 |
| test 空间单元（AOI×patch 位置） | 45 |
| HR patch / LR patch | 192×192 (10 m) / 48×48 (40 m) |
| 输入帧数 T | 12 |
| 波段 | B02, B03, B04, B08 |

各划分：

| split | AOI | 样本 |
|---|---|---|
| train | 10 | 873 |
| val | 3 | 180 |
| test | 5 | 423 |

## 2. 时间结构（真实获取日期）

| 指标 | 值 |
|---|---|
| |Δt| 中位（天） | 15 |
| |Δt| p90 | 30 |
| |Δt| 最大 | 80 |
| 相邻输入帧间隔中位 | 5 |
| 相邻输入帧间隔 p90 | 5 |
| 输入帧间隔 >10 天占比 | 0.00554 |
| **有效（晴空）观测**间隔中位 | 5 |
| **有效观测**间隔 p90 | 10 |
| **有效观测**间隔最大 | 25 |
| 有效观测间隔 >10 天占比 | 0.0621 |
| 有效观测间隔 >20 天占比 | 0.00473 |

## 3. 云 / 可靠性信号

| 指标 | 值 |
|---|---|
| 输入帧晴空率均值 | 0.442 |
| 输入帧晴空率 p10 | 0 |
| 晴空率 <30% 的帧占比 | 0.53 |
| LR 像素平均云占比 | 0.558 |
| 完全无云的 LR 像素占比 | 0.435 |
| HR 目标晴空率均值 | 0.831 |
| HR 目标晴空率 >70% 的样本占比 | 0.819 |

## 4. 每个 AOI

| AOI | split | 样本 | 日期数 | 目标数 | 可用日期 | 有效间隔中位 | 有效间隔p90 |
|---|---|---|---|---|---|---|---|
| chengdu_plain | train | 81 | 145 | 9 | 37 | 5 | 28 |
| harbin_agri | train | 90 | 145 | 10 | 70 | 5 | 10 |
| hohhot_steppe | train | 90 | 145 | 10 | 84 | 3 | 8 |
| lanzhou_valley | train | 81 | 72 | 9 | 37 | 10 | 18 |
| linzhi_valley | train | 90 | 71 | 10 | 31 | 10 | 21 |
| sjz_north_agri | train | 90 | 72 | 10 | 41 | 5 | 15 |
| urumqi_north | train | 90 | 144 | 10 | 84 | 3 | 7 |
| wuhan_lake | train | 81 | 142 | 9 | 93 | 3 | 5 |
| xian_plain | train | 90 | 69 | 10 | 33 | 5 | 20 |
| zhengzhou_east | train | 90 | 73 | 10 | 35 | 5 | 23 |
| bj_south | val | 90 | 142 | 10 | 81 | 3 | 8 |
| kunming_plateau | val | 27 | 147 | 3 | 105 | 3 | 5 |
| qingdao_inland | val | 63 | 145 | 7 | 108 | 3 | 5 |
| dg_north | test | 90 | 73 | 10 | 19 | 10 | 22 |
| dunhuang_oasis | test | 90 | 146 | 10 | 81 | 3 | 8 |
| haikou_north | test | 81 | 73 | 9 | 23 | 10 | 35 |
| sz_east | test | 81 | 71 | 9 | 22 | 10 | 45 |
| xiamen_inland | test | 81 | 72 | 9 | 22 | 15 | 35 |

## 5. 基线benchmark（在 HR 目标晴空像素上评 PSNR）

### split = val

| 基线 | 均值 PSNR | 中位 | sd | n |
|---|---|---|---|---|
| bicubic_firstframe | 21.4752 | 24.4873 | 9.5856 | 133 |
| bicubic_nearest | 21.4752 | 24.4873 | 9.5856 | 133 |
| bicubic_median | 22.6153 | 23.7776 | 6.0797 | 133 |
| timeweighted_mean | 14.7391 | 14.6830 | 3.8215 | 133 |
| cloudweighted_mean | 25.1128 | 27.9105 | 6.1709 | 133 |
| cloudaware_mean | 25.3470 | 27.9443 | 5.9611 | 133 |

- `bicubic_nearest -> bicubic_median`：Δ=+1.1401 dB，95%CI [-0.1438, +2.4240]，d_z=+0.15，p=0.0841（n=133）
- `bicubic_median -> cloudaware_mean`：Δ=+2.7317 dB，95%CI [+1.9682, +3.4952]，d_z=+0.61，p=1.09e-10（n=133）
- `bicubic_nearest -> cloudaware_mean`：Δ=+3.8718 dB，95%CI [+2.4511, +5.2925]，d_z=+0.46，p=3.92e-07（n=133）

### split = test

| 基线 | 均值 PSNR | 中位 | sd | n |
|---|---|---|---|---|
| bicubic_firstframe | 16.4467 | 16.3224 | 10.1693 | 396 |
| bicubic_nearest | 16.4467 | 16.3224 | 10.1693 | 396 |
| bicubic_median | 17.5753 | 18.5362 | 8.8984 | 396 |
| timeweighted_mean | 12.7743 | 11.9340 | 5.8235 | 396 |
| cloudweighted_mean | 27.0504 | 26.9625 | 3.6869 | 396 |
| cloudaware_mean | 27.0460 | 27.0043 | 3.7666 | 396 |

- `bicubic_nearest -> bicubic_median`：Δ=+1.1286 dB，95%CI [+0.1858, +2.0714]，d_z=+0.12，p=0.0195（n=396）
- `bicubic_median -> cloudaware_mean`：Δ=+9.4707 dB，95%CI [+8.6740, +10.2673]，d_z=+1.17，p=3.51e-76（n=396）
- `bicubic_nearest -> cloudaware_mean`：Δ=+10.5993 dB，95%CI [+9.6678, +11.5307]，d_z=+1.12，p=6.56e-72（n=396）


## 6. 文件位置

- 原始时序 npz：`/home/czy/data/s2ds/<aoi>_2024.npz`（WSL 内）
- 成型 shard：`/home/czy/data/s2ds_built/<split>_<aoi>.npz`
- 清单：`dataset/manifest_built.json`；统计：`dataset/stats.json`
- 预览图：`dataset/preview_<aoi>.png`
- 采集脚本：`dataset/collect_s2.py`；成型：`dataset/build_dataset.py`；基线：`dataset/bench_baselines.py`
