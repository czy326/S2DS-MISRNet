# S2DS benchmark and MISRNet

Code accompanying the manuscript

> **Cloud-Reliability-Gated Fusion for Multi-Temporal Sentinel-2 Super-Resolution and Its Factorial Attribution**

## Contents

| Path | Description |
|---|---|
| `misr/` | MISRNet (pixel-wise temporal attention, dual-path cloud-reliability gating) together with the zero-training and learned baselines, plus the training and evaluation entry points (`train_s2ds.py`, `eval_s2ds.py`, `analyze_s2ds_2x2.py`). |
| `dataset/` | S2DS construction, the preregistered 2x2 factor-effect analysis, the path-decomposition and cloud-fragmentation analyses, hyper-parameter scans, the frame-order shuffle control, and the statistical power analysis. |
| `figures/` | Scripts that regenerate every figure of the paper. |
| `results/` | Measured model complexity (parameters, MACs) and GPU inference latency used in Table A1 and Figure 1. |

## S2DS dataset

S2DS is a multi-temporal Sentinel-2 super-resolution benchmark. Each sample consists of
T = 12 real Sentinel-2 L2A acquisitions (bands B02, B03, B04, B08) at an equivalent 40 m
sampling, together with the real acquisition timestamps and the per-pixel Sen2Cor cloud
probability, and a 10 m target image of the target date. Four evaluation endpoints/strata
are provided (HARD, valid, EASY) via the cloud masks.

* Source imagery: Sentinel-2 L2A cloud-optimized GeoTIFFs on the AWS Open Data Registry
  (https://registry.opendata.aws/sentinel-2-l2a-cogs).
* 18 spatially disjoint regions of interest; 873 / 180 / 423 samples for training,
  validation and testing.
* License for the dataset: CC BY 4.0. Code: MIT (see `LICENSE`).

## Environment

Python 3.11, PyTorch 2.11 (CUDA 12.8), `numpy`, `opencv-python`, `scipy`, `pandas`,
`matplotlib`. All experiments were run on a single NVIDIA GeForce RTX 5080 Laptop GPU
(16 GB).

```bash
pip install -r requirements.txt
```

## Notes

* The scripts contain absolute paths from the original working environment
  (`/mnt/e/...`, `/home/czy/data`). Adjust the `RUNS` / data-root constants at the top of
  each script to your own layout.
* `figures/` expects the evaluation outputs produced by `misr/eval_s2ds.py`.
