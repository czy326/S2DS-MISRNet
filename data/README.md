# S2DS dataset

Benchmark used by the paper *Cloud-Reliability-Gated Fusion for Multi-Temporal Sentinel-2
Super-Resolution and Its Factorial Attribution*.

## Layout

```
data/
├── s2ds/          raw per-AOI collections from Sentinel-2 L2A (18 files, one per AOI)
└── s2ds_built/    benchmark splits consumed by the model (train_/val_/test_<aoi>.npz)
```

## `s2ds_built/<split>_<aoi>.npz`

Each archive holds N samples of one region of interest.

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `lr` | (N, 12, 4, 48, 48) | uint16 | 12 temporally ordered low-resolution frames, bands B02/B03/B04/B08, 40 m, reflectance x 10^4 |
| `hr` | (N, 4, 192, 192) | uint16 | 10 m target image of the target date, reflectance x 10^4 |
| `cld` | (N, 12, 48, 48) | uint8 | per-pixel cloud probability from the Sen2Cor SCL product (0 = clear, 255 = cloud) |
| `dt` | (N, 12) | int16 | signed temporal distance to the target date, in days |
| `clearfrac` | (N, 12) | float32 | global clear-sky ratio q of each frame |
| `hr_cloud` | (N, 192, 192) | uint8 | cloud mask of the target date |
| `hr_clear` | (N,) | float32 | clear-sky ratio of the target image |
| `scene`, `date`, `aoi` | (N,) | str | sample identifier, target date, AOI name |

Frames are ordered by increasing |dt|, so frame index 0 is always the nearest available
frame to the target date. The HARD / valid / EASY endpoints used in the paper are derived
from `hr_cloud` and `cld` (see `misr/eval_s2ds.py`).

## `s2ds/<aoi>_2024.npz` (raw collection)

| Key | Shape | Meaning |
|---|---|---|
| `lr40` | (M, 4, 80, 80) | all 40 m low-resolution frames available for the AOI |
| `scl20` | (M, 160, 160) | Sen2Cor SCL at 20 m |
| `dates`, `cloudfrac`, `stac_cloud` | (M,) | acquisition dates and cloud statistics |
| `hr10` | (K, 4, 320, 320) | 10 m reference chips |
| `tgt_idx`, `lr_check`, `tgt_clear` | | target index, alignment check, target clear fraction |

`s2ds_built` is produced from `s2ds` by `dataset/build_dataset.py`; the raw archives are
provided so that the dataset can be rebuilt or re-split.

## Splits

18 spatially disjoint regions of interest, 873 / 180 / 423 samples for training, validation
and testing (1,476 samples in total).

## License and source imagery

Released under CC BY 4.0. Source imagery: Sentinel-2 L2A (Copernicus programme), accessed
through the AWS Open Data Registry.
