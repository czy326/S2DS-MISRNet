"""Build the multi-temporal Sentinel-2 SR dataset shards from collected AOI time series.

Handles both npz layouts produced by collect_s2.py:
  new : lr40 (n,4,h,w) 4x-decimated 40 m + scl20 (n,2h,2w) native 20 m + hr10 (k,4,4h,4w) targets
  old : lr   (n,4,W,W) full 10 m + scl (n,W/2,W/2) 20 m  (no hr10 -> HR comes from lr itself)

Task (see dataset/README.md)
---------------------------
Given T=12 Sentinel-2 L2A frames at REAL acquisition dates around a target date t*,
predict the L2A image of t* at native 10 m.

  HR   : 4 bands (B02/B03/B04/B08) 10 m, patch 192x192   (1.92 km)
  LR   : the same bands of each frame, 4x decimated -> 48x48 (40 m GSD)
  meta : signed dt per frame, per-frame clear fraction, per-LR-pixel cloud fraction
         (soft reliability), HR-level cloud mask, HR-level valid mask (clear target
         pixels = the standard evaluation mask)

t* is never an input; target dates are restricted to acquisitions that are clear
enough (>= TARGET_CLEAR of the AOI window), so the HR reference is a real image.

Outputs
-------
  /home/czy/data/s2ds_built/<split>_<aoi>.npz
  dataset/manifest_built.json, dataset/stats.json, dataset/preview_<aoi>.png
"""
import os
import json
import glob
import argparse
import datetime

import numpy as np
import cv2

SRC = "/home/czy/data/s2ds"
DST = "/home/czy/data/s2ds_built"
OUT = "/mnt/e/论文2/dataset"
SCALE = 4
PATCH = 192          # HR patch (10 m pixels)
STRIDE = 64          # HR stride  -> 3x3 positions inside a 320 window
T = 12               # input frames
TARGET_CLEAR = 0.5   # AOI-level clear fraction required of a target date
QC_MIN_CORR = 0.5    # legacy, unused: correlation-based QC was rejected (see Shard)
CLOUD_SCL = [3, 8, 9, 10]

SPLITS = {
    "train": ["sjz_north_agri", "zhengzhou_east", "harbin_agri", "xian_plain",
              "wuhan_lake", "chengdu_plain", "hohhot_steppe", "urumqi_north",
              "lanzhou_valley", "linzhi_valley"],
    "val": ["bj_south", "kunming_plateau", "qingdao_inland"],
    "test": ["sz_east", "dg_north", "dunhuang_oasis", "xiamen_inland", "haikou_north"],
}


def to_ord(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:])).toordinal()


def area(x, size):
    return cv2.resize(x.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def pick_idx(n, k):
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k)))


class Shard:
    """Uniform accessor over both npz layouts."""

    def __init__(self, path):
        self.z = np.load(path)
        self.path = path
        self.dates = [str(d) for d in self.z["dates"]]
        self.n = len(self.dates)
        self.ords = np.array([to_ord(d) for d in self.dates])
        self.legacy = "lr" in self.z.files
        if self.legacy:
            self.lr10 = self.z["lr"]                                  # (n,4,W,W)
            self.W = self.lr10.shape[-1]
            self.scl20 = self.z["scl"]
            self.lr40 = np.stack([np.stack([area(self.lr10[i][c], self.W // SCALE)
                                            for c in range(self.lr10.shape[1])])
                                  for i in range(self.n)]).astype(np.uint16)
            self.h = self.W // SCALE
        else:
            self.lr40 = self.z["lr40"]
            self.h = self.lr40.shape[-1]
            self.W = self.h * SCALE
            self.scl20 = self.z["scl20"]
            self.lr10 = None
        # per-40m-pixel cloud fraction from the native 20 m SCL (2x2 average)
        self.cld40 = np.stack([area(np.isin(self.scl20[i], CLOUD_SCL).astype(np.float32),
                                    self.h) for i in range(self.n)])
        self.clear = 1.0 - self.cld40.reshape(self.n, -1).mean(1)
        # ---- data-integrity QC (metadata + dropping only真 invalid dates) ----------
        # Verified on this dataset: every date of an AOI shares the same pixel grid and
        # window origin (checked against the products' own transforms), so there is NO
        # misregistration to hunt for.  A naive "correlation with the temporal median"
        # test was tried and rejected: it fires on clouds and on seasonal change, not on
        # misregistration, and would have discarded ~12% of legitimate dates.
        bmean = np.array([self.lr40[i].astype(np.float32).mean() for i in range(self.n)])
        self.brightness = bmean
        finite = np.isfinite(bmean)
        self.valid = finite & (bmean > 50.0)          # drop all-zero / empty reads only
        med = float(np.median(bmean[self.valid])) if self.valid.any() else 1.0
        self.qc_corr = np.where(self.valid & (bmean < 3.0 * med), bmean / max(med, 1e-6), 3.0)
        self.bright_frac = float((bmean > 3.0 * med).mean())
        if not self.legacy and "tgt_idx" in self.z.files:
            self.tgt = [int(i) for i in self.z["tgt_idx"] if self.valid[int(i)]]
        else:
            order = np.argsort(self.ords)
            ok = [int(j) for j in order if self.valid[j]]
            us = [j for j in ok if self.clear[j] >= TARGET_CLEAR]
            if len(us) < 3:
                us = [j for j in ok if self.clear[j] >= 0.35] or ok
            self.tgt = [us[i] for i in pick_idx(len(us), 10)]

    def hr10_patch(self, ti, oy, ox):
        """HR (4,PATCH,PATCH) for a target date."""
        if self.legacy:
            return self.lr10[ti][:, oy:oy + PATCH, ox:ox + PATCH]
        k = self.tgt.index(ti)
        return self.z["hr10"][k][:, oy:oy + PATCH, ox:ox + PATCH]

    def cloud10_patch(self, ti, oy, ox):
        """Target cloud mask at 10 m, upsampled nearest from SCL 20 m."""
        full = cv2.resize(np.isin(self.scl20[ti], CLOUD_SCL).astype(np.uint8),
                          (self.W, self.W), interpolation=cv2.INTER_NEAREST)
        return full[oy:oy + PATCH, ox:ox + PATCH]


def build_aoi(name, year, split, k_check=None):
    p = os.path.join(SRC, "%s_%d.npz" % (name, year))
    if not os.path.exists(p):
        return None
    sh = Shard(p)
    pos = list(range(0, sh.W - PATCH + 1, STRIDE))
    sl = PATCH // SCALE
    acc = {k: [] for k in ["hr", "lr", "dt", "cld", "hr_cloud", "hr_clear",
                           "clearfrac", "scene", "date", "dt_signed"]}
    for ti in sh.tgt:
        d_abs = np.abs(sh.ords - sh.ords[ti])
        sel = [int(j) for j in np.argsort(d_abs) if j != ti and sh.valid[j]][:T]
        if len(sel) < T:
            continue
        hr_cloud_full = sh.cloud10_patch(ti, 0, 0) if False else None  # noqa: F841
        for oy in pos:
            for ox in pos:
                oyl, oxl = oy // SCALE, ox // SCALE
                hr_p = sh.hr10_patch(ti, oy, ox)
                lr_p = np.stack([sh.lr40[j][:, oyl:oyl + sl, oxl:oxl + sl] for j in sel])
                # nodata handling: SCL reports nodata (class 0) as "not cloud", which would
                # make zero-filled areas look perfectly clear.  A pixel that is zero in
                # every band is invalid: exclude it from the HR truth and mark the LR
                # frame pixel as fully unreliable instead of trusting it.
                hr_nodata = (hr_p.max(0) == 0)
                lr_nodata = (lr_p.max(1) == 0)                       # (T,sl,sl)
                cld_v = sh.cld40[sel, oyl:oyl + sl, oxl:oxl + sl].copy()
                cld_v[lr_nodata] = 1.0
                acc["hr"].append(hr_p)
                acc["lr"].append(lr_p)
                acc["dt"].append((sh.ords[sel] - sh.ords[ti]).astype(np.int16))
                acc["dt_signed"].append((sh.ords[sel] - sh.ords[ti]).astype(np.int16))
                acc["cld"].append(np.round(cld_v * 255).astype(np.uint8))
                hc = sh.cloud10_patch(ti, oy, ox) | hr_nodata.astype(np.uint8)
                acc["hr_cloud"].append(hc)
                acc["hr_clear"].append(np.float32(1.0 - hc.mean()))
                acc["clearfrac"].append(
                    (1.0 - cld_v.reshape(cld_v.shape[0], -1).mean(1)).astype(np.float32))
                acc["scene"].append("%s_%s_%d_%d" % (name, sh.dates[ti], oy, ox))
                acc["date"].append(sh.dates[ti])
    if not acc["hr"]:
        return None
    out = os.path.join(DST, "%s_%s.npz" % (split, name))
    np.savez_compressed(
        out, hr=np.stack(acc["hr"]), lr=np.stack(acc["lr"]),
        dt=np.stack(acc["dt"]), cld=np.stack(acc["cld"]),
        hr_cloud=np.stack(acc["hr_cloud"]),
        hr_clear=np.array(acc["hr_clear"], np.float32),
        clearfrac=np.stack(acc["clearfrac"]),
        scene=np.array(acc["scene"]), date=np.array(acc["date"]),
        aoi=np.array([name] * len(acc["hr"])),
    )
    good = np.sort(sh.ords[sh.clear >= 0.5])
    ug = np.diff(good) if len(good) > 1 else np.array([0])
    return dict(aoi=name, split=split, n=len(acc["hr"]), path=out,
                size_mb=round(os.path.getsize(out) / 1e6, 2),
                layout="legacy" if sh.legacy else "v1",
                n_dates=sh.n, n_targets=len(sh.tgt),
                n_usable=len(good),
                n_dropped_qc=int((~sh.valid).sum()),
                bright_frac=sh.bright_frac,
                gap_median=float(np.median(ug)), gap_p90=float(np.percentile(ug, 90)),
                gap_max=float(ug.max()), clear_mean=float(sh.clear.mean()))


def preview(name, year):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = os.path.join(SRC, "%s_%d.npz" % (name, year))
    if not os.path.exists(p):
        return None
    sh = Shard(p)
    if sh.legacy:
        ti = int(np.argmax(sh.clear))
        raw = sh.lr10[ti]
    else:
        if not sh.tgt:
            return None
        k = int(np.argmax([sh.clear[i] for i in sh.tgt]))
        ti = sh.tgt[k]
        raw = sh.z["hr10"][k]
    rgb = raw[[2, 1, 0]].astype(np.float32)
    rgb = np.clip(rgb / max(np.percentile(rgb, 98), 1), 0, 1)
    nir, red = raw[3].astype(np.float32), raw[2].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    med40 = np.median(sh.lr40, 0)[1].astype(np.float32)
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.3))
    ax[0].imshow(np.transpose(rgb, (1, 2, 0)))
    ax[0].set_title("HR target %s (10 m)" % sh.dates[ti])
    ax[1].imshow(np.clip(sh.lr40[ti][1] / max(sh.lr40[ti][1].max(), 1), 0, 1), cmap="gray")
    ax[1].set_title("LR frame (40 m)")
    ax[2].imshow(np.clip(med40 / max(med40.max(), 1), 0, 1), cmap="gray")
    ax[2].set_title("median of %d LR frames" % sh.n)
    ax[3].imshow(sh.cloud10_patch(ti, 0, 0), cmap="gray")
    ax[3].set_title("cloud mask (%s)" % sh.dates[ti])
    ax[4].imshow(ndvi, cmap="RdYlGn", vmin=-0.4, vmax=0.9); ax[4].set_title("NDVI (target)")
    for k in range(5):
        ax[k].axis("off")
    dst = os.path.join(OUT, "preview_%s.png" % name)
    fig.tight_layout(); fig.savefig(dst, dpi=110); plt.close(fig)
    return dst


def effective_splits(year=2024):
    """Assign the AOIs that actually finished to splits, keeping the target proportions.

    Collection may lose an AOI (transient network failure); without rebalancing a split
    could end up empty or with far fewer spatial units than planned, which destroys the
    statistical power of the interaction test.  Deterministic: sorted AOI names.
    """
    have = set()
    for f in glob.glob(os.path.join(SRC, "*_%d.npz" % year)):
        have.add(os.path.basename(f)[:-(len("_%d.npz" % year))])
    eff = {s: sorted(a for a in names if a in have) for s, names in SPLITS.items()}
    want = {"train": 10, "val": 3, "test": 5}
    for s in ("val", "test"):
        short = want[s] - len(eff[s])
        if short > 0:
            pool = [a for a in eff["train"] if a not in eff["val"] and a not in eff["test"]]
            take = sorted(pool)[:short]
            eff[s] = sorted(eff[s] + take)
            eff["train"] = [a for a in eff["train"] if a not in take]
    return eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--only", default="")
    ap.add_argument("--previews", default="all")
    args = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    only = set(x for x in args.only.split(",") if x)
    mp = os.path.join(OUT, "manifest_built.json")
    rows = json.load(open(mp)) if os.path.exists(mp) else []
    have = {r["aoi"] for r in rows}
    splits = effective_splits(args.year)
    if not only:
        print("effective splits: " + " | ".join("%s=%d" % (s, len(v))
                                                for s, v in sorted(splits.items())), flush=True)
    for split, names in splits.items():
        for nm in sorted(set(names)):
            if only and nm not in only:
                continue
            if nm in have and not only:
                continue
            r = build_aoi(nm, args.year, split)
            if r:
                rows = [x for x in rows if x["aoi"] != nm] + [r]
                have.add(nm)
                print("  %-16s %-5s n=%4d %6.1f MB  dates=%3d targets=%2d gap50=%3.0f gap90=%3.0f"
                      % (r["aoi"], r["split"], r["n"], r["size_mb"], r["n_dates"],
                         r["n_targets"], r["gap_median"], r["gap_p90"]), flush=True)
            else:
                print("  %-16s MISSING" % nm, flush=True)
    json.dump(rows, open(mp, "w"), indent=1)

    # ---------------- statistics ----------------
    P = {k: [] for k in ["dt", "cld", "hr_cloud", "hr_clear", "clearfrac"]}
    per_split, per_aoi, ugaps, ngaps = {}, {}, [], []
    for r in rows:
        z = np.load(r["path"])
        for k in P:
            P[k].append(z[k].astype(np.float32).reshape(-1))
        dt = z["dt"].astype(np.float32)
        ngaps.append(np.diff(np.sort(dt, axis=1), axis=1).reshape(-1))
        cf = z["clearfrac"]
        for i in range(cf.shape[0]):
            ok = np.sort(np.abs(dt[i][cf[i] >= 0.5]))
            if len(ok) > 1:
                ugaps.append(np.diff(ok))
        s = per_split.setdefault(r["split"], {"n_samples": 0, "n_aois": 0})
        s["n_samples"] += r["n"]; s["n_aois"] += 1
        per_aoi[r["aoi"]] = {k: r[k] for k in
                             ["split", "n", "n_dates", "n_targets", "n_usable",
                              "gap_median", "gap_p90", "gap_max", "clear_mean",
                              "n_dropped_qc", "bright_frac"]}
    cat = lambda a: np.concatenate(a) if a else np.array([0.0])  # noqa: E731
    dt = cat(P["dt"]); cld = cat(P["cld"]) / 255.0; hcl = cat(P["hr_clear"])
    cfr = cat(P["clearfrac"]); ng = cat(ngaps); ug = cat(ugaps); adt = np.abs(dt)
    stats = {
        "n_aois": len(rows), "total_samples": int(sum(r["n"] for r in rows)),
        "scale": SCALE, "patch_hr": PATCH, "stride_hr": STRIDE,
        "patch_lr": PATCH // SCALE, "T": T, "target_clear_min": TARGET_CLEAR,
        "n_spatial_units_test": int(sum(9 for r in rows if r["split"] == "test")),
        "bands": ["B02", "B03", "B04", "B08"], "cloud_scl_classes": CLOUD_SCL,
        "per_split": per_split, "per_aoi": per_aoi,
        "temporal": {
            "abs_dt_median_days": float(np.median(adt)),
            "abs_dt_p90_days": float(np.percentile(adt, 90)),
            "abs_dt_max_days": float(adt.max()),
            "frame_gap_median_days": float(np.median(ng)),
            "frame_gap_p90_days": float(np.percentile(ng, 90)),
            "frame_frac_gap_gt10": float((ng > 10).mean()),
            "usable_gap_median_days": float(np.median(ug)),
            "usable_gap_p90_days": float(np.percentile(ug, 90)),
            "usable_gap_max_days": float(ug.max()),
            "usable_frac_gap_gt10": float((ug > 10).mean()),
            "usable_frac_gap_gt20": float((ug > 20).mean()),
        },
        "cloud": {
            "frame_clear_frac_mean": float(cfr.mean()),
            "frame_clear_frac_p10": float(np.percentile(cfr, 10)),
            "frame_frac_below_30pct_clear": float((cfr < 0.3).mean()),
            "lr_pixel_cloudfrac_mean": float(cld.mean()),
            "lr_pixel_frac_clear": float((cld < 0.01).mean()),
            "hr_target_clear_frac_mean": float(hcl.mean()),
            "hr_target_frac_clear_gt70": float((hcl > 0.7).mean()),
        },
    }
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    print(json.dumps({k: stats[k] for k in
                      ["n_aois", "total_samples", "n_spatial_units_test",
                       "per_split", "temporal", "cloud"]}, indent=1))

    if args.previews == "all":
        for r in rows:
            preview(r["aoi"], args.year)
    elif args.previews:
        for nm in args.previews.split(","):
            preview(nm.strip(), args.year)
    print("BUILD DONE")


if __name__ == "__main__":
    main()
