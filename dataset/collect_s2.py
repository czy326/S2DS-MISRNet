"""Collect Sentinel-2 L2A time-series windows for AOIs from the public AWS COG mirror.

No account needed.  Two passes per AOI (this is what makes it fast):

  pass 1  every acquisition of the year, read *cheaply* at LR scale:
            4 bands (B02/B03/B04/B08) read as a 4x-decimated window  (~40 m)
            + SCL read at its native 20 m  -> per-pixel cloud / per-frame clear fraction
  pass 2  only the K target dates (chosen from pass-1 clear fractions, spread over the
            year) are re-read at full 10 m resolution -> the HR reference

Why: a 10 m window read costs ~11 s (the COG block has to be pulled), while the same
window decimated to 40 m costs ~0.7 s (16x cheaper).  Input frames only need LR, so we
pay the expensive read exactly K times instead of 150 times.

Bucket layout (FLAT, no R10m/ sub-dir):
  sentinel-s2-l2a-cogs/<zone>/<latband>/<square>/<year>/<month>/<SAT>_<tile>_<yyyymmdd>_0_L2A/<BAND>.tif

Output: /home/czy/data/s2ds/<aoi>_<year>.npz
  lr40     (n,4,h,w)  uint16   4x-decimated (40 m) bands, all dates
  scl20    (n,2h,2w)  uint8    native 20 m SCL, all dates
  dates    (n,)       '<yyyymmdd>'
  cloudfrac(n,)       float32  SCL cloud fraction (classes 3/8/9/10) over the window
  stac_cloud (n,)     float32  scene-level cloud cover reported by STAC
  hr10     (k,4,4h,4w) uint16  full-resolution 10 m bands for the target dates only
  tgt_idx  (k,)       int32    index into `dates`
"""
import os
import re
import sys
import json
import time
import argparse
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from pyproj import Transformer  # noqa: E402

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"
STAC = "https://catalogue.dataspace.copernicus.eu/stac/search"
BANDS = ["B02", "B03", "B04", "B08"]
CLOUD_SCL = [3, 8, 9, 10]
UA = {"User-Agent": "Mozilla/5.0"}
CACHE = "/home/czy/data/aoi_recon"
PRINT_LOCK = threading.Lock()

TARGET_CLEAR = 0.5      # target dates must have this clear fraction (SCL) in the window
K_TARGETS = 10          # target dates kept per AOI


def log(msg):
    with PRINT_LOCK:
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _get(url, timeout=240, tries=4):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError("GET fail %s: %s" % (url[:90], str(last)[:200]))


def stac_items(bbox, year, cache_key):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "%s_%d.json" % (cache_key, year))
    if os.path.exists(path):
        try:
            return json.load(open(path))["features"]
        except Exception:
            pass
    url = ("%s?collections=sentinel-2-l2a&bbox=%.4f,%.4f,%.4f,%.4f"
           "&datetime=%d-01-01T00:00:00Z/%d-12-31T23:59:59Z"
           "&limit=1000&fields=id,properties.datetime,properties.eo:cloud_cover"
           % (STAC, bbox[0], bbox[1], bbox[2], bbox[3], year, year))
    feats = []
    while url:
        d = json.loads(_get(url).decode("utf-8", "ignore"))
        feats.extend(d.get("features", []))
        nxt = None
        for lk in d.get("links", []):
            if lk.get("rel") == "next":
                nxt = lk.get("href")
        url = nxt if (nxt and d.get("features")) else None
    json.dump({"features": feats}, open(path, "w"))
    return feats


def parse_item(f):
    m = re.match(r"^(S2[AB])_MSIL2A_(\d{8})T.*?_(T[0-9A-Z]{5})_", f.get("id", ""))
    return (m.group(1), m.group(3)[1:], m.group(2)) if m else None


def cog_base(sat, tile, date):
    return ("%s/%s/%s/%s/%s/%d/%s_%s_%s_0_L2A"
            % (BUCKET, tile[:2], tile[2], tile[3:], date[:4], int(date[4:6]), sat, tile, date))


class OutOfFootprint(Exception):
    """The requested window is not fully inside this product -> drop the date."""


def tile_origin(base, x, y, win, epsg):
    """Aligned window origin (in 10 m pixels) for this tile, computed ONCE per AOI.

    All dates of an MGRS tile share the same pixel grid and geotransform (verified
    directly: reading a window from different dates gives bit-identical rasters), so the
    origin is computed once.  It is snapped to a multiple of 4 so that GDAL's decimated
    read (which picks the 4x overview) lands exactly on overview pixel boundaries --
    otherwise LR and HR would be offset by a fixed sub-pixel amount, which a learned
    model could partially absorb and thereby contaminate the comparison.
    """
    url = "/vsicurl/" + base + "/B02.tif"
    with rasterio.open(url) as s:
        if epsg is not None and s.crs is not None and s.crs.to_epsg() != epsg:
            raise OutOfFootprint("crs %s != %s" % (s.crs.to_epsg(), epsg))
        row, col = s.index(x, y)
        r0, c0 = int(row) - win // 2, int(col) - win // 2
        r0 -= r0 % 4
        c0 -= c0 % 4
        if r0 < 0 or c0 < 0 or r0 + win > s.height or c0 + win > s.width:
            raise OutOfFootprint("aoi window outside %dx%d" % (s.width, s.height))
        return r0, c0


def read_band(base, band, r10, c10, win, out_shape, scale=1, tries=4):
    """Read a window given by the aligned 10 m origin; `scale` = band grid factor.

    The window is validated against the footprint instead of being clamped -- clamping
    was a real bug: dates whose footprint does not contain the AOI silently returned a
    corner of the scene, i.e. a completely different location.
    """
    url = "/vsicurl/" + base + "/%s.tif" % band
    r, c, w = r10 // scale, c10 // scale, win // scale
    last = None
    for k in range(tries):
        try:
            with rasterio.open(url) as s:
                if r < 0 or c < 0 or r + w > s.height or c + w > s.width:
                    raise OutOfFootprint("window outside %dx%d" % (s.width, s.height))
                return s.read(1, window=Window(c, r, w, w), out_shape=out_shape)
        except OutOfFootprint:
            raise
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError("read fail %s %s: %s" % (band, os.path.basename(base), str(last)[:70]))


def fetch_lr(sat, tile, date, r10, c10, win10, inner=5):
    """Pass 1: cheap LR read for one date. Returns (lr40 (4,h,w) u16, scl20 (2h,2w) u8)."""
    base = cog_base(sat, tile, date)
    h = win10 // 4
    res = {}

    def job(b, oscale, scale):
        res[b] = read_band(base, b, r10, c10, win10, oscale, scale)
    with ThreadPoolExecutor(max_workers=inner) as ex:
        futs = [ex.submit(job, b, (h, h), 1) for b in BANDS]
        futs.append(ex.submit(job, "SCL", None, 2))
        for fu in as_completed(futs):
            fu.result()
    lr = np.stack([res[b] for b in BANDS])
    if lr.shape != (len(BANDS), h, h):
        raise OutOfFootprint("bad lr shape %s" % (lr.shape,))
    if res["SCL"].shape != (2 * h, 2 * h):
        raise OutOfFootprint("bad scl shape %s" % (res["SCL"].shape,))
    return lr, res["SCL"]


def fetch_hr(sat, tile, date, r10, c10, win10, inner=4):
    """Pass 2: full-resolution read for a target date."""
    base = cog_base(sat, tile, date)
    res = {}

    def job(b):
        res[b] = read_band(base, b, r10, c10, win10, (win10, win10), 1)
    with ThreadPoolExecutor(max_workers=inner) as ex:
        futs = [ex.submit(job, b) for b in BANDS]
        for fu in as_completed(futs):
            fu.result()
    return np.stack([res[b] for b in BANDS])


def pick_idx(n, k):
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k)))


def collect_aoi(name, lon, lat, year, win=320, threads=5, span=0.10, k_targets=K_TARGETS):
    bbox = (lon - span, lat - span, lon + span, lat + span)
    feats = stac_items(bbox, year, name)
    seen, rows = set(), []
    for f in feats:
        p = parse_item(f)
        if not p:
            continue
        sat, tile, date = p
        if (tile, date) in seen:
            continue
        seen.add((tile, date))
        rows.append((date, tile, sat, float(f["properties"].get("eo:cloud_cover", -1))))
    rows.sort()
    if not rows:
        log("%s: no STAC items" % name)
        return None
    tiles = {}
    for d, t, s, c in rows:
        tiles.setdefault(t, []).append((d, s, c))
    # A candidate tile may NOT contain the AOI (near tile boundaries the STAC search
    # returns several tiles).  Try them in order of date count and keep the first one
    # whose footprint actually contains the window; otherwise the whole AOI used to fail.
    epsg = None
    tile, cand, r10, c10 = None, None, None, None
    for t in sorted(tiles, key=lambda k: -len(tiles[k])):
        c_ = sorted(tiles[t])
        e = 32600 + int(t[:2])
        trr = Transformer.from_crs("EPSG:4326", "EPSG:%d" % e, always_xy=True)
        xx, yy = trr.transform(lon, lat)
        try:
            rr, cc = tile_origin(cog_base(c_[0][1], t, c_[0][0]), xx, yy, win, e)
        except Exception as ex:
            log("%s: tile %s rejected (%s)" % (name, t, str(ex)[:70]))
            continue
        tile, cand, r10, c10, epsg, x, y = t, c_, rr, cc, e, xx, yy
        break
    if tile is None:
        log("%s: no candidate tile contains the AOI -> skip AOI" % name)
        return None
    log("%s: %d dates on tile %s" % (name, len(cand), tile))

    log("%s: aligned origin row=%d col=%d (mod4=%d,%d)" % (name, r10, c10, r10 % 4, c10 % 4))

    t0 = time.time()
    out = [None] * len(cand)
    done = [0]

    def one(i):
        d, sat, cc = cand[i]
        try:
            lr, scl = fetch_lr(sat, tile, d, r10, c10, win)
            frac = float(np.isin(scl, CLOUD_SCL).mean())
            return i, dict(date=d, sat=sat, lr=lr, scl=scl, cloudfrac=frac, stac_cloud=cc)
        except OutOfFootprint as e:
            log("  skip %s %s (%s)" % (d, sat, str(e)[:50].replace("\n", " ")))
            return i, None
        except Exception as e:
            log("  ERR %s %s: %s" % (d, sat, str(e)[:220].replace("\n", " ")))
            return i, None

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(one, i) for i in range(len(cand))]
        for fu in as_completed(futs):
            i, r = fu.result()
            out[i] = r
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(cand):
                ok = sum(1 for o in out if o)
                log("%s: pass1 %d/%d (ok=%d, %.0fs)" % (name, done[0], len(cand), ok, time.time() - t0))
    good = [(i, o) for i, o in enumerate(out) if o]
    # transient cross-border throttling / connection resets hit a few % of dates.
    # Retry them with low concurrency instead of silently dropping them.
    failed = [i for i, o in enumerate(out) if o is None]
    if failed:
        log("%s: pass1 retry for %d failed dates" % (name, len(failed)))
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(one, i): i for i in failed}
            for fu in as_completed(futs):
                i, r = fu.result()
                if r:
                    out[i] = r
        good = [(i, o) for i, o in enumerate(out) if o]
        log("%s: after retry ok=%d/%d" % (name, len(good), len(cand)))
    if len(good) < 5:
        log("%s: too few frames" % name)
        return None

    # target dates: clear enough, spread over the year.
    # NOTE `cloudfrac` is the CLOUD fraction (bigger = cloudier) -> compare its complement.
    usable = [(i, o) for i, o in good if (1.0 - o["cloudfrac"]) >= TARGET_CLEAR]
    usable.sort(key=lambda t: t[1]["date"])
    if len(usable) < 3:
        usable = sorted([(i, o) for i, o in good if (1.0 - o["cloudfrac"]) >= 0.35],
                        key=lambda t: t[1]["date"]) or sorted(good, key=lambda t: t[1]["date"])
    tgt = [usable[j] for j in pick_idx(len(usable), k_targets)]
    log("%s: %d target candidates -> fetching %d at 10 m" % (name, len(usable), len(tgt)))

    hr, tgt_idx = [], []
    with ThreadPoolExecutor(max_workers=min(4, threads)) as ex:
        futs = {}
        for i, o in tgt:
            futs[ex.submit(fetch_hr, o["sat"], tile, o["date"], r10, c10, win)] = i
        for fu in as_completed(futs):
            i = futs[fu]
            try:
                hr.append(fu.result())
                tgt_idx.append(i)
            except Exception as e:
                log("  ERR HR %s: %s" % (cand[i][0], str(e)[:70]))
    if not hr:
        log("%s: no HR fetched" % name)
        return None
    row_of = {int(i): r for r, (i, o) in enumerate(good)}
    order = np.argsort(np.array(tgt_idx))
    # tgt_idx must be POSITIONS inside the saved arrays (which only contain the dates
    # that were actually fetched), not indices into the original candidate list -- the
    # two differ whenever a date was skipped (footprint / transient failure).
    tgt_idx = np.array([row_of[int(i)] for i in np.array(tgt_idx, np.int32)[order]], np.int32)
    hr = np.stack(hr)[order]

    dst = "/home/czy/data/s2ds/%s_%d.npz" % (name, year)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ds = [o["date"] for _, o in good]
    lr40_stack = np.stack([o["lr"] for _, o in good]).astype(np.uint16)
    # --- self check: the pass-1 LR (4x overview) must equal an area-average of the
    #     pass-2 HR of the same date.  A low correlation means the decimated read is not
    #     aligned with the base grid, i.e. LR and HR are not on a common grid.
    checks = []
    for k, i in enumerate(tgt_idx):
        h = hr[k]
        a = cv2.resize(h[2].astype(np.float32), (win // 4, win // 4),
                       interpolation=cv2.INTER_AREA)
        b = lr40_stack[row_of[int(i)]][2].astype(np.float32)
        c = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        rel = float(np.abs(a - b).mean() / max(a.mean(), 1e-6))
        checks.append((int(i), c, rel))
    cc = float(np.mean([c for _, c, _ in checks]))
    rr = float(np.mean([r for _, _, r in checks]))
    log("%s: LR/HR alignment check: corr=%.4f rel_err=%.4f (n=%d dates)"
        % (name, cc, rr, len(checks)))
    np.savez_compressed(
        dst,
        lr40=lr40_stack,
        scl20=np.stack([o["scl"] for _, o in good]).astype(np.uint8),
        dates=np.array(ds),
        cloudfrac=np.array([o["cloudfrac"] for _, o in good], np.float32),
        stac_cloud=np.array([o["stac_cloud"] for _, o in good], np.float32),
        hr10=hr.astype(np.uint16), tgt_idx=tgt_idx,
        lr_check=np.array([[c[0], c[1], c[2]] for c in checks], np.float32),
    )
    meta = dict(aoi=name, lon=lon, lat=lat, year=year, tile=tile, win=win,
                n_dates=len(good), n_targets=len(tgt_idx),
                lr_check_corr=cc, lr_check_rel_err=rr,
                clear_frac_mean=float(np.mean([o["cloudfrac"] for _, o in good])),
                size_mb=round(os.path.getsize(dst) / 1e6, 2))
    log("%s: SAVED %s  dates=%d targets=%d %.1f MB (%.0fs)"
        % (name, dst, len(good), len(tgt_idx), meta["size_mb"], time.time() - t0))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="all")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--win", type=int, default=320)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--k", type=int, default=K_TARGETS)
    args = ap.parse_args()
    aois = json.load(open("/mnt/e/论文2/dataset/aois.json"))
    names = list(aois) if args.aoi == "all" else args.aoi.split(",")
    os.makedirs("/home/czy/data/s2ds", exist_ok=True)
    mp = "/mnt/e/论文2/dataset/manifest.json"
    manifest = json.load(open(mp)) if os.path.exists(mp) else {}
    for nm in names:
        try:
            meta = collect_aoi(nm, aois[nm]["lon"], aois[nm]["lat"], args.year,
                               win=args.win, threads=args.threads, k_targets=args.k)
        except Exception as e:
            log("%s: FAILED %s %s" % (nm, type(e).__name__, str(e)[:120]))
            continue
        if meta:
            manifest["%s_%d" % (nm, args.year)] = meta
            json.dump(manifest, open(mp, "w"), indent=1)
    log("ALL DONE aois=%d" % len(names))


if __name__ == "__main__":
    main()
