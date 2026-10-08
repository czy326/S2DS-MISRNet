"""Re-fetch the HR targets of the collected AOIs with the CORRECT selection rule.

Bug being repaired: collect_s2.py compared the *cloud* fraction against a *clear*
threshold, so the 10 target dates of every AOI ended up being the cloudiest ones
(measured 0.89-1.00 cloud).  Pass-1 data (all dates at 40 m + SCL) is fine and already on
disk, so only the pass-2 full-resolution reads have to be redone: 10 dates per AOI.

Usage: python repair_targets.py [--threads 3] [--only aoi1,aoi2]
"""
import os
import sys
import json
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, "/mnt/e/论文2/dataset")
from collect_s2 import (cog_base, tile_origin, fetch_hr, parse_item,  # noqa: E402
                        TARGET_CLEAR, K_TARGETS, CLOUD_SCL)

RAW = "/home/czy/data/s2ds"
CACHE = "/home/czy/data/aoi_recon"
MAN = "/mnt/e/论文2/dataset/manifest.json"


def pick_idx(n, k):
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k)))


def sat_map(aoi, year):
    p = os.path.join(CACHE, "%s_%d.json" % (aoi, year))
    out = {}
    if os.path.exists(p):
        for f in json.load(open(p)).get("features", []):
            r = parse_item(f)
            if r:
                out.setdefault(r[2], r[0])          # date -> satellite
    return out


def resolve_meta(aoi, year, win=320):
    """(lon, lat, tile, sat_map) recovered without relying on manifest.json.

    manifest.json was found to be incomplete: the two collection processes did a
    read-modify-write on it concurrently and clobbered each other's entries.  Everything
    needed here is recoverable: coordinates from aois.json, the satellite from the cached
    STAC search, and the tile by trying candidates in order of date count until one
    contains the AOI (same rule the collector uses).
    """
    import json as _json
    from pyproj import Transformer
    from collect_s2 import OutOfFootprint
    aois = _json.load(open("/mnt/e/论文2/dataset/aois.json"))
    lon, lat = aois[aoi]["lon"], aois[aoi]["lat"]
    cache = os.path.join(CACHE, "%s_%d.json" % (aoi, year))
    if not os.path.exists(cache):
        return None
    feats = _json.load(open(cache)).get("features", [])
    tiles, sm = {}, {}
    for f in feats:
        r = parse_item(f)
        if not r:
            continue
        sat, tile, date = r
        tiles[tile] = tiles.get(tile, 0) + 1
        sm.setdefault(date, sat)
    for tile in sorted(tiles, key=lambda t: -tiles[t]):
        epsg = 32600 + int(tile[:2])
        tr = Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True)
        x, y = tr.transform(lon, lat)
        try:
            r10, c10 = tile_origin(cog_base(sm[next(iter(sm))], tile, next(iter(sm))),
                                   x, y, win, epsg)
        except OutOfFootprint:
            continue
        except Exception:                                        # noqa: BLE001
            continue
        return dict(aoi=aoi, year=year, lon=lon, lat=lat, tile=tile, win=win,
                    sat=sm, r10=r10, c10=c10)
    return None


def process(path, man, threads):
    key = os.path.basename(path)[:-len(".npz")]
    aoi = key.rsplit("_", 1)[0]
    year = int(key.rsplit("_", 1)[1])
    meta = man.get(key) or resolve_meta(aoi, year)
    if not meta:
        return "%s: could not resolve metadata -> skip" % key
    z = np.load(path)
    dates = [str(d) for d in z["dates"]]
    clear = 1.0 - z["cloudfrac"].astype(np.float32)
    order = list(np.argsort(dates))
    cand = [i for i in order if clear[i] >= TARGET_CLEAR]
    if len(cand) < 3:
        cand = [i for i in order if clear[i] >= 0.35] or order
    tgt = [cand[j] for j in pick_idx(len(cand), K_TARGETS)]
    sm = meta.get("sat") or sat_map(aoi, year)
    missing = [dates[i] for i in tgt if dates[i] not in sm]
    if missing:
        return "%s: missing satellite info for %d dates -> skip" % (aoi, len(missing))

    if "r10" in meta:
        r10, c10, tile = meta["r10"], meta["c10"], meta["tile"]
    else:
        from pyproj import Transformer
        epsg = 32600 + int(meta["tile"][:2])
        tr = Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True)
        x, y = tr.transform(meta["lon"], meta["lat"])
        r10, c10 = tile_origin(cog_base(sm[dates[tgt[0]]], meta["tile"], dates[tgt[0]]),
                               x, y, int(meta["win"]), epsg)
        tile = meta["tile"]
    win = int(meta["win"])

    hr = [None] * len(tgt)
    errs = []

    def job(k):
        i = tgt[k]
        a = fetch_hr(sm[dates[i]], tile, dates[i], r10, c10, win)
        print("    %s tgt%d %s ok" % (aoi, k, dates[i]), flush=True)
        return k, a

    print("  %s: fetching %d targets (clear %.3f..%.3f)"
          % (aoi, len(tgt), float(clear[np.array(tgt)].min()), float(clear[np.array(tgt)].max())),
          flush=True)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(job, k) for k in range(len(tgt))]
        for fu in as_completed(futs):
            k, arr = fu.result()
            hr[k] = arr
    if any(h is None for h in hr):
        return "%s: %d HR reads failed -> skip" % (aoi, sum(1 for h in hr if h is None))

    out = {k: z[k] for k in z.files}
    out["hr10"] = np.stack(hr).astype(np.uint16)
    out["tgt_idx"] = np.array(tgt, np.int32)
    out["tgt_clear"] = clear[np.array(tgt)].astype(np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return ("%s: %d targets redone  clear min/med = %.3f / %.3f"
            % (aoi, len(tgt), float(clear[np.array(tgt)].min()),
               float(np.median(clear[np.array(tgt)]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()
    man = json.load(open(MAN))
    files = sorted(glob.glob(os.path.join(RAW, "*.npz")))
    only = set(x for x in args.only.split(",") if x)
    if only:
        files = [f for f in files if os.path.basename(f)[:-9] in only]
    if args.procs > 1:
        files = [f for i, f in enumerate(files) if i % args.procs == args.shard]
    for p in files:
        try:
            print(process(p, man, args.threads), flush=True)
        except Exception as e:                                    # noqa: BLE001
            print("%s: FAILED %s %s" % (os.path.basename(p), type(e).__name__, str(e)[:120]), flush=True)
    print("REPAIR DONE shard=%d/%d files=%d" % (args.shard, args.procs, len(files)))


if __name__ == "__main__":
    main()
