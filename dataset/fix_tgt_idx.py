"""Repair the `tgt_idx` field of collector output files.

Bug (fixed in collect_s2.py): tgt_idx stored indices into the *candidate* list, while the
saved arrays only contain the dates actually fetched.  Whenever a date was skipped
(footprint / transient failure) the indices shifted, and build_dataset then crashed with
IndexError or -- worse -- associated an HR image with the wrong input frames.

Recovery: the HR image of a target date must equal the 4x area-average of that date's LR
image (the collector's own alignment self-check).  So for every target we find the stored
LR entry that matches it, which gives the correct position unambiguously.

Usage: python fix_tgt_idx.py [--dry-run]
"""
import os
import glob
import argparse

import numpy as np
import cv2

RAW = "/home/czy/data/s2ds"


def area(x, size):
    return cv2.resize(x.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RAW, "*.npz")))
    print("files:", len(files))
    n_fixed = n_ok = n_skip = 0
    for p in files:
        z = np.load(p)
        if "hr10" not in z.files:
            print("  %-26s legacy format -> skip" % os.path.basename(p))
            n_skip += 1
            continue
        lr = z["lr40"]
        hr = z["hr10"]
        old = [int(i) for i in z["tgt_idx"]]
        n = lr.shape[0]
        new, corrs, margins = [], [], []
        for k in range(hr.shape[0]):
            a = area(hr[k][2], lr.shape[-1])
            av = (a - a.mean()).ravel()
            na = np.linalg.norm(av)
            c = []
            for j in range(n):
                b = lr[j][2].astype(np.float32)
                bv = (b - b.mean()).ravel()
                nb = np.linalg.norm(bv)
                c.append(float((av * bv).sum() / (na * nb)) if na * nb > 1e-9 else -1.0)
            c = np.array(c)
            order = np.argsort(c)[::-1]
            new.append(int(order[0]))
            corrs.append(float(c[order[0]]))
            margins.append(float(c[order[0]] - c[order[1]]))
        bad = [k for k in range(len(corrs)) if corrs[k] < 0.99 or margins[k] < 0.05]
        tag = "OK  " if not bad else "WARN"
        changed = sum(1 for a_, b_ in zip(old, new) if a_ != b_)
        print("  %-26s n_dates=%3d targets=%2d  min_corr=%.4f min_margin=%.4f  remapped=%d/%d %s"
              % (os.path.basename(p), n, len(new), min(corrs), min(margins), changed, len(new), tag))
        if bad:
            print("      unresolved targets:", bad)
        if not args.dry_run:
            d = {k: z[k] for k in z.files}
            d["tgt_idx"] = np.array(new, np.int32)
            tmp = p + ".tmp.npz"
            np.savez_compressed(tmp, **d)
            os.replace(tmp, p)
        n_fixed += (1 if changed else 0)
        n_ok += (0 if bad else 1)
    print("done: fixed=%d clean=%d skipped=%d%s"
          % (n_fixed, n_ok, n_skip, " (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
