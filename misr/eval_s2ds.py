"""Per-sample evaluation of a trained S2DS arm, with a selectable pixel mask.

Masks (all are subsets of the target's clear pixels):
  valid : every clear pixel of the target            (original endpoint)
  hard  : clear in the target BUT cloudy in the temporally nearest input frame
          -> this is where a single frame cannot help and temporal fusion is required
  easy  : clear in the target AND in the nearest frame

Usage: python eval_s2ds.py --run <run> --split test --mask hard [--ckpt best.pt]
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import cv2

sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
from misr.models import build_model                      # noqa: E402
from misr.train_s2ds import ARMS, psnr_masked            # noqa: E402
from s2ds_dataset import S2DS                            # noqa: E402


def mask_for(sample, kind):
    """Boolean HR-grid mask for the requested pixel stratum."""
    valid = (sample["valid"].numpy() > 0.5)
    if kind == "valid":
        return valid
    dt = sample["dt"].numpy()
    cld = sample["cld"].numpy()
    j0 = int(np.argmin(np.abs(dt)))
    h, w = sample["hr"].shape[-2:]
    c0 = cv2.resize(cld[j0], (w, h), interpolation=cv2.INTER_NEAREST)
    if kind == "hard":
        return valid & (c0 > 0.5)
    return valid & (c0 <= 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask", default="valid", choices=["valid", "hard", "easy"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--shuffle-frames", action="store_true",
                    help="evaluate with the frame axis permuted (must match training)")
    ap.add_argument("--permute-cld", type=int, default=0,
                    help="block size for the cloud-mask negative control (0 = off)")
    ap.add_argument("--flip-cld", type=float, default=0.0,
                    help="fraction of cloud-mask pixels randomly flipped, simulating "
                         "missed detections and false alarms of a real cloud detector")
    ap.add_argument("--tag", default="", help="suffix for the output json filename")
    args = ap.parse_args()
    rd = os.path.join("/mnt/e/论文2/runs_s2ds", args.run)
    ck_name = args.ckpt or ("best.pt" if os.path.exists(os.path.join(rd, "best.pt"))
                            else "last.pt")
    ck = torch.load(os.path.join(rd, ck_name), map_location="cpu", weights_only=False)
    a = ck["args"]
    use_dt, use_q = ARMS[a["arm"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cin=4, c=a["c"], scale=4, use_dt=use_dt, use_q=use_q,
                att_mode=a.get("att_mode", "frame"),
                base_resid=a.get("base_resid", False),
                head_init=a.get("head_init", 0.0),
                dt_mode=a.get("dt_mode", "add")).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    # the frame permutation / cloud scrambling must be identical to what the run was
    # trained with, otherwise the evaluation measures a distribution shift, not the factor
    shuf = args.shuffle_frames or bool(a.get("shuffle_frames", False))
    pcld = args.permute_cld or int(a.get("permute_cld", 0))
    fld = args.flip_cld or float(a.get("flip_cld", 0.0))
    ds = S2DS(args.split, shuffle=shuf, seed=int(a.get("seed", 0)),
              permute_cld=pcld, flip_cld=fld)
    out, by_aoi = {}, {}
    with torch.no_grad():
        for j in range(len(ds)):
            s = ds[j]
            p = model(lr=s["lr"][None].to(dev), q=s["q"][None].to(dev),
                      cld=s["cld"][None].to(dev), dt=s["dt"][None].to(dev))[0].cpu().numpy()
            m = mask_for(s, args.mask).astype(np.float32)
            ps = psnr_masked(p, s["hr"].numpy(), m)
            out[s["scene"]] = float(ps)
            by_aoi.setdefault(s["aoi"], []).append(float(ps))
    sf = ("_%s" % args.tag) if args.tag else ""      # never clobber the main endpoint
    json.dump(out, open(os.path.join(rd, "scene_psnr_%s_%s%s.json"
                                     % (args.split, args.mask, sf)), "w"), indent=1)
    if args.mask == "valid":
        json.dump(out, open(os.path.join(rd, "scene_psnr_%s%s.json"
                                         % (args.split, sf)), "w"), indent=1)
    v = np.array(list(out.values()))
    print("%-22s split=%-5s mask=%-5s ckpt=%-11s n=%3d mean=%.4f median=%.4f sd=%.4f"
          % (args.run, args.split, args.mask, ck_name, len(v), np.nanmean(v),
             np.nanmedian(v), np.nanstd(v)))
    for k, val in sorted(by_aoi.items()):
        print("    %-16s n=%3d mean=%.4f" % (k, len(val), float(np.nanmean(val))))


if __name__ == "__main__":
    main()
