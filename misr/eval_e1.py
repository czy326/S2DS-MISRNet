"""Per-sample evaluation for the E1 q-route ablation arms (B / B1 / B2).

Same masks and same output layout as misr/eval_s2ds.py; the only difference is that the
model is rebuilt from misr/models_e1.py, because `arm` is a route combination (B/B1/B2)
and not a (use_dt, use_q) pair.
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
from misr.build_any import build_from_spec                # noqa: E402
from s2ds_dataset import S2DS                             # noqa: E402


def psnr_masked(pred, gt, mask, peak=1.0):
    m = mask > 0.5
    if int(m.sum()) < 32:
        return float("nan")
    d = (pred - gt)[:, m]
    mse = float((d ** 2).mean())
    return 10 * np.log10(peak ** 2 / (mse + 1e-12))


def mask_for(sample, kind):
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
    ap.add_argument("--tag", default="", help="suffix for the output json filename")
    args = ap.parse_args()
    rd = os.path.join("/mnt/e/论文2/runs_s2ds", args.run)
    ck_name = args.ckpt or ("best.pt" if os.path.exists(os.path.join(rd, "best.pt"))
                            else "last.pt")
    ck = torch.load(os.path.join(rd, ck_name), map_location="cpu", weights_only=False)
    a = ck["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_from_spec(a, cin=4, scale=4).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    ds = S2DS(args.split)
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
    sf = ("_%s" % args.tag) if args.tag else ""
    json.dump(out, open(os.path.join(rd, "scene_psnr_%s_%s%s.json"
                                     % (args.split, args.mask, sf)), "w"), indent=1)
    v = np.array(list(out.values()))
    print("%-22s split=%-5s mask=%-5s ckpt=%-11s n=%3d mean=%.4f sd=%.4f"
          % (args.run, args.split, args.mask, ck_name, len(v), np.nanmean(v),
             np.nanstd(v)))
    for k, val in sorted(by_aoi.items()):
        print("    %-16s n=%3d mean=%.4f" % (k, len(val), float(np.nanmean(val))))


if __name__ == "__main__":
    main()
