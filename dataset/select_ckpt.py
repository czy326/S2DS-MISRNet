"""Pick a validation-best checkpoint for a run (removes the +-0.5 dB last.pt noise).

Each arm's validation curve oscillates by ~0.5 dB with no trend, and evaluating `last.pt`
therefore lands on an arbitrary local dip.  This script scores every `ckpt_*.pt` on the
validation split (validity mask) and copies the best one to `best.pt`.

Usage: python select_ckpt.py --run s2dsL_armA_s2026 [--split val] [--limit 0]
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
from misr.models import build_model                      # noqa: E402
from misr.train_s2ds import ARMS, psnr_masked            # noqa: E402
from s2ds_dataset import S2DS                            # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"


def score(ckpt_path, ds, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    ud, uq = ARMS[a["arm"]]
    model = build_model(cin=4, c=a["c"], scale=4, use_dt=ud, use_q=uq,
                        att_mode=a.get("att_mode", "frame"),
                        base_resid=a.get("base_resid", False),
                        head_init=a.get("head_init", 0.0),
                        dt_mode=a.get("dt_mode", "add")).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    out = []
    with torch.no_grad():
        for j in range(len(ds)):
            s = ds[j]
            p = model(lr=s["lr"][None].to(dev), q=s["q"][None].to(dev),
                      cld=s["cld"][None].to(dev), dt=s["dt"][None].to(dev))[0].cpu().numpy()
            out.append(psnr_masked(p, s["hr"].numpy(), s["valid"].numpy()))
    return float(np.nanmean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pick", default="mean", choices=["mean", "last"])
    args = ap.parse_args()
    rd = os.path.join(RUNS, args.run)
    cks = sorted(glob.glob(os.path.join(rd, "ckpt_*.pt")))
    if not cks:
        print("%s: no ckpt_*.pt -> keep last.pt" % args.run)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.load(cks[0], map_location="cpu", weights_only=False)["args"]
    use_dt, use_q = ARMS[a["arm"]]
    ds = S2DS(args.split)
    scores = []
    for cp in cks:
        s = score(cp, ds, dev)
        scores.append((s, os.path.basename(cp)))
        print("  %-22s %s val=%.4f" % (args.run, os.path.basename(cp), s), flush=True)
    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    src = os.path.join(rd, best_name)
    torch.save(torch.load(src, map_location="cpu", weights_only=False),
               os.path.join(rd, "best.pt"))
    last_score = [s for s, n in scores if n == os.path.basename(cks[-1])]
    json.dump({"run": args.run, "best_ckpt": best_name, "best_val": best_score,
               "last_ckpt": os.path.basename(cks[-1]),
               "last_val": (last_score[0] if last_score else None),
               "spread": float(scores[0][0] - scores[-1][0]),
               "all": [{"ckpt": n, "val": s} for s, n in scores]},
              open(os.path.join(rd, "ckpt_selection.json"), "w"), indent=1)
    print("%s: best=%s val=%.4f | last val=%.4f | ckpt spread=%.3f dB"
          % (args.run, best_name, best_score,
             float(last_score[0]) if last_score else float("nan"),
             float(scores[0][0] - scores[-1][0])))


if __name__ == "__main__":
    main()
