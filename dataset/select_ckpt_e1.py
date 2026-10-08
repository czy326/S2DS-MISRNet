"""Validation-best checkpoint selection for the E1 arms (B / B1 / B2).

Same job as select_ckpt.py, but the arm is a q-route combination rather than a
(use_dt, use_q) pair, so the model is rebuilt from misr.models_e1.
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
from misr.build_any import build_from_spec                # noqa: E402
from s2ds_dataset import S2DS                             # noqa: E402

RUNS = "/mnt/e/论文2/runs_s2ds"


def psnr_masked(pred, gt, mask, peak=1.0):
    m = mask > 0.5
    if int(m.sum()) < 32:
        return float("nan")
    d = (pred - gt)[:, m]
    return float(10 * np.log10(peak ** 2 / (float((d ** 2).mean()) + 1e-12)))


def score(ckpt_path, ds, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_from_spec(a, cin=4, scale=4).to(dev)
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
    args = ap.parse_args()
    rd = os.path.join(RUNS, args.run)
    cks = sorted(glob.glob(os.path.join(rd, "ckpt_*.pt")))
    if not cks:
        print("%s: no ckpt_*.pt -> keep last.pt" % args.run)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = S2DS(args.split)
    scores = []
    for cp in cks:
        s = score(cp, ds, dev)
        scores.append((s, os.path.basename(cp)))
        print("  %-22s %s val=%.4f" % (args.run, os.path.basename(cp), s), flush=True)
    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    torch.save(torch.load(os.path.join(rd, best_name), map_location="cpu",
                          weights_only=False), os.path.join(rd, "best.pt"))
    last_name = os.path.basename(cks[-1])
    last_score = [s for s, n in scores if n == last_name]
    json.dump({"run": args.run, "best_ckpt": best_name, "best_val": best_score,
               "last_ckpt": last_name,
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
