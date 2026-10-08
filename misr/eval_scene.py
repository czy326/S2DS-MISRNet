"""Evaluate a trained MISR checkpoint scene-by-scene (for paired tests)."""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, "/mnt/e/论文2")
from misr.data import ProbavDataset, MuS2Dataset
from misr.models import build_model
from misr.train import ARMS, masked_psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir name under runs_misr/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--crop", type=int, default=192)
    args = ap.parse_args()

    rd = os.path.join("/mnt/e/论文2/runs_misr", args.run)
    ck = torch.load(os.path.join(rd, "last.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]; stats = ck["stats"]
    use_dt, use_q = ARMS[a["arm"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a["data"] == "probav":
        ds = ProbavDataset(band=a["band"], split=args.split, T=args.T, stats=stats, seed=a["seed"])
        cin = 1
    else:
        ds = MuS2Dataset(split=args.split, T=args.T, crop=args.crop, stats=stats, seed=a["seed"])
        cin = 3
    model = build_model(cin=cin, c=a["c"], scale=3, use_dt=use_dt, use_q=use_q).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    out = {}
    with torch.no_grad():
        for j in range(len(ds)):
            sb = ds[j]
            b = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in sb.items()}
            kw = {"lr": b["lr"]}
            if use_q:
                kw["q"] = b["q"]; kw["cld"] = b["cld"] if "cld" in b else b["qm"]
            if use_dt:
                kw["dt"] = b["dt"] if "dt" in b else torch.zeros(1, b["lr"].shape[1], device=dev)
            pv = model(**kw)
            hs_, hm_ = stats["hr_std"], stats["hr_mean"]
            pv = (pv * hs_ + hm_).cpu(); hrv = (b["hr"] * hs_ + hm_).cpu()
            mkv = b["sm"].cpu() if "sm" in b else None
            key = sb["scene"] if not torch.is_tensor(sb.get("scene", None)) else str(j)
            out[str(key)] = float(masked_psnr(pv, hrv, mkv))
    p = os.path.join(rd, "scene_psnr_%s.json" % args.split)
    json.dump(out, open(p, "w"), indent=1)
    v = np.array(list(out.values()))
    print("%s | %s n=%d mean=%.4f sd=%.4f min=%.3f max=%.3f" % (args.run, args.split, len(v), v.mean(), v.std(), v.min(), v.max()))
    print("saved", p)


if __name__ == "__main__":
    main()
