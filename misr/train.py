"""MISR trainer for the 2x2 factor screening (A plain / B +q / C +dt / D +dt+q)."""
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/e/论文2")
from misr.data import ProbavDataset, MuS2Dataset
from misr.models import build_model, n_params

ARMS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}


def get_stats(ds, key, k=40):
    vals = []
    for i in range(min(k, len(ds))):
        s = ds[i]
        vals.append(s[key].numpy().ravel()[::37])
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std() + 1e-6)


def masked_psnr(pred, hr, mask=None):
    if mask is not None:
        m = mask.bool().expand_as(pred)
        d = (pred - hr)[m]
    else:
        d = (pred - hr).reshape(-1)
    if d.numel() < 10:
        return float("nan")
    mse = float((d ** 2).mean())
    rng = float(hr.max() - hr.min() + 1e-6)
    return 10 * np.log10(rng ** 2 / (mse + 1e-12))


def build_dataset(args, split, stats=None):
    if args.data == "probav":
        return ProbavDataset(band=args.band, split=split, T=args.T, stats=stats, seed=args.seed)
    return MuS2Dataset(split=split, T=args.T, crop=args.crop, stats=stats, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="probav", choices=["probav", "mus2"])
    ap.add_argument("--band", default="RED")
    ap.add_argument("--arm", default="A", choices=list(ARMS))
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-train", type=int, default=0)
    args = ap.parse_args()

    use_dt, use_q = ARMS[args.arm]
    name = args.out or ("%s_%s_arm%s_seed%d" % (args.data, args.band, args.arm, args.seed))
    outdir = "/mnt/e/论文2/runs_misr/%s" % name
    os.makedirs(outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr = build_dataset(args, "train")
    va = build_dataset(args, "val")
    if args.max_train:
        tr.scenes = tr.scenes[:args.max_train]
    key = "hr" if args.data == "probav" else "lr"
    lm, ls = get_stats(tr, key)
    hm, hs = (lm, ls) if args.data != "probav" else get_stats(tr, "hr")
    stats = {"lr_mean": lm, "lr_std": ls, "hr_mean": hm, "hr_std": hs}
    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    tr.stats = stats; va.stats = stats

    cin = 1 if args.data == "probav" else 3
    model = build_model(cin=cin, c=args.c, scale=3, use_dt=use_dt, use_q=use_q).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    print("[%s] params=%d train=%d val=%d dev=%s" % (name, n_params(model), len(tr), len(va), dev), flush=True)
    log = open(os.path.join(outdir, "log.csv"), "w")
    log.write("iter,loss,val_psnr,sec\n")

    def batch_to(d, b):
        kwargs = {"lr": b["lr"].to(dev)}
        if use_q:
            kwargs["q"] = b["q"].to(dev)
            kwargs["cld"] = (b["cld"] if "cld" in b else b.get("qm")).to(dev) if ("cld" in b or "qm" in b) else None
        if use_dt:
            kwargs["dt"] = b.get("dt", torch.zeros(b["lr"].shape[0], b["lr"].shape[1])).to(dev)
        return kwargs

    it = 0
    t0 = time.time()
    while it < args.iters:
        for b in trl:
            kwargs = batch_to(None, b)
            hr = b["hr"].to(dev)
            mask = b.get("sm")
            mask = mask.to(dev) if mask is not None else None
            pred = model(**kwargs)
            if mask is not None:
                loss = ((pred - hr).abs() * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)
            else:
                loss = (pred - hr).abs().mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            it += 1
            if it % args.eval_every == 0 or it == args.iters:
                model.eval(); scores = []; scene_scores = {}
                hs_, hm_ = stats["hr_std"], stats["hr_mean"]
                with torch.no_grad():
                    for j in range(len(va)):
                        sb = va[j]
                        b = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in sb.items()}
                        kw = batch_to(None, b)
                        pv = model(**kw)
                        pv = (pv * hs_ + hm_).cpu()
                        hrv = (b["hr"] * hs_ + hm_).cpu()
                        mkv = b["sm"].cpu() if "sm" in b else None
                        one = masked_psnr(pv, hrv, mkv)
                        scores.append(one)
                        ckey = str(sb.get("scene", j)) if not torch.is_tensor(sb.get("scene", j)) else str(j)
                        scene_scores[ckey] = float(one)
                json.dump(scene_scores, open(os.path.join(outdir, "val_scene_psnr_%d.json" % it), "w"), indent=1)
                model.train()
                ps = float(np.nanmean(scores))
                el = time.time() - t0
                log.write("%d,%.6f,%.4f,%.1f\n" % (it, float(loss), ps, el)); log.flush()
                print("iter %d loss %.5f valPSNR %.3f (%.0fs)" % (it, float(loss), ps, el), flush=True)
                torch.save({"model": model.state_dict(), "args": vars(args), "stats": stats},
                           os.path.join(outdir, "last.pt"))
            if it >= args.iters:
                break
    log.close()
    print("DONE", name, flush=True)


if __name__ == "__main__":
    main()
