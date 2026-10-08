"""Trainer for the 2x2 interaction experiment on the self-built S2 dataset.

Arms (factor F1 = dt embedding, F2 = cloud reliability gating):
  A (off,off)  B (off,on)  C (on,off)  D (on,on)

Protocol is pre-registered in dataset/EXPERIMENT_PROTOCOL.md: equal iteration budget for
all arms, masked L1 (clear target pixels only), per-sample PSNR dumped at the end.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
from misr.models import build_model, n_params          # noqa: E402
from s2ds_dataset import S2DS                          # noqa: E402

ARMS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}


def psnr_masked(pred, gt, mask, peak=1.0):
    m = mask > 0.5
    if int(m.sum()) < 32:
        return float("nan")
    d = (pred - gt)[:, m]
    mse = float((d ** 2).mean())
    return 10 * np.log10(peak ** 2 / (mse + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=0,
                    help="also dump ckpt_<iter>.pt every N iters (needed to pick a "
                         "validation-best checkpoint instead of blindly using last.pt)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--att-mode", default="pixel", choices=["pixel", "frame"])
    ap.add_argument("--base-resid", action="store_true")
    ap.add_argument("--head-init", type=float, default=0.0)
    ap.add_argument("--lr-schedule", default="cosine", choices=["const", "cosine"])
    ap.add_argument("--dt-mode", default="add", choices=["add", "film", "gate"])
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shuffle-frames", action="store_true",
                    help="permute the T frame axis per sample: removes the ordinal time "
                         "rank that the canonical |dt|-ascending order leaks to every arm")
    ap.add_argument("--permute-cld", type=int, default=0,
                    help="block size for the cloud-mask negative control (0 = off)")
    args = ap.parse_args()

    use_dt, use_q = ARMS[args.arm]
    name = args.out or ("s2ds_arm%s_s%d" % (args.arm, args.seed))
    rd = os.path.join("/mnt/e/论文2/runs_s2ds", name)
    os.makedirs(rd, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dskw = dict(shuffle=args.shuffle_frames, seed=args.seed, permute_cld=args.permute_cld)
    tr = S2DS("train", **dskw)
    try:
        va = S2DS("val", **dskw)
    except Exception as e:
        # val is only used for monitoring; the endpoint is the test split, evaluated
        # separately by eval_s2ds.py.  Falling back keeps a partially-collected dataset
        # usable instead of aborting the whole pipeline.
        print("[warn] no val shards (%s) -> monitoring on a train subset" % str(e)[:60], flush=True)
        va = S2DS("train", max_samples=40, **dskw)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                     drop_last=True, persistent_workers=False)
    model = build_model(cin=4, c=args.c, scale=4, use_dt=use_dt, use_q=use_q,
                        att_mode=args.att_mode, base_resid=args.base_resid,
                        head_init=args.head_init, dt_mode=args.dt_mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def set_lr(step):
        if args.lr_schedule == "const":
            lr = args.lr
        elif step < args.warmup:
            lr = args.lr * (step + 1) / max(args.warmup, 1)
        else:
            import math
            p = (step - args.warmup) / max(args.iters - args.warmup, 1)
            lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr
    print("[%s] params=%d train=%d val=%d dev=%s iters=%d"
          % (name, n_params(model), len(tr), len(va), dev, args.iters), flush=True)

    log = open(os.path.join(rd, "log.csv"), "w")
    log.write("iter,loss,val_psnr,sec\n")
    it, t0 = 0, time.time()

    def to_dev(b):
        return dict(lr=b["lr"].to(dev), dt=b["dt"].to(dev), cld=b["cld"].to(dev),
                    q=b["q"].to(dev), hr=b["hr"].to(dev), valid=b["valid"].to(dev))

    def evaluate(n_max=40):
        model.eval()
        sc = []
        with torch.no_grad():
            for j in range(min(len(va), n_max)):
                b = to_dev({k: va[j][k][None] for k in
                            ["lr", "dt", "cld", "q", "hr", "valid"]})
                p = model(lr=b["lr"], q=b["q"], cld=b["cld"], dt=b["dt"])
                sc.append(psnr_masked(p[0].cpu().numpy(), b["hr"][0].cpu().numpy(),
                                      b["valid"][0].cpu().numpy()))
        model.train()
        return float(np.nanmean(sc))

    while it < args.iters:
        for b in trl:
            b = to_dev(b)
            pred = model(lr=b["lr"], q=b["q"], cld=b["cld"], dt=b["dt"])
            m = b["valid"].unsqueeze(1)
            loss = ((pred - b["hr"]).abs() * m).sum() / (m.sum() * pred.shape[1] + 1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1
            set_lr(it)
            if it % args.eval_every == 0 or it >= args.iters:
                v = evaluate()
                log.write("%d,%.6f,%.4f,%.1f\n" % (it, float(loss), v, time.time() - t0))
                log.flush()
                print("iter %d loss %.5f valPSNR %.3f (%.0fs)" % (it, float(loss), v, time.time() - t0),
                      flush=True)
                torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(rd, "last.pt"))
                if args.save_every and it % args.save_every == 0:
                    torch.save({"model": model.state_dict(), "args": vars(args), "val_psnr": v},
                               os.path.join(rd, "ckpt_%06d.pt" % it))
            if it >= args.iters:
                break
    log.close()
    print("DONE", name, flush=True)


if __name__ == "__main__":
    main()
