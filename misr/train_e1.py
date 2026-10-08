"""E1 -- internal ablation of the cloud-reliability factor q (training entry point).

Identical protocol to misr/train_s2ds.py (30k iterations, cosine LR, masked L1,
per-pixel attention, dt-mode gate), but the model comes from misr/models_e1.py so that the
three q routes can be switched individually:

    B   all routes          (reference; identical to the existing arm B)
    B1  per-pixel only      (feature suppression + cld in the attention gate)
    B2  frame-level only    (q_mlp clear-fraction embedding)

arm A (no cloud input at all) already exists for all three seeds and is reused as the
reference -- it never consumes cld, so it is unaffected by any of these switches.
"""
import os
import sys
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
from misr.build_any import build_from_spec, MODELS                  # noqa: E402
from misr.models_e1 import n_params, Q_ROUTES                       # noqa: E402
from s2ds_dataset import S2DS                                       # noqa: E402


def psnr_masked(pred, gt, mask, peak=1.0):
    m = mask > 0.5
    if int(m.sum()) < 32:
        return float("nan")
    d = (pred - gt)[:, m]
    mse = float((d ** 2).mean())
    return 10 * np.log10(peak ** 2 / (mse + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e1", choices=MODELS,
                    help="e1 = q-route ablation arms; highresnet / rams = E2 baselines")
    ap.add_argument("--arm", default="B1", choices=sorted(Q_ROUTES))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--att-mode", default="pixel", choices=["pixel", "frame"])
    ap.add_argument("--lr-schedule", default="cosine", choices=["const", "cosine"])
    ap.add_argument("--dt-mode", default="gate", choices=["add", "film", "gate"])
    ap.add_argument("--gamma0", type=float, default=0.0,
                    help="initial value of the learnable q-suppression scalar gamma "
                         "(sigmoid(0)=0.5, which is what every main run uses)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out:
        name = args.out
    elif args.model == "e1":
        name = "s2dsE1_arm%s_s%d" % (args.arm, args.seed)
    else:
        name = "s2dsE2_%s_s%d" % (args.model, args.seed)
    rd = os.path.join("/mnt/e/论文2/runs_s2ds", name)
    os.makedirs(rd, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = S2DS("train")
    try:
        va = S2DS("val")
    except Exception as e:
        print("[warn] no val shards (%s) -> monitoring on a train subset" % str(e)[:60],
              flush=True)
        va = S2DS("train", max_samples=40)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                     drop_last=True, persistent_workers=False)
    model = build_from_spec(args, cin=4, scale=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def set_lr(step):
        import math
        if args.lr_schedule == "const":
            lr = args.lr
        elif step < args.warmup:
            lr = args.lr * (step + 1) / max(args.warmup, 1)
        else:
            p = (step - args.warmup) / max(args.iters - args.warmup, 1)
            lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    print("[%s] model=%s arm=%s params=%d train=%d val=%d dev=%s iters=%d batch=%d"
          % (name, args.model, args.arm, n_params(model), len(tr), len(va), dev,
             args.iters, args.batch), flush=True)

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
                log.write("%d,%.6f,%.4f,%.1f\n" % (it, float(loss.detach()), v, time.time() - t0))
                log.flush()
                print("iter %d loss %.5f valPSNR %.3f (%.0fs)"
                      % (it, float(loss.detach()), v, time.time() - t0), flush=True)
                torch.save({"model": model.state_dict(), "args": vars(args)},
                           os.path.join(rd, "last.pt"))
                if args.save_every and it % args.save_every == 0:
                    torch.save({"model": model.state_dict(), "args": vars(args), "val_psnr": v},
                               os.path.join(rd, "ckpt_%06d.pt" % it))
            if it >= args.iters:
                break
    log.close()
    print("DONE", name, flush=True)


if __name__ == "__main__":
    main()
