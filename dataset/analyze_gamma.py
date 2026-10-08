# -*- coding: utf-8 -*-
"""Checklist item 6(1), the cheap half: how far does the learnable suppression scalar
gamma actually move?

The q route suppresses cloud pixels with  f_t <- f_t * (1 - sigmoid(gamma) * cld_t)  and
gamma starts at 0, i.e. sigmoid = 0.5.  If gamma stays near its initialisation over the
whole training run, then the per-pixel branch is effectively a fixed 50 % suppression and
"the gate learned something" would be the wrong reading of the result.  Reading the scalar
straight off the trained checkpoints costs nothing, so it is reported for every q arm
before the gamma0 scan in E8 is even run.
"""
import glob
import json
import os

import torch

RUNS = "/mnt/e/论文2/runs_s2ds"
OUT = "/mnt/e/论文2/dataset/gamma_report.json"
PREF = ["s2dsR_arm", "s2dsV2_arm", "s2dsE8_"]


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "s2ds*"))):
        name = os.path.basename(d)
        if not any(name.startswith(p) for p in PREF):
            continue
        for ck in ("ckpt_003000.pt", "ckpt_015000.pt", "last.pt"):
            p = os.path.join(d, ck)
            if not os.path.exists(p):
                continue
            ckv = torch.load(p, map_location="cpu", weights_only=False)
            sd = ckv["model"]
            g = None
            for k, v in sd.items():
                if "q_gate" in k:
                    g = float(v.reshape(-1)[0])
            if g is None:
                continue
            rows.append(dict(run=name, ckpt=ck, gamma=round(g, 6),
                             sigmoid=round(float(torch.sigmoid(torch.tensor(g))), 5)))
    # collapse to one line per run
    seen = {}
    for r in rows:
        seen.setdefault(r["run"], {})[r["ckpt"]] = r
    print("%-28s %-26s %-26s %s" % ("run", "gamma@3k (sig)", "gamma@15k (sig)",
                                    "gamma@30k (sig)"))
    out = {}
    for run in sorted(seen):
        cells = []
        rec = {}
        for ck in ("ckpt_003000.pt", "ckpt_015000.pt", "last.pt"):
            r = seen[run].get(ck)
            cells.append("%+.4f (%.4f)" % (r["gamma"], r["sigmoid"]) if r else "-")
            if r:
                rec[ck] = (r["gamma"], r["sigmoid"])
        print("%-28s %-26s %-26s %s" % (run, cells[0], cells[1], cells[2]))
        out[run] = rec
    # across-run spread at the primary endpoint
    final = [v["last.pt"][1] for v in out.values() if "last.pt" in v]
    if final:
        print("\nsigmoid(gamma) at 30k over %d runs: min %.4f  max %.4f  mean %.4f"
              % (len(final), min(final), max(final), sum(final) / len(final)))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
