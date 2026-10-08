# -*- coding: utf-8 -*-
"""Generate the exact cells for the revised 表 5 (v4 manuscript).

Fixes the valid-column bug found in the v4 draft: the draft paired the
VALID-stratum run PSNR against the HARD-stratum zero-training baseline.
Both strata must come from the same endpoint.
"""
import os, json, sys, io
import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUNS = "/mnt/e/论文2/runs_s2ds"
BASE = json.load(open("/mnt/e/论文2/dataset/zero_baseline_scene.json", encoding="utf-8"))
SEEDS = [2026, 2027, 2028]


def run_file(run, mask, tag):
    fn = "scene_psnr_test_%s%s.json" % (mask, ("_%s" % tag) if tag else "")
    p = os.path.join(RUNS, run, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cell(run, mask, tag):
    d = run_file(run, mask, tag)
    b = BASE[mask]
    ks = sorted(set(d) & set(b))
    x = np.array([b[k] for k in ks], float)
    y = np.array([d[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, n = x[m], y[m], int(m.sum())
    diff = y - x
    se = diff.std(ddof=1) / np.sqrt(n)
    w = int((diff > 0).sum())
    return dict(delta=float(diff.mean()), lo=float(diff.mean() - 1.96 * se),
                hi=float(diff.mean() + 1.96 * se), win=w, n=n,
                p=float(stats.ttest_rel(y, x).pvalue))


def fmt(c, with_ci=True):
    if with_ci:
        return "%+0.3f [%+0.3f, %+0.3f]" % (c["delta"], c["lo"], c["hi"])
    return "%+0.3f" % c["delta"]


def wr(c):
    return "%d%%（%d/%d）" % (round(c["win"] / c["n"] * 100), c["win"], c["n"])


ROWS = [
    ("HighRes-net", "s2dsE2_highresnet_s%d", "fixed", True),
    ("HighRes-net + cld（云掩膜作为第 5 输入通道）", "s2dsE9_highresnet_cld_s%d", "fixed", True),
    ("RAMS", "s2dsE2_rams_s%d", "fixed", True),
]

print("### 逐种子行（HARD / valid，固定迭代口径）")
for name, pat, tag, ci in ROWS:
    for s in SEEDS:
        run = pat % s
        if not os.path.isdir(os.path.join(RUNS, run)):
            continue
        h = cell(run, "hard", tag)
        v = cell(run, "valid", tag)
        print("%-34s %d | %s | %s | %s | %s" % (name, s, fmt(h, ci), wr(h), fmt(v, ci), wr(v)))

print("\n### arm A / arm D（untagged 文件即固定迭代端点）")
for nm, arm in [("arm A", "armA"), ("arm D", "armD")]:
    hs, vs, hw, vw = [], [], [], []
    for s in SEEDS:
        h = cell("s2dsV2_%s_s%d" % (arm, s), "hard", None)
        v = cell("s2dsV2_%s_s%d" % (arm, s), "valid", None)
        hs.append("%+0.3f" % h["delta"])
        vs.append("%+0.3f" % v["delta"])
        hw.append("%d%%" % round(h["win"] / h["n"] * 100))
        vw.append("%d%%" % round(v["win"] / v["n"] * 100))
    print("%s HARD delta: %s" % (nm, " / ".join(hs)))
    print("%s HARD win  : %s" % (nm, " / ".join(hw)))
    print("%s valid delta: %s" % (nm, " / ".join(vs)))
    print("%s valid win  : %s" % (nm, " / ".join(vw)))

print("\n### E9 关键对照（cld 版 vs 原版 / vs arm，种子级）")
for mask in ["hard", "valid"]:
    for nm, a, b in [("cld-blind", "s2dsE9_highresnet_cld_s%d", "s2dsE2_highresnet_s%d"),
                     ("armA-cld", "s2dsV2_armA_s%d", "s2dsE9_highresnet_cld_s%d"),
                     ("armD-cld", "s2dsV2_armD_s%d", "s2dsE9_highresnet_cld_s%d")]:
        vals, wins = [], []
        for s in SEEDS:
            ta = None if a.startswith("s2dsV2") else "fixed"
            tb = None if b.startswith("s2dsV2") else "fixed"
            da = run_file(a % s, mask, ta)
            db = run_file(b % s, mask, tb)
            ks = sorted(set(da) & set(db))
            x = np.array([db[k] for k in ks], float)
            y = np.array([da[k] for k in ks], float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            d = y - x
            vals.append(float(d.mean()))
            wins.append(round(float((d > 0).mean()) * 100))
        v = np.array(vals)
        t, p = stats.ttest_1samp(v, 0.0)
        print("%-5s %-11s per-seed %s mean=%+0.3f sd=%0.3f p_seed=%0.4f win=%s"
              % (mask, nm, ["%+.3f" % z for z in vals], v.mean(), v.std(ddof=1), p, wins))
