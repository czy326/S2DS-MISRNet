"""Live progress dashboard for the self-built dataset + 2x2 interaction experiment.

Run (from Windows):
    wsl.exe -e bash -lc '/home/czy/miniconda3/envs/emssm/bin/python /mnt/e/论文2/dataset/watch.py'

Options:
    --once              print a single snapshot and exit
    --interval 15       refresh seconds (default 15)
    --no-clear          do not clear the screen between refreshes
"""
import os
import re
import json
import time
import glob
import argparse
import datetime
import subprocess

D = "/mnt/e/论文2/dataset"
RAW = "/home/czy/data/s2ds"
BUILT = "/home/czy/data/s2ds_built"
RUNS = "/mnt/e/论文2/runs_s2ds"
LOGS = {"g1": "/home/czy/g1.log", "g2": "/home/czy/g2.log", "g3": "/home/czy/g3.log"}


def sh(cmd):
    try:
        return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ""


def tail(path, n=1, pat=None):
    if not os.path.exists(path):
        return []
    try:
        lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:                                     # noqa: BLE001
        return []
    if pat:
        lines = [x for x in lines if re.search(pat, x)]
    return lines[-n:]


def aoi_names():
    try:
        return list(json.load(open(os.path.join(D, "aois.json"))))
    except Exception:                                     # noqa: BLE001
        return []


def collected():
    out = {}
    for f in glob.glob(os.path.join(RAW, "*.npz")):
        nm = os.path.basename(f)[:-len("_2024.npz")]
        out[nm] = os.path.getsize(f) / 1e6
    return out


def saved_info():
    """aoi -> (dates, targets, seconds, corr) parsed from the collector logs."""
    info = {}
    for lp in LOGS.values():
        for ln in tail(lp, 10 ** 6, r"SAVED"):
            m = re.match(r"\[(\d\d:\d\d:\d\d)\] (\S+): SAVED .*?dates=(\d+) targets=(\d+) [\d.]+ MB \((\d+)s\)", ln)
            if m:
                info[m.group(2)] = dict(t=m.group(1), dates=int(m.group(3)),
                                        targets=int(m.group(4)), secs=int(m.group(5)), corr=None)
        for ln in tail(lp, 10 ** 6, r"alignment check"):
            m = re.match(r"\[(\d\d:\d\d:\d\d)\] (\S+): LR/HR alignment check: corr=([\d.]+) rel_err=([\d.]+)", ln)
            if m and m.group(2) in info:
                info[m.group(2)]["corr"] = float(m.group(3))
    return info


def running_info():
    """aoi -> last status line, and aoi -> (done, total, secs) of its latest pass1 progress."""
    out, prog = {}, {}
    for lp in LOGS.values():
        for ln in tail(lp, 10 ** 6, r"^(?!.*(SAVED|ALL DONE))"):
            m = re.match(r"\[(\d\d:\d\d:\d\d)\] (\S+): (.*)$", ln)
            if not m:
                continue
            nm, msg = m.group(2), m.group(3)
            out[nm] = (m.group(1), msg)
            p = re.search(r"pass1 (\d+)/(\d+) \(ok=\d+, (\d+)s\)", msg)
            if p:
                prog[nm] = (int(p.group(1)), int(p.group(2)), int(p.group(3)))
    return out, prog


def training_rows():
    rows = []
    for rd in sorted(glob.glob(os.path.join(RUNS, "s2ds_arm*_s*"))):
        name = os.path.basename(rd)
        lc = os.path.join(rd, "log.csv")
        m = re.match(r"s2ds_arm([ABCD])_s(\d+)", name)
        if not m:
            continue
        arm, seed = m.group(1), m.group(2)
        last = tail(lc, 1)
        it, val = "-", "-"
        if last and "," in last[0] and not last[0].startswith("iter"):
            p = last[0].split(",")
            it, val = p[0], ("%.3f" % float(p[2])) if len(p) > 2 else "-"
        done = os.path.exists(os.path.join(rd, "scene_psnr_test.json"))
        rows.append((arm, seed, it, val, done))
    return rows


def verdict_lines():
    p = os.path.join(D, "INTERACTION_RESULT.md")
    if not os.path.exists(p):
        return []
    txt = open(p, encoding="utf-8", errors="ignore").read()
    return [x for x in txt.splitlines() if "交互判定" in x or "多种子结论" in x][:6]


def snapshot():
    L = []
    A = L.append
    now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    A("\033[1m=== 自建 S2 数据集中间态 · 2×2 交互效应验证 ===\033[0m  %s" % now)
    A("")
    names = aoi_names()
    done = collected()
    si = saved_info()
    ri, prog = running_info()
    ncol = sh("pgrep -fc '[c]ollect_s2'") or "0"
    npipe = sh("pgrep -fc '[r]un_all'") or "0"
    ntrain = sh("pgrep -fc '[t]rain_s2ds'") or "0"

    # ---- collection ----
    A("\033[1m【采集】\033[0m %d/%d AOI 完成   进程: 采集 %s / 流水线 %s / 训练 %s"
      % (len(done), len(names) or 18, ncol, npipe, ntrain))
    recent = [v["secs"] / max(v["dates"], 1) for v in si.values() if v["secs"] > 60]
    rate = sum(recent[-3:]) / max(len(recent[-3:]), 1) if recent else None
    for nm in names:
        if nm in done:
            v = si.get(nm)
            if v:
                A("  \033[32m✓\033[0m %-17s %3d 日期 %2d 目标  %5.1f s/日期  对齐 corr=%s"
                  % (nm, v["dates"], v["targets"], v["secs"] / max(v["dates"], 1),
                     ("%.4f" % v["corr"]) if v["corr"] is not None else "?"))
            else:
                A("  \033[32m✓\033[0m %-17s (%.1f MB)" % (nm, done[nm]))
        elif nm in ri:
            t, msg = ri[nm]
            A("  \033[33m⏳\033[0m %-17s %s  %s" % (nm, t, msg[:66]))
        else:
            A("  \033[90m·\033[0m %-17s 排队中" % nm)

    # ---- ETA ----
    rem_dates, done_dates, secs = 0, 0, 0
    for nm in names:
        if nm in done:
            continue
        if nm in prog:
            d, tot, s = prog[nm]
            rem_dates += max(tot - d, 0)
            done_dates += d
            secs += s
        else:
            v = si.get(nm)
            rem_dates += v["dates"] if v else 120
    if done_dates > 20:
        live = secs / done_dates
        lanes = max(int(ncol) // 2, 1)
        eta = rem_dates * live / lanes / 60
        A("")
        A("  \033[1m在跑 AOI 实测 %.1f s/日期\033[0m（%d 组并行）  剩余 %d 日期（未开始的按 120 日期估）"
          % (live, lanes, rem_dates))
        A("  ⇒ \033[1m采集预计还需 %.0f–%.0f 分钟\033[0m（快/慢时段差异大，取下限与 1.6× 上限）"
          % (eta, eta * 1.6))
    elif rate:
        A("")
        A("  \033[1m近 3 个 AOI 速率 %.1f s/日期\033[0m  ⇒ 采集剩余约 %.0f 分钟"
          % (rate, rem_dates * rate / 2 / 60))

    # ---- built shards ----
    A("")
    A("\033[1m【成型】\033[0m shard %d 个" % len(glob.glob(os.path.join(BUILT, "*.npz"))))
    sp = os.path.join(D, "stats.json")
    if os.path.exists(sp):
        try:
            st = json.load(open(sp))
            t = st.get("temporal", {})
            c = st.get("cloud", {})
            A("  样本 %s  AOI %s  test 空间单元 %s" % (st.get("total_samples"), st.get("n_aois"),
                                                       st.get("n_spatial_units_test")))
            A("  |Δt| 中位 %.0f d / p90 %.0f d   输入帧晴空率 %.3f   逐 LR 像素云占比 %.3f   目标晴空率 %.3f"
              % (t.get("abs_dt_median_days", float("nan")), t.get("abs_dt_p90_days", float("nan")),
                 c.get("frame_clear_frac_mean", float("nan")),
                 c.get("lr_pixel_cloudfrac_mean", float("nan")),
                 c.get("hr_target_clear_frac_mean", float("nan"))))
        except Exception:                                 # noqa: BLE001
            pass

    # ---- pipeline ----
    A("")
    A("\033[1m【流水线】\033[0m")
    pl = tail(os.path.join(D, "pipeline.log"), 4)
    for ln in pl:
        A("  " + ln[:110])
    if not pl:
        A("  (尚无日志)")

    # ---- training ----
    tr = training_rows()
    if tr:
        A("")
        A("\033[1m【训练】\033[0m (arm, seed, 当前 iter, val PSNR, 已评测)")
        for arm, seed, it, val, done_ in tr:
            A("  arm %s  seed %s  iter %-5s val %-8s %s"
              % (arm, seed, it, val, "\033[32m✓test 已评测\033[0m" if done_ else ""))

    # ---- verdict ----
    v = verdict_lines()
    A("")
    A("\033[1m【交互判定】\033[0m")
    if v:
        for ln in v:
            A("  " + ln[:130])
    else:
        A("  (尚无 INTERACTION_RESULT.md；判定规则：I=(D−C)−(B−A) ≥ +0.10 dB 且 CI 下界>0)")

    # ---- baselines (short) ----
    bp = os.path.join(D, "bench_baselines.json")
    if os.path.exists(bp):
        try:
            b = json.load(open(bp))
            for split, sv in (b.get("splits") or {}).items():
                pb = sv.get("per_baseline", {})
                if pb:
                    A("")
                    A("\033[1m【基线 %s】\033[0m " % split + "  ".join(
                        "%s=%.2f" % (k, v["mean"]) for k, v in pb.items()))
        except Exception:                                 # noqa: BLE001
            pass
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--no-clear", action="store_true")
    args = ap.parse_args()
    try:
        while True:
            body = "\n".join(snapshot())
            if not args.no_clear and not args.once:
                print("\033[H\033[J" + body, flush=True)
            else:
                print(body, flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
