#!/bin/bash
# One-screen status for the E7 / E8 batches.  Run it from WSL:
#     bash /mnt/e/论文2/dataset/status_e7e8.sh
#
# Deliberately filesystem-based: on this box ps / pgrep / uptime are not trustworthy
# across separate wsl.exe invocations (each call sees an isolated process view), so
# progress is judged from checkpoint timestamps and the pipeline logs instead.
set -u
R=/mnt/e/论文2/runs_s2ds
D=/mnt/e/论文2/dataset
NOW=$(date +%s)

echo "now: $(date +%m-%d_%H:%M:%S)"
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"
echo

for PREFIX in s2dsE7_ s2dsE8_; do
  echo "=== $PREFIX ==="
  for d in $R/${PREFIX}*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    last=$(ls -t "$d"/ckpt_*.pt 2>/dev/null | head -1)
    if [ -z "$last" ]; then
      it="no ckpt yet"
    else
      it="$(basename "$last" .pt) @ $(date -r "$last" +%H:%M:%S)"
    fi
    fin=""
    [ -f "$d/scene_psnr_test_hard_fixed.json" ] && fin="[hard fixed done]"
    age=99999
    for f in "$d/log.csv" "$d/ckpt_003000.pt" "$d/scene_psnr_test_hard_fixed.json"; do
      [ -f "$f" ] || continue
      a=$((NOW - $(stat -c %Y "$f")))
      [ "$a" -lt "$age" ] && age=$a
    done
    printf "  %-26s %-28s last write %5ds ago %s\n" "$name" "$it" "$age" "$fin"
  done
  echo
done

echo "=== pipeline markers ==="
for L in $D/pipeline_e7.log $D/pipeline_e8.log $D/pipeline_chain.log; do
  [ -f "$L" ] || continue
  echo "--- $(basename $L)  (mtime $(date -r "$L" +%H:%M:%S))"
  grep -E "train start|train end|eval end|DONE|done|skip" "$L" | tail -8
done
