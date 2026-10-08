#!/bin/bash
# Archive hygiene: the per-run scene_psnr_*.json of the round-3 A/B runs had been
# overwritten by the val-best snapshot, while every number reported in the paper
# (analysis_s20*_test_*.json, fixed-iteration primary endpoint) was computed from
# last.pt.  Re-evaluate A/B at last.pt and store it under the default filename so
# the archive agrees with the paper.  The val-best values are preserved in
# runs_s2ds/_snap_best/.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
LOG=$R/dataset/pipeline_ab_lastpt.log

lane () {
  for NAME in "$@"; do
    for M in hard valid; do
      $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M \
          --ckpt last.pt >> $LOG 2>&1
    done
    echo "=== done $NAME $(date +%H:%M:%S) ===" >> $LOG
  done
}

echo "=== start $(date +%H:%M:%S) ===" >> $LOG
lane s2dsR_armA_s2026 s2dsR_armB_s2026 &
P1=$!
lane s2dsR_armA_s2027 s2dsR_armB_s2027 &
P2=$!
wait $P1 $P2
echo "=== all done $(date +%H:%M:%S) ===" >> $LOG
