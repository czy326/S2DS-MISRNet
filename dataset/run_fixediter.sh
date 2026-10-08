#!/bin/bash
# Diagnostic: remove checkpoint-selection noise by comparing all arms at the SAME
# iteration (last.pt = ckpt_030000, full cosine schedule, identical budget).
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
LOG=$R/dataset/pipeline_fixediter.log
SNAP=$R/runs_s2ds/_snap_best
say(){ echo "=== $* $(date +%H:%M:%S) ===" >> $LOG; }
mkdir -p $SNAP
for d in $R/runs_s2ds/s2dsR_*; do
  n=$(basename $d); mkdir -p $SNAP/$n
  cp $d/scene_psnr_*.json $SNAP/$n/ 2>/dev/null
done
say "snapshot done"
lane () {
  for NAME in "$@"; do
    for M in hard valid; do
      $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M --ckpt last.pt >> $LOG 2>&1
    done
    say "fixed-iter eval done $NAME"
  done
}
say "start"
lane s2dsR_armA_s2026 s2dsR_armC_s2026 s2dsR_armA_s2027 s2dsR_armC_s2027 &
P1=$!
lane s2dsR_armB_s2026 s2dsR_armD_s2026 s2dsR_armB_s2027 s2dsR_armD_s2027 &
P2=$!
wait $P1 $P2
say "eval done"
for SEED in 2026 2027; do
  for M in hard valid; do
    $PY -u $R/misr/analyze_s2ds_2x2.py --seed $SEED --split test --mask $M \
        --prefix "s2dsR_arm%s_s%d" >> $LOG 2>&1
  done
done
say "FIXEDITER DONE"
