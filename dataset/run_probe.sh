#!/bin/bash
# Probe: can the new architecture cross the zero-training cloud-aware baseline?
#   P1 : per-pixel temporal attention + cosine LR
#   P2 : P1 + fixed cloud-aware weighted-mean residual base (guarantees the floor)
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
LOG=$D/pipeline_probe.log
say(){ echo "=== $* $(date +%H:%M:%S) ===" >> $LOG; }
run1 () {
  local NAME=s2dsP_armD_s2026
  say "train $NAME (pixel att)"
  $PY -u $R/misr/train_s2ds.py --arm D --seed 2026 --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --att-mode pixel --out $NAME >> $LOG 2>&1
  $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
  for M in valid hard; do $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1; done
  say "$NAME done"
}
run2 () {
  local NAME=s2dsP2_armD_s2026
  say "train $NAME (pixel att + base residual)"
  $PY -u $R/misr/train_s2ds.py --arm D --seed 2026 --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --att-mode pixel --base-resid --out $NAME >> $LOG 2>&1
  $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
  for M in valid hard; do $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1; done
  say "$NAME done"
}
say "probe start"
run1 &
P1=$!
run2 &
P2=$!
wait $P1 $P2
say "PROBE DONE"
