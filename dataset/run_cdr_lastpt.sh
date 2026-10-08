#!/bin/bash
# Same archive hygiene as run_ab_lastpt.sh, for the round-3 C/D runs (the "add"
# injection cell of the dt-injection ablation).  The ablation compares
# ((C-A)+(D-B))/2 within one prefix, so all four arms of s2dsR must sit on the
# same endpoint.  A/B were just moved to last.pt (fixed-iteration primary), so
# C/D have to follow; their val-best values are kept in _snap_best/.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
LOG=$R/dataset/pipeline_cdr_lastpt.log

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
lane s2dsR_armC_s2026 s2dsR_armD_s2026 &
P1=$!
lane s2dsR_armC_s2027 s2dsR_armD_s2027 &
P2=$!
wait $P1 $P2
echo "=== all done $(date +%H:%M:%S) ===" >> $LOG
