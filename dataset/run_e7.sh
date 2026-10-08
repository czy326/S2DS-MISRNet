#!/bin/bash
# E7 -- item 7 of the major-revision checklist: reproduce BreizhSR / MISR-S2, the most
# direct published competitor (HighRes-net encoder + L-TAE fusion with a target-relative
# temporal positional encoding), on S2DS under this paper's fixed-iteration protocol.
#
# Two model variants, 3 seeds each, 2 lanes (the box saturates at 2 concurrent runs):
#   breizhsr       L-TAE with the temporal positional encoding (the published model)
#   breizhsr_nope  identical weights, positional encoding switched off
#
# The second variant is the point of the run.  The paper's claim is that the temporal
# POSITIONAL ENCODING is what makes irregular multi-date fusion work; if switching it off
# changes nothing on S2DS, then the temporal factor fails here for a reason that lives in
# the data, not in our architecture.
#
# Fixed-iteration (last.pt) is the primary endpoint, exactly as everywhere else in this
# paper; the val-best endpoint is evaluated too and archived under _snap_e7/best.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
LOG=$D/pipeline_e7.log
mkdir -p $RUNS/_snap_e7/best $RUNS/_snap_e7/fixed
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

run_one () {
  local MODEL=$1
  local SEED=$2
  local NAME=s2dsE7_${MODEL}_s${SEED}
  if [ -f $RUNS/$NAME/scene_psnr_test_hard_fixed.json ] && \
     [ -f $RUNS/$NAME/scene_psnr_test_valid_fixed.json ]; then
    say "skip $NAME (fixed-endpoint outputs already present)"
    return 0
  fi
  say "train start $NAME"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_e1.py --model $MODEL --seed $SEED --c 64 --batch 8 \
      --iters 30000 --eval-every 3000 --save-every 3000 --lr-schedule cosine \
      --out $NAME >> $LOG 2>&1
  say "train end $NAME rc=$?"
  $PY -u $D/select_ckpt_e1.py --run $NAME >> $LOG 2>&1
  for M in hard valid; do
    $PY -u $R/misr/eval_e1.py --run $NAME --split test --mask $M >> $LOG 2>&1
    mkdir -p $RUNS/_snap_e7/best/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}.json $RUNS/_snap_e7/best/$NAME/ 2>/dev/null
    $PY -u $R/misr/eval_e1.py --run $NAME --split test --mask $M --ckpt last.pt \
        --tag fixed >> $LOG 2>&1
    cp $RUNS/$NAME/scene_psnr_test_${M}_fixed.json $RUNS/_snap_e7/fixed/$NAME/ 2>/dev/null
  done
  say "eval end $NAME"
}

lane1 () { for S in 2026 2027 2028; do run_one breizhsr $S; done; say "lane1 done"; }
lane2 () { for S in 2026 2027 2028; do run_one breizhsr_nope $S; done; say "lane2 done"; }

say "E7 start (breizhsr / breizhsr_nope, 3 seeds each)"
lane1 &
P1=$!
lane2 &
P2=$!
wait $P1 $P2
say "E7 training + eval done"
