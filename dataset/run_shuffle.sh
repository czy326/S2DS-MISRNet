#!/bin/bash
# Round 5 (E3) -- frame-order shuffle control.
#
# Motivation (paper 3.1 / 7): the dataset orders the T=12 frames by |dt| ascending, so
# frame index 0 is always the temporally nearest frame.  All arms can read that ordinal
# time rank through the frame axis, which means the dt factor tests "extra value beyond
# the ordinal rank", not "temporal information vs none".
#
# Design: retrain all four arms (A/B/C/D, gate injection, 3 seeds) with the frame axis
# randomly permuted per sample (fixed permutation, reproducible).  Ordered counterparts
# are the existing s2dsV2_* runs (identical protocol).
#
# Predictions, pre-registered two-sided:
#   (i)  (C-A)_shuffled  >  (C-A)_ordered     dt earns its keep once the rank is gone
#   (ii) A_shuffled      <  A_ordered on HARD A loses "which frame is nearest"
#   (iii)(D-B)_shuffled  >  (D-B)_ordered     same for dt on top of q
# If they hold, the dt null is upgraded from "no effect" to "masked by a redundant
# ordinal cue"; if not, the frame-order channel is excluded and the triple-redundancy
# explanation narrows to {cloud weight, learned attention}.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
OUT=$RUNS/_snap_shuffle
LOG=$D/pipeline_shuffle.log
mkdir -p $OUT
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

run_one () {
  local ARM=$1 NAME=$2 SEED=$3
  say "train start $NAME (arm=$ARM seed=$SEED shuffle)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_s2ds.py --arm $ARM --seed $SEED --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --att-mode pixel --lr-schedule cosine \
      --dt-mode gate --shuffle-frames --out $NAME >> $LOG 2>&1
  say "train end $NAME rc=$?"
  $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
  # best-ckpt endpoint (secondary)
  for M in hard valid; do
    $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1
    mkdir -p $OUT/best/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}.json $OUT/best/$NAME/ 2>/dev/null
  done
  # fixed-iteration endpoint (primary): identical budget for every arm, tagged so it
  # never clobbers the plain file written by the best-ckpt pass above
  for M in hard valid; do
    $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M --ckpt last.pt \
        --tag fixed >> $LOG 2>&1
    mkdir -p $OUT/fixed/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}_fixed.json $OUT/fixed/$NAME/ 2>/dev/null
  done
  say "eval end $NAME"
}

# 12 runs / 2 lanes = 6 each (~5 min per run at 100 it/s, ~1 h wall with contention)
lane1 () { for S in 2026 2027 2028; do run_one A s2dsS_armA_s$S $S; done; \
           for S in 2026 2027 2028; do run_one B s2dsS_armB_s$S $S; done; say "lane1 done"; }
lane2 () { for S in 2026 2027 2028; do run_one C s2dsS_armC_s$S $S; done; \
           for S in 2026 2027 2028; do run_one D s2dsS_armD_s$S $S; done; say "lane2 done"; }

say "E3 start (frame-shuffle, arms A/B/C/D, seeds 2026/2027/2028)"
lane1 &
P1=$!
lane2 &
P2=$!
wait $P1 $P2
say "training done"

$PY -u $D/analyze_shuffle.py >> $LOG 2>&1
say "E3 DONE"
