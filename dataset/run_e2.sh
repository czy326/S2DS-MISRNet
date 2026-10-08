#!/bin/bash
# E2 -- published multi-image SR baselines as an absolute anchor.
#
# The 2x2 only says what dt and q are worth INSIDE our architecture.  These two runs say
# whether the architecture is competitive at all:
#   highresnet  Deudon et al. 2018, recursive pairwise fusion in the LR domain
#   rams        Salvetti et al. 2020, bicubic-upsample-first + per-pixel residual
#               attention over frames, fused at HR resolution
#
# Both are cloud-blind and time-blind (they take only the LR stack), i.e. they are the
# published-method counterpart of our arm A.  Channel widths are chosen so that both land
# near our arms' parameter count: highresnet c=36 -> ~204k, rams c=32 -> ~125k (RAMS is
# kept smaller on purpose: it encodes every one of the 12 frames at 192x192, which costs
# roughly 5x the step time of the LR-domain models, so three seeds did not fit the budget
# -- it is run for one seed and reported descriptively).
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
OUT=$RUNS/_snap_e2
LOG=$D/pipeline_e2.log
mkdir -p $OUT
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

run_one () {
  # same `set -u` trap as in run_e1.sh: split the assignments onto separate lines
  local MODEL=$1
  local SEED=$2
  local C=$3
  local BATCH=$4
  local NAME=s2dsE2_${MODEL}_s${SEED}
  # resume guard: these runs are 1-2 h each, so a restart must never redo a finished one
  if [ -f $RUNS/$NAME/scene_psnr_test_hard_fixed.json ] && \
     [ -f $RUNS/$NAME/scene_psnr_test_valid_fixed.json ]; then
    say "skip $NAME (fixed-endpoint outputs already present)"
    return 0
  fi
  say "train start $NAME (model=$MODEL seed=$SEED c=$C batch=$BATCH)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_e1.py --model $MODEL --seed $SEED --c $C --batch $BATCH \
      --iters 30000 --eval-every 3000 --save-every 3000 --lr-schedule cosine \
      --out $NAME >> $LOG 2>&1
  say "train end $NAME rc=$?"
  $PY -u $D/select_ckpt_e1.py --run $NAME >> $LOG 2>&1
  for M in hard valid; do
    $PY -u $R/misr/eval_e1.py --run $NAME --split test --mask $M >> $LOG 2>&1
    mkdir -p $OUT/best/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}.json $OUT/best/$NAME/ 2>/dev/null
    $PY -u $R/misr/eval_e1.py --run $NAME --split test --mask $M --ckpt last.pt \
        --tag fixed >> $LOG 2>&1
    mkdir -p $OUT/fixed/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}_fixed.json $OUT/fixed/$NAME/ 2>/dev/null
  done
  say "eval end $NAME"
}

lane1 () { for S in 2026 2027 2028; do run_one highresnet $S 36 8; done; say "lane1 done"; }
lane2 () { run_one rams 2026 32 8; say "lane2 done"; }

say "E2 start (highresnet 3 seeds / rams 1 seed)"
$PY -u $D/report_baseline_params.py >> $LOG 2>&1
lane1 &
P1=$!
lane2 &
P2=$!
wait $P1 $P2
say "training done"

$PY -u $D/analyze_e2.py >> $LOG 2>&1
say "E2 DONE"
