#!/bin/bash
# E9 -- HighRes-net fed with the cloud mask (reviewer control for "cloud-blind baselines").
# Same protocol as run_e2.sh (30k iters, batch 8, cosine LR, c=36), 3 seeds, 2 lanes.
# Model: highresnet_cld (cld as 5th input channel; nothing else changes).
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$HOME/basicsr_stub
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
OUT=$RUNS/_snap_e9
LOG=$D/pipeline_e9.log
mkdir -p $OUT
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

run_one () {
  local SEED=$1
  local NAME=s2dsE9_highresnet_cld_s${SEED}
  if [ -f $RUNS/$NAME/scene_psnr_test_hard_fixed.json ] && \
     [ -f $RUNS/$NAME/scene_psnr_test_valid_fixed.json ]; then
    say "skip $NAME (fixed-endpoint outputs already present)"
    return 0
  fi
  say "train start $NAME (model=highresnet_cld seed=$SEED c=36 batch=8)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_e1.py --model highresnet_cld --seed $SEED --c 36 --batch 8 \
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

lane1 () { run_one 2026; run_one 2028; say "lane1 done"; }
lane2 () { run_one 2027; say "lane2 done"; }

say "E9 start (highresnet_cld 3 seeds)"
lane1 & P1=$!
lane2 & P2=$!
wait $P1 $P2
say "training done"

$PY -u $D/analyze_e9.py >> $LOG 2>&1
say "E9 DONE"
