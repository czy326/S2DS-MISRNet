#!/bin/bash
# Round 4b -- third seed (2028) for the gate design, to pin down the dt result.
# A/B are needed too (the 2x2 needs all four cells at the same seed); they do not
# consume dt, so they run with the unchanged architecture.
#
# Both endpoints are produced:
#   best-ckpt  : val-selected checkpoint (secondary, biased towards under-trained)
#   fixed-iter : last.pt = 30000 steps for every arm (primary endpoint)
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
OUT=$RUNS/_snap_dtfix
LOG=$D/pipeline_s2028.log
mkdir -p $OUT/best
say(){ echo "=== $* $(date +%H:%M:%S) ===" >> $LOG; }

run_one () {
  local ARM=$1 DT=$2 NAME=$3
  say "train start $NAME (arm=$ARM dt-mode=$DT)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_s2ds.py --arm $ARM --seed 2028 --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --att-mode pixel --lr-schedule cosine \
      --dt-mode $DT --out $NAME >> $LOG 2>&1
  say "train end $NAME rc=$?"
  $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
  for M in hard valid; do
    $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1
  done
  say "best-eval end $NAME"
}

lane1 () { run_one A add s2dsV2_armA_s2028; run_one B add s2dsV2_armB_s2028; say "lane1 done"; }
lane2 () { run_one C gate s2dsV2_armC_s2028; run_one D gate s2dsV2_armD_s2028; say "lane2 done"; }

say "round-4b start (seed 2028, gate design)"
lane1 &
P1=$!
lane2 &
P2=$!
wait $P1 $P2
say "training done"

# best-ckpt endpoint ----------------------------------------------------------
for ARM in A B C D; do
  mkdir -p $OUT/best/s2dsV2_arm${ARM}_s2028
  cp $RUNS/s2dsV2_arm${ARM}_s2028/scene_psnr_test*.json $OUT/best/s2dsV2_arm${ARM}_s2028/ 2>/dev/null
done
for M in hard valid; do
  $PY -u $R/misr/analyze_s2ds_2x2.py --seed 2028 --split test --mask $M \
      --prefix "s2dsV2_arm%s_s%d" >> $LOG 2>&1
  cp $RUNS/analysis_s2028_test_${M}.json $OUT/analysis_V2_s2028_test_${M}.json 2>/dev/null
done
say "BEST ENDPOINT DONE"

# fixed-iteration endpoint (primary) ------------------------------------------
for ARM in A B C D; do
  for M in hard valid; do
    $PY -u $R/misr/eval_s2ds.py --run s2dsV2_arm${ARM}_s2028 --split test --mask $M \
        --ckpt last.pt >> $LOG 2>&1
  done
done
for M in hard valid; do
  $PY -u $R/misr/analyze_s2ds_2x2.py --seed 2028 --split test --mask $M \
      --prefix "s2dsV2_arm%s_s%d" >> $LOG 2>&1
  cp $RUNS/analysis_s2028_test_${M}.json $OUT/analysis_V2_fixed_s2028_test_${M}.json 2>/dev/null
done
say "ROUND4B DONE"
