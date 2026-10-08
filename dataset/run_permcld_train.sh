#!/bin/bash
# E5 (train-time arm): retrain arm B with the per-pixel cloud pattern block-permuted
# (8x8 blocks) inside every frame, and evaluate with the SAME scrambled masks.
#
# Why train-time and not eval-time only: the eval-time control (already measured on
# seed 2028, retained fraction 0.37) needs the trained weights, and only seed 2028 still
# has them.  Retraining gives all three seeds and, more importantly, answers the harder
# question -- "can q be relearned from a spatially meaningless mask?" -- rather than only
# "does the already-learned mechanism break?".
#
# What the permutation does and does not destroy (48x48 is divisible by 8, so the block
# permutation is exact and information-preserving at the frame level):
#   destroyed : per-pixel spatial correspondence -> route (i)  feature suppression
#               f * (1 - sigmoid(q_gate) * cld),  route (iii) cld channel in the
#               per-pixel attention conv.
#   preserved : frame-level clear fraction q = 1 - mean(cld)  -> route (ii) q_mlp
#               embedding, which depends only on the frame mean.
# So the retained fraction is the share of q's gain attributable to the frame-level
# route, and 1 - retained is the share attributable to the per-pixel routes.
#
# arm A never consumes cld, so its existing PSNR is exactly reusable as the reference;
# gain_intact = B_true - A and gain_scrambled = B_perm - A are both paired per scene.
#
# Pre-registered expectation: mean retained fraction <= 0.5  (q needs the pixels).
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
OUT=$RUNS/_snap_permcld
LOG=$D/pipeline_permcld_train.log
mkdir -p $OUT
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

run_one () {
  local SEED=$1 NAME=s2dsP_armB_s$1
  say "train start $NAME (arm=B seed=$SEED permute-cld 8)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_s2ds.py --arm B --seed $SEED --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --att-mode pixel --lr-schedule cosine \
      --dt-mode gate --permute-cld 8 --out $NAME >> $LOG 2>&1
  say "train end $NAME rc=$?"
  $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
  for M in hard valid; do
    # best-ckpt endpoint (secondary)
    $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1
    mkdir -p $OUT/best/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}.json $OUT/best/$NAME/ 2>/dev/null
    # fixed-iteration endpoint (primary): identical 30k budget for every arm
    $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M --ckpt last.pt \
        --tag fixed >> $LOG 2>&1
    mkdir -p $OUT/fixed/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}_fixed.json $OUT/fixed/$NAME/ 2>/dev/null
  done
  say "eval end $NAME"
}

say "E5-train start (permuted-cloud, arm B, seeds 2026/2027/2028)"
for S in 2026 2027 2028; do run_one $S; done
say "training done"

$PY -u $D/analyze_permcld_train.py >> $LOG 2>&1
say "E5-train DONE"
