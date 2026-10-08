#!/bin/bash
# E1 -- internal ablation of the cloud-reliability factor q.
#
# arm B spends its cloud budget through three routes at once:
#   (i)   per-pixel feature suppression  f * (1 - sigmoid(q_gate) * cld)
#   (ii)  cld concatenated into the per-pixel attention conv
#   (iii) frame-level embedding         f + q_mlp(q),  q = 1 - mean(cld)
#
#   B1 = (i)+(ii) only      -> per-pixel routes, no frame-level clear fraction
#   B2 = (iii) only         -> frame-level route, no per-pixel cloud information
#
# arm A (no cloud input) already exists for the three seeds and is reused as the reference.
# arm B (all three routes) already exists too; before this batch is trusted,
# check_e1_equivalence.py verifies that build_model_e1(arm="B") is functionally identical
# to the arm B used in the main 2x2, so the three are directly comparable.
#
# Pre-registered expectation: both B1-A and B2-A are positive on HARD (each route carries
# part of q's value), and B1-A > B2-A on HARD (the per-pixel routes are the larger share,
# consistent with E5's retained fraction below 0.5).
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
OUT=$RUNS/_snap_e1
LOG=$D/pipeline_e1.log
mkdir -p $OUT
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

say "E1 equivalence check (build_model_e1 arm=B  vs  main arm B)"
$PY -u $D/check_e1_equivalence.py >> $LOG 2>&1
say "equivalence rc=$?"

run_one () {
  # NOTE: do NOT write `local ARM=$1 SEED=$2 NAME=...${ARM}...` -- word expansion happens
  # before `local` runs, so ${ARM} is still unset at that point and `set -u` aborts.
  local ARM=$1
  local SEED=$2
  local NAME=s2dsE1_arm${ARM}_s${SEED}
  # resume guard: if the machine sleeps or WSL is torn down mid-batch, re-running this
  # script must not redo finished runs (each run is ~15 min and deterministic per seed)
  if [ -f $RUNS/$NAME/scene_psnr_test_hard_fixed.json ] && \
     [ -f $RUNS/$NAME/scene_psnr_test_valid_fixed.json ]; then
    say "skip $NAME (fixed-endpoint outputs already present)"
    return 0
  fi
  say "train start $NAME (arm=$ARM seed=$SEED)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_e1.py --arm $ARM --seed $SEED --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --att-mode pixel --lr-schedule cosine \
      --dt-mode gate --out $NAME >> $LOG 2>&1
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

lane1 () { for S in 2026 2027 2028; do run_one B1 $S; done; say "lane1 done"; }
lane2 () { for S in 2026 2027 2028; do run_one B2 $S; done; say "lane2 done"; }

say "E1 start (arms B1/B2, seeds 2026/2027/2028)"
lane1 &
P1=$!
lane2 &
P2=$!
wait $P1 $P2
say "training done"

$PY -u $D/analyze_e1.py >> $LOG 2>&1
say "E1 DONE"
