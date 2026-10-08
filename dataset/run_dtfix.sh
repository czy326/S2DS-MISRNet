#!/bin/bash
# Round 4 -- fix the dt injection, keep everything else identical to round 3.
#
#   V1 = "film" : dt normalised with log1p(|dt|/30)*sign(dt), injected as FiLM
#                 (gamma/beta zero-init => the arm starts exactly at the no-dt arm)
#   V2 = "gate" : dt only enters the per-pixel attention gate, never the features
#   legacy "add": round-3 C/D (raw day count added to the features) -- known harmful
#
# A/B arms do not consume dt at all, so round-3 A/B runs are reused as the no-dt
# baseline cells of both 2x2 designs.
#
# Both endpoints are reported:
#   best-ckpt  : checkpoint chosen by val (round-3 primary)
#   fixed-iter : last.pt = 30000 steps for EVERY arm (removes the 0.35-1.00 dB
#               checkpoint-selection noise that flipped the sign of dt in round 3)
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
SNAP_BEST=$RUNS/_snap_best
OUT=$RUNS/_snap_dtfix
LOG=$D/pipeline_dtfix.log
mkdir -p $OUT $OUT/best
say(){ echo "=== $* $(date +%H:%M:%S) ===" >> $LOG; }

train_lane () {
  local V=$1 DTMODE=$2
  for SEED in 2026 2027; do
    for ARM in C D; do
      local NAME=s2ds${V}_arm${ARM}_s${SEED}
      say "train start $NAME (dt-mode=$DTMODE)"
      rm -rf $RUNS/$NAME
      $PY -u $R/misr/train_s2ds.py --arm $ARM --seed $SEED --iters 30000 --batch 8 \
          --eval-every 3000 --save-every 3000 --att-mode pixel --lr-schedule cosine \
          --dt-mode $DTMODE --out $NAME >> $LOG 2>&1
      say "train end $NAME rc=$?"
      $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
      for M in hard valid; do
        $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1
      done
      say "best-eval end $NAME"
    done
  done
  say "lane $V done"
}

say "round-4 start"
train_lane V1 film &
P1=$!
train_lane V2 gate &
P2=$!
wait $P1 $P2
say "training done"

# ---- best-ckpt endpoint: give each design its own A/B cells ------------------
for V in V1 V2; do
  for SEED in 2026 2027; do
    for ARM in A B; do
      mkdir -p $RUNS/s2ds${V}_arm${ARM}_s${SEED}
      cp $RUNS/s2dsR_arm${ARM}_s${SEED}/scene_psnr_test.json \
         $RUNS/s2dsR_arm${ARM}_s${SEED}/scene_psnr_test_hard.json \
         $RUNS/s2dsR_arm${ARM}_s${SEED}/scene_psnr_test_valid.json \
         $RUNS/s2ds${V}_arm${ARM}_s${SEED}/ 2>/dev/null
    done
    for ARM in C D; do
      mkdir -p $OUT/best/s2ds${V}_arm${ARM}_s${SEED}
      cp $RUNS/s2ds${V}_arm${ARM}_s${SEED}/scene_psnr_test*.json \
         $OUT/best/s2ds${V}_arm${ARM}_s${SEED}/ 2>/dev/null
    done
  done
  for SEED in 2026 2027; do
    for M in hard valid; do
      $PY -u $R/misr/analyze_s2ds_2x2.py --seed $SEED --split test --mask $M \
          --prefix "s2ds${V}_arm%s_s%d" >> $LOG 2>&1
      cp $RUNS/analysis_s${SEED}_test_${M}.json \
         $OUT/analysis_${V}_s${SEED}_test_${M}.json 2>/dev/null
    done
  done
done
say "BEST ENDPOINT DONE"

# ---- fixed-iteration endpoint (the primary one after round 3) ---------------
for V in V1 V2; do
  for SEED in 2026 2027; do
    for ARM in C D; do
      NAME=s2ds${V}_arm${ARM}_s${SEED}
      for M in hard valid; do
        $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M \
            --ckpt last.pt >> $LOG 2>&1
      done
    done
  done
done
# A/B at last.pt, then hand the numbers to both designs and restore round-3 files
for SEED in 2026 2027; do
  for ARM in A B; do
    NAME=s2dsR_arm${ARM}_s${SEED}
    for M in hard valid; do
      $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M --ckpt last.pt >> $LOG 2>&1
    done
    for V in V1 V2; do
      mkdir -p $RUNS/s2ds${V}_arm${ARM}_s${SEED}
      cp $RUNS/$NAME/scene_psnr_test.json $RUNS/$NAME/scene_psnr_test_hard.json \
         $RUNS/$NAME/scene_psnr_test_valid.json $RUNS/s2ds${V}_arm${ARM}_s${SEED}/ 2>/dev/null
    done
    cp $SNAP_BEST/$NAME/scene_psnr_*.json $RUNS/$NAME/ 2>/dev/null
  done
done
for V in V1 V2; do
  for SEED in 2026 2027; do
    for M in hard valid; do
      $PY -u $R/misr/analyze_s2ds_2x2.py --seed $SEED --split test --mask $M \
          --prefix "s2ds${V}_arm%s_s%d" >> $LOG 2>&1
      cp $RUNS/analysis_s${SEED}_test_${M}.json \
         $OUT/analysis_${V}_fixed_s${SEED}_test_${M}.json 2>/dev/null
    done
  done
done
# restore round-3 analysis jsons for the record
say "ROUND4 DONE"
