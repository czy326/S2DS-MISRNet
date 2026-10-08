#!/bin/bash
# Round 3: per-pixel temporal attention (P1) + cosine LR, full 2x2 x 2 seeds.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
LOG=$D/pipeline_r3.log
say(){ echo "=== $* $(date +%H:%M:%S) ===" >> $LOG; }
lane () {
  local ARMS=$1
  for SEED in 2026 2027; do
    for ARM in $ARMS; do
      local NAME="s2dsR_arm${ARM}_s${SEED}"
      say "train start $NAME"
      $PY -u $R/misr/train_s2ds.py --arm $ARM --seed $SEED --iters 30000 --batch 8 \
          --eval-every 3000 --save-every 3000 --att-mode pixel --lr-schedule cosine \
          --out $NAME >> $LOG 2>&1
      say "train end $NAME rc=$?"
      $PY -u $D/select_ckpt.py --run $NAME >> $LOG 2>&1
      for M in valid hard; do
        $PY -u $R/misr/eval_s2ds.py --run $NAME --split val  --mask $M >> $LOG 2>&1
        $PY -u $R/misr/eval_s2ds.py --run $NAME --split test --mask $M >> $LOG 2>&1
      done
      say "eval end $NAME"
    done
  done
  say "lane($ARMS) done"
}
say "round-3 start"
lane "A C" &
P1=$!
lane "B D" &
P2=$!
wait $P1 $P2
say "training done"
for SEED in 2026 2027; do
  for sp in val test; do
    for M in hard valid; do
      $PY -u $R/misr/analyze_s2ds_2x2.py --seed $SEED --split $sp --mask $M \
          --prefix "s2dsR_arm%s_s%d" >> $LOG 2>&1
    done
  done
done
$PY -u $D/write_verdict.py >> $LOG 2>&1
say "ROUND3 DONE"
