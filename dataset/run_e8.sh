#!/bin/bash
# E8 -- checklist item 6, sub-items (1) and (2): hyper-parameter sensitivity of the q gate.
#
# The checklist asks for three scans.  The third (the zero-training time decay tau) is
# already done and written up in 6.9; these are the other two:
#
#   (1) the learnable suppression scalar gamma   f_t <- f_t * (1 - sigmoid(gamma) * cld_t)
#       Main runs initialise it to gamma=0, i.e. sigmoid=0.5 -- an initial HALF suppression,
#       and on the trained checkpoints it barely moves (sigmoid(gamma) = 0.5015 .. 0.5188,
#       see dataset/analyze_gamma.py).  So the real question is whether the q result is an
#       artefact of that initial 0.5.  Two opposite initialisations are run, gamma0 = -2
#       (sigmoid 0.12) and gamma0 = +2 (sigmoid 0.88), 3 seeds each.
#   (2) the width of the attention gate network  (Conv2d(g_in, c, 1) -> ReLU -> Conv2d(c, 1, 1))
#       The gate width is tied to the model width c, fixed at 32 in every main run.  Two
#       settings, c = 16 (half) and c = 48 (1.5x), 3 seeds each, arm B (q only).
#
# All runs are arm B, T = 12, 30k iterations, batch 8, lr 2e-4 cosine -- identical to the
# main runs except for the scanned knob.  Fixed-iteration (last.pt) is the endpoint.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
RUNS=$R/runs_s2ds
LOG=$D/pipeline_e8.log
mkdir -p $RUNS/_snap_e8/fixed
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

run_one () {
  local NAME=$1
  shift
  if [ -f $RUNS/$NAME/scene_psnr_test_hard_fixed.json ] && \
     [ -f $RUNS/$NAME/scene_psnr_test_valid_fixed.json ]; then
    say "skip $NAME (fixed-endpoint outputs already present)"
    return 0
  fi
  say "train start $NAME ($*)"
  rm -rf $RUNS/$NAME
  $PY -u $R/misr/train_e1.py --model e1 --arm B --iters 30000 --batch 8 \
      --eval-every 3000 --save-every 3000 --lr-schedule cosine --out $NAME "$@" \
      >> $LOG 2>&1
  say "train end $NAME rc=$?"
  $PY -u $D/select_ckpt_e1.py --run $NAME >> $LOG 2>&1
  for M in hard valid; do
    $PY -u $R/misr/eval_e1.py --run $NAME --split test --mask $M --ckpt last.pt \
        --tag fixed >> $LOG 2>&1
    mkdir -p $RUNS/_snap_e8/fixed/$NAME
    cp $RUNS/$NAME/scene_psnr_test_${M}_fixed.json $RUNS/_snap_e8/fixed/$NAME/ 2>/dev/null
  done
  say "eval end $NAME"
}

lane1 () {
  for S in 2026 2027 2028; do
    run_one s2dsE8_gm2_s${S} --seed $S --c 32 --gamma0 -2.0
  done
  for S in 2026 2027 2028; do
    run_one s2dsE8_w16_s${S} --seed $S --c 16
  done
  say "lane1 done"
}

lane2 () {
  for S in 2026 2027 2028; do
    run_one s2dsE8_gp2_s${S} --seed $S --c 32 --gamma0 2.0
  done
  for S in 2026 2027 2028; do
    run_one s2dsE8_w48_s${S} --seed $S --c 48
  done
  say "lane2 done"
}

say "E8 start (gamma0 in {-2,+2} x 3 seeds; gate width in {16,48} x 3 seeds)"
lane1 &
P1=$!
lane2 &
P2=$!
wait $P1 $P2
say "E8 training + eval done"
