#!/bin/bash
# E9-EASY -- learned-arm evaluation on the EASY stratum (clear in target AND in the
# nearest frame).  Reviewer item: "clear-scene dt helps" so far rests on zero-training
# evidence only; close the loop by re-scoring the 12 existing checkpoints on EASY.
# Inference only, last.pt (fixed-iteration protocol).  A/B s2026/2027 use the s2dsR
# family (verified pixel-identical to V2 for arms A/B).
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$HOME/basicsr_stub
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
LOG=$R/dataset/pipeline_e9easy.log
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

doit(){
  local N=$1
  if [ -f $R/runs_s2ds/$N/scene_psnr_test_easy.json ]; then say "skip $N"; return 0; fi
  say "eval easy $N"
  $PY -u $R/misr/eval_s2ds.py --run $N --split test --mask easy --ckpt last.pt >> $LOG 2>&1
  say "done $N rc=$?"
}

lane1(){ for N in s2dsR_armA_s2026 s2dsR_armA_s2027 s2dsV2_armA_s2028 s2dsR_armB_s2026 s2dsR_armB_s2027 s2dsV2_armB_s2028; do doit $N; done; say "lane1 done"; }
lane2(){ for N in s2dsV2_armC_s2026 s2dsV2_armC_s2027 s2dsV2_armC_s2028 s2dsV2_armD_s2026 s2dsV2_armD_s2027 s2dsV2_armD_s2028; do doit $N; done; say "lane2 done"; }

say "E9-EASY start"
lane1 & P1=$!
lane2 & P2=$!
wait $P1 $P2
say "E9-EASY DONE"
