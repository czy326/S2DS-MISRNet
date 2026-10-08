#!/bin/bash
# Long-budget confirmation of the 2x2 interaction (screening at 4000 iters was NOT
# converged: val still swinging +/-0.5 dB and the model was below the trivial baseline).
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
D=$R/dataset
ITERS=30000
lane () {
  local ARMS=$1
  for SEED in 2026 2027; do
    for ARM in $ARMS; do
      local NAME="s2dsL_arm${ARM}_s${SEED}"
      echo "=== long train start $NAME $(date +%H:%M:%S) ===" >> $D/pipeline_long.log
      $PY -u $R/misr/train_s2ds.py --arm $ARM --seed $SEED --iters $ITERS --batch 8 \
          --eval-every 3000 --out $NAME >> $D/pipeline_long.log 2>&1
      echo "=== long train end $NAME rc=$? $(date +%H:%M:%S) ===" >> $D/pipeline_long.log
      for sp in val test; do
        $PY -u $R/misr/eval_s2ds.py --run $NAME --split $sp >> $D/pipeline_long.log 2>&1
      done
    done
  done
  echo "=== lane($ARMS) done $(date +%H:%M:%S) ===" >> $D/pipeline_long.log
}
lane "A C" &
P1=$!
lane "B D" &
P2=$!
wait $P1 $P2
for SEED in 2026 2027; do
  for sp in val test; do
    $PY -u $R/misr/analyze_s2ds_2x2.py --seed $SEED --split $sp --prefix "s2dsL_arm%s_s%d" >> $D/pipeline_long.log 2>&1
  done
done
$PY -u $D/write_verdict.py >> $D/pipeline_long.log 2>&1
echo "=== LONG PIPELINE DONE $(date +%H:%M:%S) ===" >> $D/pipeline_long.log
