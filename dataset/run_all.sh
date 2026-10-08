#!/bin/bash
# Master pipeline: wait for collection -> build -> baselines -> 2x2 interaction -> report.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
D=/mnt/e/论文2/dataset
R=/mnt/e/论文2
LOG=$D/pipeline.log
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

say "pipeline start"
n=0
while pgrep -f "[c]ollect_s2" > /dev/null; do
  n=$((n+1))
  if [ $((n % 15)) -eq 0 ]; then say "waiting for collectors (alive=$(pgrep -fc "[c]ollect_s2"))"; fi
  sleep 60
done
say "collectors finished"

$PY -u $D/build_dataset.py --previews all >> $LOG 2>&1
say "build rc=$?"
$PY -u $D/bench_baselines.py --splits val,test >> $LOG 2>&1
say "bench rc=$?"
$PY -u $D/make_report.py >> $LOG 2>&1
say "report rc=$?"

train_lane () {
  local LANE=$1
  local ARM1 ARM2
  if [ "$LANE" == "1" ]; then ARM1=A; ARM2=C; else ARM1=B; ARM2=D; fi
  for SEED in 2026 2027; do
    for ARM in $ARM1 $ARM2; do
      local NAME="s2ds_arm${ARM}_s${SEED}"
      say "train start $NAME"
      $PY -u $R/misr/train_s2ds.py --arm $ARM --seed $SEED --iters 4000 --batch 8 \
          --eval-every 1000 --out $NAME >> $LOG 2>&1
      say "train end $NAME rc=$?"
      for sp in val test; do
        $PY -u $R/misr/eval_s2ds.py --run $NAME --split $sp >> $LOG 2>&1
      done
      say "eval end $NAME"
    done
  done
  say "lane$LANE done"
}
export -f train_lane 2>/dev/null || true
train_lane 1 &
P1=$!
train_lane 2 &
P2=$!
wait $P1 $P2
say "all training lanes done"

for SEED in 2026 2027; do
  for sp in val test; do
    $PY -u $R/misr/analyze_s2ds_2x2.py --seed $SEED --split $sp >> $LOG 2>&1
  done
done
$PY -u $D/write_verdict.py >> $LOG 2>&1
say "verdict written rc=$?"
say "PIPELINE DONE"
