#!/bin/bash
# Waits for all collect_s2.py processes to exit, then builds the dataset, benchmarks the
# baselines and writes the report.  Idempotent: build skips AOIs already in the manifest.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
D=/mnt/e/论文2/dataset
LOG=$D/after_collect.log
echo "=== driver start $(date) ===" >> $LOG
n=0
while pgrep -f "[c]ollect_s2.py" > /dev/null; do
  n=$((n+1))
  if [ $((n % 10)) -eq 0 ]; then echo "  waiting... $(date) collectors=$(pgrep -fc "[c]ollect_s2.py")" >> $LOG; fi
  sleep 60
done
echo "=== collectors finished $(date) ===" >> $LOG
$PY -u $D/build_dataset.py --previews all >> $LOG 2>&1
echo "=== build rc=$? $(date) ===" >> $LOG
$PY -u $D/bench_baselines.py --splits val,test >> $LOG 2>&1
echo "=== bench rc=$? $(date) ===" >> $LOG
$PY -u $D/make_report.py >> $LOG 2>&1
echo "=== report rc=$? $(date) ===" >> $LOG
echo "=== ALL DONE $(date) ===" >> $LOG
