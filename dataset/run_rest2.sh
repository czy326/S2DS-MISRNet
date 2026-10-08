#!/bin/bash
# Chained supplement batches, with a MANUAL GATE in front of E2.
#
# E2 (published-baseline re-implementation) costs ~3 h of GPU and is the only batch that
# is purely descriptive.  It is therefore held until the E3 verdict has been read and
# approved: create the file dataset/GO_E2.flag to release it.
#
#   E3 frame shuffle   (already running, 12 runs)
#   E5 permuted cloud  (already DONE)
#   E1 q-route ablation  -> runs automatically
#   E2 baselines         -> waits for dataset/GO_E2.flag
#
# Nothing here edits misr/models.py, misr/train_s2ds.py, misr/eval_s2ds.py or
# dataset/s2ds_dataset.py -- the running lanes re-import those files on every new run.
set -u
D=/mnt/e/论文2/dataset
LOG=$D/pipeline_rest2.log
GATE=$D/GO_E2.flag
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

wait_for () {  # $1 = log file, $2 = marker
  while ! grep -q "$2" "$1" 2>/dev/null; do sleep 60; done
  say "observed marker '$2' in $(basename $1)"
}

say "waiting for E3 (frame shuffle)"
wait_for $D/pipeline_shuffle.log "E3 DONE"
say "waiting for E5-train (permuted cloud)"
wait_for $D/pipeline_permcld_train.log "E5-train DONE"

say "E1 start"
bash $D/run_e1.sh
say "E1 finished"

say "E2 HOLD: waiting for $GATE  (create it to release the 3 h baseline batch)"
while [ ! -f "$GATE" ]; do sleep 60; done
say "gate opened by $(cat $GATE 2>/dev/null || echo unknown)"

say "E2 start"
bash $D/run_e2.sh
say "E2 finished"
say "ALL REMAINING BATCHES DONE"
