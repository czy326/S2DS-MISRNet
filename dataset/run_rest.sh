#!/bin/bash
# Chain the remaining supplement batches after the ones already running.
#
# E3 (frame shuffle, 12 runs) and E5-train (permuted cloud, 3 runs) are already using both
# GPU lanes.  Rather than oversubscribing the card, this script waits for their DONE
# markers and then runs E1 (q-route ablation) followed by E2 (published baselines).
#
# Nothing here edits misr/models.py, misr/train_s2ds.py, misr/eval_s2ds.py or
# dataset/s2ds_dataset.py -- the running lanes re-import those files on every new run, so
# touching them mid-flight would silently change the arms they train.  E1/E2 live in their
# own modules (models_e1.py / baselines_s2ds.py / build_any.py / train_e1.py / eval_e1.py).
set -u
D=/mnt/e/论文2/dataset
LOG=$D/pipeline_rest.log
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
say "E1 finished (rc from run_e1.sh = $?)"

say "E2 start"
bash $D/run_e2.sh
say "E2 finished"
say "ALL REMAINING BATCHES DONE"
