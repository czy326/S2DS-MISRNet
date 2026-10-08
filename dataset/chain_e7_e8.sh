#!/bin/bash
# Wait for the E7 batch to finish, then run the E8 hyper-parameter sensitivity batch.
#
# Both batches are long (E7 ~1.5 h, E8 ~2.5 h) and each has a resume guard inside its own
# run_one, so this chain is idempotent: re-running it after an interruption picks up exactly
# where it stopped.  The caller must HOLD this script in the foreground (WorkBuddy's Bash
# tool with run_in_background), never `setsid nohup` -- on this box a detached WSL job is
# silently reaped as soon as wsl.exe returns.
set -u
D=/mnt/e/论文2/dataset
LOG=$D/pipeline_chain.log
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

say "chain start"
n=0
while ! grep -q "E7 training + eval done" $D/pipeline_e7.log 2>/dev/null; do
  n=$((n + 1))
  if [ $n -gt 720 ]; then
    say "gave up waiting for E7 after $n checks"
    exit 1
  fi
  sleep 60
done
say "E7 done, starting E8"
bash $D/run_e8.sh
say "chain done"
