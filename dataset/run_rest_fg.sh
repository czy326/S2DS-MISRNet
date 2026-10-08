#!/bin/bash
# Foreground-held chain: E1 (automatic) -> E2 (released by dataset/GO_E2.flag).
#
# Why foreground: on this machine `setsid nohup ... &` inside a `wsl.exe -e bash -c`
# call does NOT survive -- wsl.exe returns and the children are reaped.  The chain is
# therefore run in the foreground and held by the caller (Bash tool / run_in_background),
# which keeps a WSL session open for the whole batch.
#
# Resume-safe: run_e1.sh / run_e2.sh both skip runs whose fixed-endpoint per-scene
# outputs already exist, so a restart never redoes finished work.
set -u
D=/mnt/e/论文2/dataset
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ==="; }

say "E1 start (q-route ablation B1/B2 x 3 seeds)"
bash $D/run_e1.sh
say "E1 finished"

say "E2 HOLD: waiting for $D/GO_E2.flag (create it to release the ~3 h baseline batch)"
while [ ! -f $D/GO_E2.flag ]; do sleep 60; done
say "gate opened"

say "E2 start (highresnet x3 || rams x1)"
bash $D/run_e2.sh
say "ALL REMAINING BATCHES DONE"
