#!/usr/bin/env bash
# regenerate every figure of the paper and print the layout audit
set -u
cd /mnt/e/论文2/figures || exit 1
for f in make_fig01_dataset.py make_fig02_strata.py make_fig03_architecture.py \
         make_fig04_qualitative.py make_fig05_zerotrain.py make_fig06_forest.py \
         make_fig07_dt_ablation.py make_fig08_ckpt_noise.py ; do
  echo "===== $f"
  python "$f" 2>&1 | grep -E "layout warning|no text overlap|TEXT/|saved |runs plotted|val-best|budget|zoom window|^   [A-Z]|mean \+" || true
done
