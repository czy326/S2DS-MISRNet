#!/bin/bash
# E5 (step 0, zero training cost): evaluate the ALREADY TRAINED arm B with the per-pixel
# cloud pattern scrambled inside every frame (8x8 block permutation).  Frame-level clear
# fraction is preserved, spatial correspondence with the image content is destroyed.
#
# arm A does not consume cld at all, so its PSNR is unchanged; the q gain under the
# control is  B_permuted - A_original.  Pre-registered expectation: the gain drops to
# <= 1/2 of the intact value  =>  q really exploits per-pixel structure.
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
R=/mnt/e/论文2
LOG=$R/dataset/pipeline_permcld.log
say(){ echo "=== $* $(date +%m-%d_%H:%M:%S) ===" >> $LOG; }

say "E5 start (permuted-cloud negative control, arm B, fixed-iter ckpt)"
for S in 2026 2027 2028; do
  for M in hard valid; do
    $PY -u $R/misr/eval_s2ds.py --run "s2dsV2_armB_s${S}" --split test --mask "$M" \
        --ckpt last.pt --permute-cld 8 --tag permcld >> $LOG 2>&1
    echo "done B s$S $M rc=$?" >> $LOG
  done
done
say "E5 evals done"
