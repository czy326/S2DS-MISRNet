#!/bin/bash
# 大修清单 第 5 项 —— 云掩膜翻转噪声鲁棒性（漏检 + 虚警）
# 在已有 3 种子 last.pt 检查点上重推理，不重训。
set -u
cd /mnt/e/论文2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$HOME/basicsr_stub

for seed in 2026 2027 2028; do
  for arm in B D; do
    run="s2dsV2_arm${arm}_s${seed}"
    for rate in 000 050 100 150; do
      f=$(python3 -c "print(${rate}/1000.0)")
      tag="flip${rate}"
      out="runs_s2ds/${run}/scene_psnr_test_hard_${tag}.json"
      if [ -f "$out" ]; then echo "skip $run $tag"; continue; fi
      echo "== $run flip=$f =="
      python3 misr/eval_s2ds.py --run "$run" --split test --mask hard \
        --ckpt last.pt --flip-cld "$f" --tag "$tag" 2>&1 | tail -2
    done
  done
done
echo "ALL DONE"
