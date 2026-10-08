#!/bin/bash
# 大修清单 第 5 项 —— 云掩膜翻转噪声鲁棒性（漏检 + 虚警）
# 在已有 3 种子 last.pt 检查点上重推理，不重训。
# 注意：s2dsV2_armB_s2026 / s2027 的检查点未保留，改用 s2dsR_armB_*（已验证两者
# 逐样本 PSNR 完全一致，max|diff| = 0.000000）。
set -u
cd /mnt/e/论文2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$HOME/basicsr_stub

JOBS="
s2dsR_armB_s2026:0.00
s2dsR_armB_s2026:0.05
s2dsR_armB_s2026:0.10
s2dsR_armB_s2026:0.15
s2dsR_armB_s2027:0.00
s2dsR_armB_s2027:0.05
s2dsR_armB_s2027:0.10
s2dsR_armB_s2027:0.15
s2dsV2_armB_s2028:0.00
s2dsV2_armB_s2028:0.05
s2dsV2_armB_s2028:0.10
s2dsV2_armB_s2028:0.15
s2dsV2_armD_s2026:0.00
s2dsV2_armD_s2026:0.05
s2dsV2_armD_s2026:0.10
s2dsV2_armD_s2026:0.15
s2dsV2_armD_s2027:0.00
s2dsV2_armD_s2027:0.05
s2dsV2_armD_s2027:0.10
s2dsV2_armD_s2027:0.15
s2dsV2_armD_s2028:0.00
s2dsV2_armD_s2028:0.05
s2dsV2_armD_s2028:0.10
s2dsV2_armD_s2028:0.15
"

for job in $JOBS; do
  run="${job%%:*}"
  f="${job##*:}"
  case "$f" in
    0.00) tag="flip000";;
    0.05) tag="flip050";;
    0.10) tag="flip100";;
    0.15) tag="flip150";;
    *) tag="flipxxx";;
  esac
  out="runs_s2ds/${run}/scene_psnr_test_hard_${tag}.json"
  if [ -f "$out" ]; then echo "skip $run $tag"; continue; fi
  echo "== $run flip=$f =="
  python3 misr/eval_s2ds.py --run "$run" --split test --mask hard \
    --ckpt last.pt --flip-cld "$f" --tag "$tag" 2>&1 | tail -3
done
echo "ALL DONE"
