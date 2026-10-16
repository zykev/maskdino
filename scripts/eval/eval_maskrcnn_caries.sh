#!/usr/bin/env bash
# 评估：Mask R-CNN + caries（实例分割），遍历 5 个编码器变体。
# checkpoint 和输出目录都取自配置的 OUTPUT_DIR：
#   configs/maskrcnn/caries/<variant>.yaml -> <OUTPUT_DIR>/model_final.pth
# 可调环境变量见 scripts/common.sh。
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_maskrcnn_caries.sh
#   EVAL_SPLITS=test bash scripts/eval/eval_maskrcnn_caries.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

for variant in "${VARIANTS[@]}"; do
  run_eval maskrcnn caries "${variant}"
done
