#!/usr/bin/env bash
# 评估：MaskDINO + orth（只做牙位检测框），遍历 5 个编码器变体。
# checkpoint 和输出目录都取自配置的 OUTPUT_DIR：
#   configs/maskdino/orth/<variant>.yaml -> <OUTPUT_DIR>/model_final.pth
# 可调环境变量见 scripts/common.sh。
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_maskdino_orth.sh
#   EVAL_SPLITS=test IMAGE_SUBDIR=tooth_mask bash scripts/eval/eval_maskdino_orth.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

for variant in "${VARIANTS[@]}"; do
  run_eval maskdino orth "${variant}"
done
