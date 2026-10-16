#!/usr/bin/env bash
# 训练：Mask R-CNN + caries（实例分割），遍历 5 个编码器变体。
# 配置：configs/maskrcnn/caries/{no_ssl,simclr,swinmae,dentmag_swin,dinov2}.yaml
# OUTPUT_DIR / SWANLAB.NAME 都写在变体 YAML 里，这里不覆盖。
# 可调环境变量见 scripts/common.sh。
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/train_maskrcnn_caries.sh
#   VARIANTS="no_ssl simclr" bash scripts/train/train_maskrcnn_caries.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

for variant in "${VARIANTS[@]}"; do
  run_train maskrcnn caries "${variant}"
done
