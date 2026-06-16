#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  IFS=',' read -ra SELECTED_GPUS <<< "${GPU_IDS}"
  NUM_GPUS="${#SELECTED_GPUS[@]}"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
python maskrcnn_unify.py \
    --task caries \
    --input_dir .datasets/intraoral_anno/single_ch_0225 \
    --config_file configs/default_maskrcnn_caries_config.yaml \
    --output_dir output/maskrcnn_caries \
    --num_gpus "${NUM_GPUS}" \
    --wandb_name "maskrcnn_caries"
