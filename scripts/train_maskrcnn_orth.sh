#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-3}"
NUM_GPUS="${NUM_GPUS:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  IFS=',' read -ra SELECTED_GPUS <<< "${GPU_IDS}"
  NUM_GPUS="${#SELECTED_GPUS[@]}"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs/default_maskrcnn_orth_config.yaml \
    --data_dir .datasets/intraoral_anno/orth_0616/orth_0616 \
    --train_json .datasets/intraoral_anno/orth_0616/orth_detection_train.json \
    --test_json .datasets/intraoral_anno/orth_0616/orth_detection_test.json \
    --output_dir output/maskrcnn_orth_0616 \
    --num_gpus "${NUM_GPUS}" \
    --wandb_name "maskrcnn_orth_0616"
