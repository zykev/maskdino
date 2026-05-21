#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/default_maskdino_orth_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-output/maskdino_orth_resnet}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -ra SELECTED_GPUS <<< "${GPU_IDS}"
  NUM_GPUS="${#SELECTED_GPUS[@]}"
fi

EXTRA_OPTS=()
if [[ -n "${MODEL_WEIGHTS}" ]]; then
  EXTRA_OPTS+=(MODEL.WEIGHTS "${MODEL_WEIGHTS}")
fi

python maskdino_unify.py \
  --task orth \
  --config-file "${CONFIG_FILE}" \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_OPTS[@]}" \
  "$@"
