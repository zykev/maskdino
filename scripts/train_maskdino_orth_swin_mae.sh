#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/default_maskdino_orth_swin_mae_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-output/maskdino_orth_swin_mae}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"
LOW_MEMORY="${LOW_MEMORY:-1}"
IMS_PER_BATCH="${IMS_PER_BATCH:-1}"
MIN_SIZE_TRAIN="${MIN_SIZE_TRAIN:-512}"
MAX_SIZE_TRAIN="${MAX_SIZE_TRAIN:-896}"
MIN_SIZE_TEST="${MIN_SIZE_TEST:-640}"
MAX_SIZE_TEST="${MAX_SIZE_TEST:-896}"
NUM_OBJECT_QUERIES="${NUM_OBJECT_QUERIES:-30}"
DEC_LAYERS="${DEC_LAYERS:-6}"
ENC_LAYERS="${ENC_LAYERS:-3}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -ra SELECTED_GPUS <<< "${GPU_IDS}"
  NUM_GPUS="${#SELECTED_GPUS[@]}"
fi

EXTRA_OPTS=()
if [[ -n "${MODEL_WEIGHTS}" ]]; then
  EXTRA_OPTS+=(MODEL.WEIGHTS "${MODEL_WEIGHTS}")
fi
if [[ "${LOW_MEMORY}" == "1" ]]; then
  EXTRA_OPTS+=(
    SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}"
    INPUT.MIN_SIZE_TRAIN "[${MIN_SIZE_TRAIN}]"
    INPUT.MAX_SIZE_TRAIN "${MAX_SIZE_TRAIN}"
    INPUT.MIN_SIZE_TEST "${MIN_SIZE_TEST}"
    INPUT.MAX_SIZE_TEST "${MAX_SIZE_TEST}"
    MODEL.MaskDINO.NUM_OBJECT_QUERIES "${NUM_OBJECT_QUERIES}"
    MODEL.MaskDINO.DEC_LAYERS "${DEC_LAYERS}"
    MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS "${ENC_LAYERS}"
  )
fi

python maskdino_unify.py \
  --task orth \
  --config_file "${CONFIG_FILE}" \
  --num_gpus "${NUM_GPUS}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_OPTS[@]}" \
  "$@"
