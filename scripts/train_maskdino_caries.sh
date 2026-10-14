#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 \
python maskdino_unify.py \
  --task caries \
  --input_dir .datasets/intraoral_anno/single_ch_0225 \
  --config_file configs/maskdino/caries/no_ssl.yaml
