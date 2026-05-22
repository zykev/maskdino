CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_pred.py \
  --task orth \
  --config_file configs/default_maskrcnn_orth_config.yaml \
  --weights output/maskrcnn_orth/model_final.pth \
  --output_dir output/maskrcnn_orth_pred \
  --eval_splits train test
