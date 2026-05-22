CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_pred.py \
  --task caries \
  --config_file configs/default_maskrcnn_caries_config.yaml \
  --weights output/maskrcnn_caries/model_final.pth \
  --output_dir output/maskrcnn_caries_pred \
  --eval_splits train test
