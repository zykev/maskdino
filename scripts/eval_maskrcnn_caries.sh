CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_pred.py \
  --task caries \
  --weights output/maskrcnn_caries/model_final.pth \
  --output_dir output/maskrcnn_caries_pred \
  --eval_splits train test
