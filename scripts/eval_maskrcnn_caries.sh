CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_pred.py \
  --task caries \
  --input_dir .datasets/intraoral_anno/single_ch_0225 \
  --weights output/maskrcnn_caries/model_final.pth \
  --output_dir output/maskrcnn_caries_pred \
  --eval_splits train test
