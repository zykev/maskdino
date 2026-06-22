CUDA_VISIBLE_DEVICES=4 \
python maskrcnn_pred.py \
  --task orth \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --weights output/maskrcnn_orth_0622/model_final.pth \
  --output_dir output/maskrcnn_orth_pred \
  --eval_splits train test
