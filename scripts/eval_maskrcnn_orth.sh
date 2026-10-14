CUDA_VISIBLE_DEVICES=1 \
python maskrcnn_pred.py \
  --task orth \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --weights output/maskrcnn_orth_0707/model_final.pth \
  --output_dir output/maskrcnn_orth_pred \
  --image_subdir tooth_mask \
  --eval_splits train test
