CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_pred.py \
  --task orth \
  --weights output/maskrcnn_orth/model_final.pth \
  --output_dir output/maskrcnn_orth_pred \
  --eval_splits train test
