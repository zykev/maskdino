CUDA_VISIBLE_DEVICES=2 \
python maskdino_pred.py \
  --task orth \
  --weights output/maskdino_swin_ort_0616/model_final.pth \
  --output_dir output/maskdino_swin_orth_pred \
  --eval_splits train test \
