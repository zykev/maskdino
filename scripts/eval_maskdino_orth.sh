CUDA_VISIBLE_DEVICES=2 \
python maskdino_pred.py \
  --task orth \
  --data_dir .datasets/intraoral_anno/orth_test/orth_test \
  --weights output/maskdino_orth_swin/model_final.pth \
  --output_dir output/maskdino_orth_swin_pred \
  --eval_splits train test \
