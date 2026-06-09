CUDA_VISIBLE_DEVICES=0 \
python maskdino_pred.py \
  --task orth \
  --data_dir .datasets/intraoral_anno/orth_test/orth_test \
  --weights output/maskdino_orth_resnet/model_final.pth \
  --output_dir output/maskdino_orth_resnet_pred \
  --eval_splits train test \
  DATALOADER.NUM_WORKERS 0
