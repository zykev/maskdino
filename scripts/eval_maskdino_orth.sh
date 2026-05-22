CUDA_VISIBLE_DEVICES=0 \
python maskdino_pred.py \
  --task orth \
  --data_dir .datasets/intraoral_anno/orth_test/ \
  --config_file configs/default_maskdino_orth_config.yaml \
  --weights output/maskdino_orth_resnet/model_final.pth \
  --output_dir output/maskdino_orth_resnet_pred \
  --eval_splits train test \
