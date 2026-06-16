CUDA_VISIBLE_DEVICES=4,5 \
python maskdino_unify.py \
  --task orth \
  --config_file configs/default_maskdino_orth_swin_config.yaml \
  --data_dir .datasets/intraoral_anno/orth_0616/orth_0616 \
  --train_json .datasets/intraoral_anno/orth_0616/orth_detection_train.json \
  --test_json .datasets/intraoral_anno/orth_0616/orth_detection_test.json \
  --num_gpus 2 \
  --output_dir output/maskdino_swin_orth_0616 \
  --wandb_name maskdino_swin_orth_0616