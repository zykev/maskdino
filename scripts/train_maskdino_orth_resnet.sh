CUDA_VISIBLE_DEVICES=1,2 \
python maskdino_unify.py \
  --task orth \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --config_file configs/default_maskdino_orth_resnet_config.yaml \
  --batch_size 1 \
  --output_dir output/maskdino_orth_resnet \
  --wandb_name "maskdino_orth_resnet"
