CUDA_VISIBLE_DEVICES=1,2 \
python maskdino_unify.py \
  --task orth \
  --config_file configs/default_maskdino_orth_swin_config.yaml \
  --num_gpus 2 \
  --output_dir output/maskdino_orth_swin \
  --wandb_name "maskdino_orth_swin"