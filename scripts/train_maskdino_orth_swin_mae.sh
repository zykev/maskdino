CUDA_VISIBLE_DEVICES=4,5 \
python maskdino_unify.py \
  --task orth \
  --config_file configs/default_maskdino_orth_swin_config.yaml \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --num_gpus 2 \
  --output_dir output/maskdino_swin_orth_0616 \
  --wandb_name maskdino_swin_orth_0616
