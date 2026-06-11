CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs/default_maskrcnn_orth_config.yaml \
    --output_dir output/maskrcnn_orth \
    --num_gpus 1 \
    --wandb_name "maskrcnn_orth"