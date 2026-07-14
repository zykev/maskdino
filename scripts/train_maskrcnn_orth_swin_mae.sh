CUDA_VISIBLE_DEVICES=1,2 \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs\default_maskrcnn_orth_swin_mae_config.yaml \
    --input_dir .datasets/intraoral_anno/orth_0616 \
    --output_dir output/maskrcnn_orth_0714 \
    --batch_size 8 \
    --image_subdir tooth_mask \
    --wandb_name maskrcnn_orth_0714
