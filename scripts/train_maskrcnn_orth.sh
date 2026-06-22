CUDA_VISIBLE_DEVICES=4,5 \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs/default_maskrcnn_orth_config.yaml \
    --input_dir .datasets/intraoral_anno/orth_0616 \
    --output_dir output/maskrcnn_orth_0622 \
    --num_gpus 2 \
    --wandb_name "maskrcnn_orth_0622"
