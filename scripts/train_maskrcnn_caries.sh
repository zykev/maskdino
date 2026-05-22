CUDA_VISIBLE_DEVICES=0 \
python maskrcnn_unify.py \
    --task caries \
    --config_file configs/default_maskrcnn_caries_config.yaml \
    --output_dir output/maskrcnn_caries \
    --num_gpus 1 \