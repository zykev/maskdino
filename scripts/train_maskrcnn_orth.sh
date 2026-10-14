CUDA_VISIBLE_DEVICES=3,4 \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs/maskrcnn/orth/no_ssl.yaml \
    --input_dir .datasets/intraoral_anno/orth_0616 \
    --output_dir output/maskrcnn_orth_0707 \
    --batch_size 8 \
    --image_subdir tooth_mask \
    --swanlab_name maskrcnn_orth_0707
