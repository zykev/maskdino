CUDA_VISIBLE_DEVICES=1,2 \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs/maskrcnn/orth/swinmae.yaml \
    --input_dir .datasets/intraoral_anno/orth_0616 \
    --output_dir output/maskrcnn_orth_0714 \
    --batch_size 4 \
    --image_subdir tooth_mask \
    --swanlab_name maskrcnn_orth_0714
