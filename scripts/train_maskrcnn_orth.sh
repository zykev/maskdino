CUDA_VISIBLE_DEVICES=3 \
python maskrcnn_unify.py \
    --task orth \
    --config_file configs/default_maskrcnn_orth_config.yaml \
    --data_dir .datasets/intraoral_anno/orth_0616/orth_0616 \
    --train_json .datasets/intraoral_anno/orth_0616/orth_detection_train.json \
    --test_json .datasets/intraoral_anno/orth_0616/orth_detection_test.json \
    --output_dir output/maskrcnn_orth_0616 \
    --num_gpus 1 \
    --wandb_name "maskrcnn_orth_0616"