CUDA_VISIBLE_DEVICES=1,2 \
python maskdino_unify.py \
  --task orth \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --config_file configs/maskdino/orth/no_ssl.yaml \
  --batch_size 1 \
  --output_dir output/maskdino_orth_resnet \
  --swanlab_name "maskdino_orth_resnet"
