CUDA_VISIBLE_DEVICES=3,6 \
python maskdino_unify.py \
  --task orth \
  --config_file configs/maskdino/orth/swinmae.yaml \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --batch_size 2 \
  --output_dir output/maskdino_swin_mae_orth_0702 \
  --swanlab_name maskdino_swin_mae_orth_0702
