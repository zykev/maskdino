CUDA_VISIBLE_DEVICES=0 

python maskdino_unify.py \
  --task orth \
  --config-file configs/default_maskdino_orth_config.yaml \
  --num-gpus 1 \
  --output-dir output/maskdino_orth_resnet \
