python maskdino_train.py \
  --config-file configs/coco/instance-segmentation/maskdino_R50_caries.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS ".checkpoints/intraoral/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"