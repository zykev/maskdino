# %%
import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskDINO Imports
from maskdino import add_maskdino_config

from detectron2.engine import default_setup

from maskdino_train import register_caries_dataset, get_caries_dicts

class MaskDINOPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.input_format = cfg.INPUT.FORMAT

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            predictions = self.model(inputs)[0]
            return predictions

# %%
# 初始化配置
def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    
    # 1. 注册数据集，以便配置可以引用 "tooth_train"
    register_caries_dataset()
    
    # # 2. 加载命令行传入的 config (通常是 maskdino 的 yaml)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)

    cfg.OUTPUT_DIR = "output/maskdino_caries_v1"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 3. --- 强制覆盖为 Caries 数据集配置 ---
    cfg.DATASETS.TRAIN = ("tooth_train",)
    cfg.DATASETS.TEST = ("tooth_val",)
    # cfg.DATALOADER.NUM_WORKERS = 4
    
    # # 训练超参数 (根据你的需求调整)
    # cfg.SOLVER.IMS_PER_BATCH = 2  # Batch Size
    # cfg.SOLVER.BASE_LR = 0.00025  # Learning Rate
    # cfg.SOLVER.MAX_ITER = 1000    # Iterations
    # cfg.SOLVER.STEPS = []         # No decay step
    
    # 4. --- 关键：修改类别数量 ---
    # MaskDINO/Mask2Former 架构依赖 SemSegHead.NUM_CLASSES
    # 必须设置为你的实际类别数 (9)
    num_classes = 9
    # 如果 config 中包含 MaskDINO 特有的 key，也一并修改以防万一
    try:
        cfg.MODEL.SemSegHead.NUM_CLASSES = num_classes
        cfg.MODEL.RetinaNet.NUM_CLASSES = num_classes 
        cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    except AttributeError:
        pass
        
    # 为了兼容部分 Evaluator，也设置 ROI_HEADS (尽管 MaskDINO 不直接用它)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.5

    # cfg.freeze()
    # default_setup(cfg, args)
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg

# %%
cfg = setup() # 假设 args 已定义，包含 config_file 和 opts
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = MaskDINOPredictor(cfg)

data_root = '.datasets/intraoral/annosample_ch'
dataset_dicts_val = get_caries_dicts(data_root) # 确保 data_root 已定义
scores_per_class = defaultdict(list)
tooth_metadata = MetadataCatalog.get("tooth_val")
class_names = tooth_metadata.thing_classes

print("Analyzing MaskDINO validation scores...")
for d in dataset_dicts_val:
    im = cv2.imread(d["file_name"])
    if im is None: continue
    
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    
    if instances.has("scores"):
        classes = instances.pred_classes.tolist()
        scores = instances.scores.tolist()
        for cls_id, score in zip(classes, scores):
            scores_per_class[cls_id].append(score)

plt.figure(figsize=(20, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.suptitle("Confidence Score Distribution per Class", fontsize=20)

for i in range(len(class_names)):
    plt.subplot(3, 3, i + 1)
    data = scores_per_class[i]
    
    if len(data) > 0:
        # 绘制直方图
        plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(data), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(data):.2f}')
        plt.title(f"Class: {class_names[i]}\n(Samples: {len(data)})")
    else:
        plt.text(0.5, 0.5, "No Predictions", ha='center', va='center')
        plt.title(f"Class: {class_names[i]}")
        
    plt.xlim(0, 1.0)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(cfg.OUTPUT_DIR, "confidence_score_distribution.png"))


# %%
predictor = MaskDINOPredictor(cfg)

max_samples = 5
cell_size = 300       
padding = 15          
title_width = 250     
bg_color = (40, 40, 40) 
total_width = title_width + (cell_size + padding) * max_samples


class_names = MetadataCatalog.get("tooth_train").thing_classes
num_classes = len(class_names)
category_to_samples = {i: [] for i in range(num_classes)}
for d in dataset_dicts_val:
    for ann in d["annotations"]:
        cat_id = ann["category_id"]
        if cat_id < num_classes:
            if d not in category_to_samples[cat_id]:
                category_to_samples[cat_id].append(d)
for i, name in enumerate(class_names):
    print(f"Category {i} ({name}): Found {len(category_to_samples[i])} samples")

all_category_blocks = []
for cat_id, cat_name in enumerate(class_names):
    samples = category_to_samples[cat_id]
    cat_block_height = cell_size * 2 + padding
    cat_block = np.full((cat_block_height, total_width, 3), bg_color, dtype=np.uint8)
    
    cv2.putText(cat_block, cat_name, (20, cat_block_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if not samples:
        cv2.putText(cat_block, "NO SAMPLES", (title_width + 50, cat_block_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    else:
        selected = random.sample(samples, min(len(samples), max_samples))
        for idx, d in enumerate(selected):
            img_bgr = cv2.imread(d["file_name"])
            x_offset = title_width + idx * (cell_size + padding)
            
            # --- GT ---
            v_gt = Visualizer(img_bgr[:, :, ::-1], metadata=tooth_metadata)
            img_gt = v_gt.draw_dataset_dict(d).get_image()[:, :, ::-1]
            img_gt = cv2.resize(img_gt, (cell_size, cell_size))
            
            # --- MaskDINO Pred ---
            outputs = predictor(img_bgr)

            instances = outputs["instances"].to("cpu")
            
            # --- 关键修改：过滤掉低置信度的预测 ---
            CONF_THRESH = 0.5  # 你可以根据你生成的 Confidence Score Distribution 调整这个值
            if instances.has("scores"):
                # 找出分数大于阈值的索引
                keep = instances.scores > CONF_THRESH
                # 仅保留高分数的 instances
                instances = instances[keep]
            # -------------------------------------

            v_pred = Visualizer(img_bgr[:, :, ::-1], metadata=tooth_metadata)
            img_pred = v_pred.draw_instance_predictions(instances).get_image()[:, :, ::-1]
            img_pred = cv2.resize(img_pred, (cell_size, cell_size))
            
            # 拼接
            cat_block[0:cell_size, x_offset:x_offset+cell_size] = img_gt
            cat_block[cell_size+padding:cell_size*2+padding, x_offset:x_offset+cell_size] = img_pred

    all_category_blocks.append(cat_block)
    all_category_blocks.append(np.full((15, total_width, 3), (20, 20, 20), dtype=np.uint8))

final_grid = np.vstack(all_category_blocks)
cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, "maskdino_comparison_grid.jpg"), final_grid)

# %%
# 确保评估时使用标准测试阈值
evaluator = COCOEvaluator("tooth_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "tooth_val")

# 直接传入 predictor.model
print("Running mAP Evaluation for MaskDINO...")
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)
# %%
