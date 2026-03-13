# %% [1] 环境初始化与配置
import torch
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from collections import defaultdict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode

from maskrcnn import get_caries_dicts



# 1. 基本参数设置
data_root = ".datasets/intraoral/single_ch_0225"
output_dir = "output/maskrcnn_caries_v3_pred" # 训练时定义的输出目录
os.makedirs(output_dir, exist_ok=True)

model_weights = os.path.join("output/maskrcnn_caries_v3", "model_final.pth") # 预训练权重路径
CLASS_NAMES = [
    "caries", "white_spot_lesion", "filling_no_caries", 
    "filling_with_caries", "fissure_sealant", "non_caries_hard_tissue", 
    "staining", "abnormal_central_cusp", "palatal_radicular_groove"
]

# 2. 重新注册验证集 (确保环境干净)
def register_tooth_datasets():
    datasets_to_reg = ["tooth_train", "tooth_val"]
    
    for dataset_name in datasets_to_reg:
        # 如果已存在，先移除，确保环境干净
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)
        
        # 注册数据集
        if "train" in dataset_name:
            DatasetCatalog.register(dataset_name, lambda: get_caries_dicts(data_root, is_train=True, in_eval=True))
        else:
            DatasetCatalog.register(dataset_name, lambda: get_caries_dicts(data_root, is_train=False, in_eval=True))
        # 显式设置类别名称，防止 ID 映射错乱
        MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES)

# 执行注册
register_tooth_datasets()



# 3. 加载配置与模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4, 8, 16, 32, 64]]

# 2. 修正输入尺寸 (Input Size)
cfg.INPUT.MIN_SIZE_TRAIN = (300, 400, 500)
cfg.INPUT.MAX_SIZE_TRAIN = 600
cfg.INPUT.MIN_SIZE_TEST = 400

cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3, 0.45]
cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, -1, 1] 
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.2, 0.6]

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # 修改为你的实际类别数：9
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5     # 确保正样本占一半

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
cfg.MODEL.WEIGHTS = model_weights # 关键：加载你跑好的模型
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 推理阈值
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader



import shutil

# 1. 定义评估函数，强制隔离环境
def safe_evaluate(model, dataset_name, cfg, base_output_dir):
    # 每次评估前创建一个独立的子目录
    eval_path = os.path.join(base_output_dir, f"eval_{dataset_name}")
    if os.path.exists(eval_path):
        shutil.rmtree(eval_path) # 清理旧的缓存文件
    os.makedirs(eval_path)
    
    evaluator = COCOEvaluator(dataset_name, cfg, True, output_dir=eval_path)
    loader = build_detection_test_loader(cfg, dataset_name)
    
    print(f"--- Running evaluation on {dataset_name} ---")
    return inference_on_dataset(model, loader, evaluator)

# 2. 调用
# 评估验证集
res_val = safe_evaluate(predictor.model, "tooth_val", cfg, os.path.join(output_dir, "pred_val"))
# 评估训练集
# %%
res_train = safe_evaluate(predictor.model, "tooth_train", cfg, os.path.join(output_dir, "pred_train"))
# %%
