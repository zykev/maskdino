# %%
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import matplotlib.pyplot as plt
from collections import defaultdict



# %%
def get_caries_dicts(root_dir, is_train=False, keep_healthy_ratio=0.1):
    """
    img_dir: 包含 single_tooth 文件夹的根目录
    json_path: 转换后的 coco json 文件路径
    """
    if is_train:
        json_path = os.path.join(root_dir, "caries_sample_dataset_train.json")
    else:
        json_path = os.path.join(root_dir, "caries_sample_dataset_test.json")
    
    if not os.path.exists(json_path):
        json_path = os.path.join(root_dir, "caries_sample_dataset.json")

    with open(json_path) as f:
        coco_data = json.load(f)

    # 建立 image_id 到 image 信息的映射，方便快速查找
    images = {img["id"]: img for img in coco_data["images"]}
    
    # 建立 image_id 到 annotations 的映射 (一张图可能有多个标注)
    img_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    dataset_dicts = []
    
    # 遍历所有图片（包括健康的牙齿）
    for img_id, img_info in images.items():

        # 获取该图片对应的标注，如果没有（健康牙齿），anns 为空列表 []
        anns = img_to_anns.get(img_id, [])

        # --- 新增：健康样本降采样逻辑 ---
        if len(anns) == 0:
            # 如果是训练集，且是健康图片，按概率决定是否保留
            if is_train and random.random() > keep_healthy_ratio:
                continue
        # ------------------------------
        record = {}
        
        # 拼接图片完整路径    
        record["file_name"] = img_info["file_name"]
        record["image_id"] = img_id
        record["height"] = img_info["height"]
        record["width"] = img_info["width"]
      
        objs = []
        for ann in anns:
            # COCO 的 bbox 是 [x, y, w, h]，而 Detectron2 常用 XYXY_ABS [x1, y1, x2, y2]
            # 我们根据转换时的定义进行还原
            x, y, w, h = ann["bbox"]
            
            obj = {
                "bbox": [x, y, x + w, y + h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann["segmentation"], # 已经是 [[x1, y1, x2, y2, ...]] 格式
                "category_id": ann["category_id"] - 1, # 重要：Detectron2 内部类别从 0 开始映射
            }
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts


# %%
# 1. 定义类别名称（顺序必须与 category_id - 1 对应）
CLASS_NAMES = [
    "caries", "white_spot_lesion", "filling_no_caries", 
    "filling_with_caries", "fissure_sealant", "non_caries_hard_tissue", 
    "staining", "abnormal_central_cusp", "palatal_radicular_groove"
]

# 2. 注册数据集
# 假设你的目录结构是 .datasets/intraoral/annosample_ch/train 和 val
data_root = ".datasets/intraoral/single_ch_0225"

for d in ["train", "val"]:
    dataset_name = "tooth_" + d
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(dataset_name)
    
    # 区分训练集和验证集
    is_train_set = (d == "train")
    DatasetCatalog.register(
        dataset_name, 
        lambda is_t=is_train_set: get_caries_dicts(data_root, is_train=is_t, keep_healthy_ratio=0.1)
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES)

tooth_metadata = MetadataCatalog.get("tooth_train")

# 3. 配置训练参数
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("tooth_train",)
cfg.DATASETS.TEST = ("tooth_val",) # 这样可以在训练中进行评估
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 3000    # 9个类别需要更长的迭代，建议至少 3000+
cfg.SOLVER.STEPS = []        

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # 修改为你的实际类别数：9
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# --- 新增：处理类别不平衡的核心配置 ---

# 允许 Dataloader 加载没有标注的图片（即我们保留的那 10% 健康牙齿，作为负样本）
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False 

# 启用重复因子采样器 (Repeat Factor Sampler)
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"

# 设置重复阈值。频率低于这个比例的类别（如 Caries, Filling）会被上采样（即在同一个 Epoch 中多次出现）
# 0.05 到 0.1 是常见的经验值，你可以根据类别频率分布进行微调
cfg.DATALOADER.REPEAT_THRESHOLD = 0.05 
# ------------------------------------

cfg.OUTPUT_DIR = "output/maskrcnn_caries_v2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# %%
# 4. 开始训练
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# %%
# 5. 开始测试
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)
dataset_dicts_val = DatasetCatalog.get("tooth_val")

# %%
# distribution of confidence scores for each class in the validation set

scores_per_class = defaultdict(list)
class_names = MetadataCatalog.get("tooth_train").thing_classes

print("Analyzing validation set scores... this may take a minute.")
# 建议遍历完整的验证集以获得准确分布
for d in dataset_dicts_val:
    im = cv2.imread(d["file_name"])
    if im is None: continue
    
    with torch.no_grad():
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        
        # 提取类别 ID 和对应的分数
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
# 5. 推理与可视化
class_names = MetadataCatalog.get("tooth_train").thing_classes
num_classes = len(class_names)
max_samples = 5
cell_size = 300       
padding = 15          
title_width = 250     
text_font_scale = 0.4 
bg_color = (40, 40, 40) 

total_width = title_width + (cell_size + padding) * max_samples

category_to_samples = {i: [] for i in range(num_classes)}
for d in dataset_dicts_val:
    for ann in d["annotations"]:
        cat_id = ann["category_id"]
        if cat_id < num_classes:
            if d not in category_to_samples[cat_id]:
                category_to_samples[cat_id].append(d)
for i, name in enumerate(class_names):
    print(f"Category {i} ({name}): Found {len(category_to_samples[i])} samples")

def add_label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                text_font_scale, color, 1, cv2.LINE_AA)
    return img

all_category_blocks = []

for cat_id, cat_name in enumerate(class_names):
    samples = category_to_samples[cat_id]
    
    # 创建这一类别的双行容器块
    cat_block_height = cell_size * 2 + padding
    cat_block = np.full((cat_block_height, total_width, 3), bg_color, dtype=np.uint8)

    # 绘制左侧类别标题
    cv2.putText(cat_block, cat_name, (20, cat_block_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if not samples:
        cv2.putText(cat_block, "NO SAMPLES IN VAL SET", (title_width + 50, cat_block_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    else:
        # 随机抽取样本
        selected = random.sample(samples, min(len(samples), max_samples))
        
        for idx, d in enumerate(selected):
            img_bgr = cv2.imread(d["file_name"])
            x_offset = title_width + idx * (cell_size + padding)
            
            # --- 处理 GT (上行) ---
            v_gt = Visualizer(img_bgr[:, :, ::-1], metadata=tooth_metadata, scale=0.5)
            img_gt = v_gt.draw_dataset_dict(d).get_image()[:, :, ::-1]
            img_gt = cv2.resize(img_gt, (cell_size, cell_size))
            img_gt = add_label(img_gt, "GT", (0, 255, 0))
            
            # --- 处理 Prediction (下行) ---
            outputs = predictor(img_bgr)
            v_pred = Visualizer(img_bgr[:, :, ::-1], metadata=tooth_metadata, scale=0.5)
            img_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
            img_pred = cv2.resize(img_pred, (cell_size, cell_size))
            img_pred = add_label(img_pred, f"PRED (th=0.5)", (0, 0, 255))
            
            # 填入容器
            cat_block[0:cell_size, x_offset : x_offset+cell_size] = img_gt
            cat_block[cell_size + padding : cell_size*2 + padding, x_offset : x_offset+cell_size] = img_pred

    all_category_blocks.append(cat_block)
    
    # 类别之间的分割线
    separator = np.full((15, total_width, 3), (20, 20, 20), dtype=np.uint8)
    all_category_blocks.append(separator)

final_master_grid = np.vstack(all_category_blocks)
save_path = os.path.join(cfg.OUTPUT_DIR, "res_comparison_grid.jpg")
cv2.imwrite(save_path, final_master_grid)


# %%
# 6. 性能评估 (mAP)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("tooth_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "tooth_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))


# %%
vis_save_dir = os.path.join(cfg.OUTPUT_DIR, "custom_visualizations")
os.makedirs(vis_save_dir, exist_ok=True)

class_names = MetadataCatalog.get("tooth_train").thing_classes
num_classes = len(class_names)
max_samples = 5

# 定义一套固定的颜色表 (BGR格式，用于 OpenCV)
# 确保各个类别的颜色区分度高
COLORS = [
    (0, 0, 255),     # caries: 红色
    (0, 255, 255),   # white_spot_lesion: 黄色
    (255, 0, 0),     # filling_no_caries: 蓝色
    (255, 0, 255),   # filling_with_caries: 紫色
    (0, 255, 0),     # fissure_sealant: 绿色
    (255, 255, 0),   # non_caries_hard_tissue: 青色
    (128, 0, 128),   # staining: 深紫
    (0, 128, 255),   # abnormal_central_cusp: 橙色
    (128, 128, 0)    # palatal_radicular_groove: 深青
]

# 按类别收集所有样本
category_to_samples = {i: [] for i in range(num_classes)}
for d in dataset_dicts_val:
    for ann in d["annotations"]:
        cat_id = ann["category_id"]
        if cat_id < num_classes:
            if d not in category_to_samples[cat_id]:
                category_to_samples[cat_id].append(d)

for i, name in enumerate(class_names):
    print(f"Category {i} ({name}): Found {len(category_to_samples[i])} samples")

# --- 自定义绘制函数：Ground Truth ---
def draw_custom_gt(image_bgr, annotations, height, width):
    vis_img = image_bgr.copy()
    for ann in annotations:
        cls_id = ann["category_id"]
        color = COLORS[cls_id % len(COLORS)]
        
        # 1. 绘制矩形框
        x1, y1, x2, y2 = map(int, ann["bbox"])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # 2. 绘制分割掩码 (半透明)
        segm = ann["segmentation"]
        if isinstance(segm, list): # 处理多边形格式
            for poly in segm:
                poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_pts], 1)
                bool_mask = mask.astype(bool)
                # Alpha 融合 (0.5 透明度)
                vis_img[bool_mask] = vis_img[bool_mask] * 0.5 + np.array(color) * 0.5
    return vis_img

# --- 自定义绘制函数：Predictions ---
def draw_custom_pred(image_bgr, instances, conf_thresh=0.5):
    vis_img = image_bgr.copy()
    
    # 1. 拦截空预测：如果模型什么都没预测出来，直接返回原图
    if len(instances) == 0:
        return vis_img
        
    # 2. 拦截缺少 Box 的情况
    if not instances.has("pred_boxes"):
        return vis_img
        
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    
    # 安全获取 scores
    if instances.has("scores"):
        scores = instances.scores.cpu().numpy()
    else:
        scores = np.ones(len(boxes)) # 防御性编程：如果没有 scores 默认全通过
        
    # 3. 安全获取 masks
    has_masks = instances.has("pred_masks")
    if has_masks:
        masks = instances.pred_masks.cpu().numpy()
    
    # 4. 遍历并绘制
    for i in range(len(boxes)):
        if scores[i] < conf_thresh: # 过滤低置信度预测
            continue
            
        cls_id = classes[i]
        color = COLORS[cls_id % len(COLORS)]
        
        # 绘制矩形框
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 绘制分割掩码 (半透明)
        if has_masks:
            mask = masks[i]
            # 确保 mask 是 boolean 格式
            if mask.dtype != bool:
                mask = mask > 0.5 
            vis_img[mask] = vis_img[mask] * 0.5 + np.array(color) * 0.5
            
    return vis_img

# --- 主保存循环 ---
for cat_id, cat_name in enumerate(class_names):
    samples = category_to_samples[cat_id]
    if not samples:
        continue
        
    # 为当前类别创建一个独立的文件夹
    cat_dir = os.path.join(vis_save_dir, cat_name)
    os.makedirs(cat_dir, exist_ok=True)
    
    # 根据之前设定的随机种子抽取固定样本
    selected = random.sample(samples, min(len(samples), max_samples))
    
    for idx, d in enumerate(selected):
        img_path = d["file_name"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        height, width = img_bgr.shape[:2]
        
        # 1. 保存原图 (Original)
        orig_save_path = os.path.join(cat_dir, f"sample_{idx}_0_orig.jpg")
        cv2.imwrite(orig_save_path, img_bgr)
        
        # 2. 生成并保存 Ground Truth (GT)
        img_gt = draw_custom_gt(img_bgr, d["annotations"], height, width)
        gt_save_path = os.path.join(cat_dir, f"sample_{idx}_1_gt.jpg")
        cv2.imwrite(gt_save_path, img_gt)
        
        # 3. 生成并保存 Predictions (Pred)
        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")
        img_pred = draw_custom_pred(img_bgr, instances, conf_thresh=0.5)
        pred_save_path = os.path.join(cat_dir, f"sample_{idx}_2_pred.jpg")
        cv2.imwrite(pred_save_path, img_pred)

print(f"\n✅ 所有的独立可视化图片已成功保存至: {vis_save_dir}")
# %%
