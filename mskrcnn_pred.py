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
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 推理阈值
# predictor = DefaultPredictor(cfg)

# 获取验证集数据
dataset_dicts_val = DatasetCatalog.get("tooth_val")
print(f"Loaded {len(dataset_dicts_val)} validation images.")

# %% [2] 统计置信度分布 (Score Distribution)
scores_per_class = defaultdict(list)
gt_counts_per_class = defaultdict(int) # 用于存储 GT 的样本数量
class_names = MetadataCatalog.get("tooth_val").thing_classes

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # 推理阈值
predictor = DefaultPredictor(cfg)
print("Analyzing validation set scores... this may take a minute.")
# 1. 首先统计验证集中的 GT 样本分布
for d in dataset_dicts_val:
    for ann in d.get("annotations", []):
        cat_id = ann["category_id"]
        gt_counts_per_class[cat_id] += 1
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
    pred_data = scores_per_class[i]
    gt_size = gt_counts_per_class[i] # 获取当前类别的 GT 数量
    caption = f"Class: {class_names[i]}\n(GT Size: {gt_size} | Preds: {len(pred_data)})"
    if len(pred_data) > 0:
        # 绘制直方图
        plt.hist(pred_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(pred_data), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(pred_data):.2f}')
        plt.title(caption)
    else:
        plt.text(0.5, 0.5, "No Predictions", ha='center', va='center')
        plt.title(f"Class: {class_names[i]}\n(GT Size: {gt_size}")
        
    plt.xlim(0, 1.0)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "confidence_score_distribution.png"))

# %% iou stat
from detectron2.structures import Boxes, pairwise_iou

# --- 1. 初始化统计字典 ---
iou_stats_per_class = defaultdict(list)
class_names = MetadataCatalog.get("tooth_val").thing_classes

print("开始计算 IoU 统计数据... 请稍候。")

# 确保阈值较低，以便观察模型原始的定位能力
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # 推理阈值
predictor = DefaultPredictor(cfg)

# --- 2. 遍历验证集 ---
for d in dataset_dicts_val:
    img = cv2.imread(d["file_name"])
    if img is None: continue
    
    with torch.no_grad():
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        # 提取预测信息
        pred_boxes = instances.pred_boxes
        pred_classes = instances.pred_classes.tolist()
        
        # 提取 GT 信息
        gt_boxes = [ann["bbox"] for ann in d["annotations"]]
        gt_classes = [ann["category_id"] for ann in d["annotations"]]
        
        if len(gt_boxes) == 0:
            continue # 跳过没有标注的图片

        # 按类别进行匹配计算
        for cls_id in range(len(class_names)):
            # 筛选当前类别的预测和 GT
            curr_p_idx = [i for i, c in enumerate(pred_classes) if c == cls_id]
            curr_g_idx = [i for i, c in enumerate(gt_classes) if c == cls_id]
            
            # 如果该类既有预测又有 GT，计算 IoU
            if len(curr_p_idx) > 0 and len(curr_g_idx) > 0:
                p_boxes = pred_boxes[curr_p_idx]
                g_boxes = Boxes(torch.as_tensor([gt_boxes[i] for i in curr_g_idx]))
                
                # 计算两两之间的 IoU 矩阵 [M, N]
                ious = pairwise_iou(p_boxes, g_boxes)
                
                # 对每个预测框，找到它与所有 GT 框中最大的那个 IoU
                max_ious = ious.max(dim=1)[0].tolist()
                iou_stats_per_class[cls_id].extend(max_ious)
            
            # 如果该类有预测但没 GT，这些预测的 IoU 全部为 0 (误报)
            elif len(curr_p_idx) > 0 and len(curr_g_idx) == 0:
                iou_stats_per_class[cls_id].extend([0.0] * len(curr_p_idx))


# --- 3. 打印分析报告 ---
print("\n" + "="*50)
print(f"{'Class Name':<25} | {'Avg IoU':<10} | {'Max IoU':<10} | {'Match Rate(>0.1)':<10}")
print("-" * 50)

overall_ious = []
for i, name in enumerate(class_names):
    data = iou_stats_per_class[i]
    if len(data) > 0:
        avg_iou = np.mean(data)
        max_iou = np.max(data)
        # 计算有多少比例的预测框至少蹭到了 GT (IoU > 0.1)
        match_rate = np.mean([1 if x > 0.1 else 0 for x in data])
        
        print(f"{name[:24]:<25} | {avg_iou:<10.4f} | {max_iou:<10.4f} | {match_rate:<10.2%}")
        overall_ious.extend(data)
    else:
        print(f"{name[:24]:<25} | {'N/A':<10} | {'N/A':<10} | {'0%':<10}")

if overall_ious:
    print("-" * 50)
    print(f"{'ALL CLASSES TOTAL':<25} | {np.mean(overall_ious):<10.4f} | {np.max(overall_ious):<10.4f} |")
print("="*50)


# %%[3] 绘制全类别 PR 曲线
from sklearn.metrics import precision_recall_curve, average_precision_score

# 存储每个类别的真伪标签和置信度
all_labels_per_class = defaultdict(list)
all_scores_per_class = defaultdict(list)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # 推理阈值
predictor = DefaultPredictor(cfg)
print("Calculating PR Curves... matching predictions with Ground Truth.")

for d in dataset_dicts_val:
    img = cv2.imread(d["file_name"])
    if img is None: continue
    
    with torch.no_grad():
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes
        pred_scores = instances.scores.tolist()
        pred_classes = instances.pred_classes.tolist()
        
        gt_boxes = [ann["bbox"] for ann in d["annotations"]]
        gt_classes = [ann["category_id"] for ann in d["annotations"]]
        
        # 对每个类别独立进行匹配
        for cls_id in range(len(class_names)):
            # 提取当前图片的预测和 GT
            curr_pred_idx = [i for i, c in enumerate(pred_classes) if c == cls_id]
            curr_gt_idx = [i for i, c in enumerate(gt_classes) if c == cls_id]
            
            if not curr_pred_idx:
                continue
                
            # 如果该图没有该类别的 GT，所有预测均为 False Positive (Label=0)
            if not curr_gt_idx:
                for idx in curr_pred_idx:
                    all_labels_per_class[cls_id].append(0)
                    all_scores_per_class[cls_id].append(pred_scores[idx])
                continue

            # 计算 IoU 矩阵并进行简单匹配 (IoU > 0.5)
            # 这里简化处理：只要预测框与任一 GT 框 IoU > 0.5 即视为 True Positive
            from detectron2.structures import pairwise_iou, Boxes
            p_boxes = pred_boxes[curr_pred_idx]
            g_boxes = Boxes(torch.as_tensor([gt_boxes[i] for i in curr_gt_idx]))
            ious = pairwise_iou(p_boxes, g_boxes)
            
            for i, idx in enumerate(curr_pred_idx):
                max_iou = ious[i].max().item()
                label = 1 if max_iou >= 0.5 else 0
                all_labels_per_class[cls_id].append(label)
                all_scores_per_class[cls_id].append(pred_scores[idx])

# 绘图
plt.figure(figsize=(20, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.suptitle("Precision-Recall Curves per Class (IoU@0.5)", fontsize=20)

for i in range(len(class_names)):
    plt.subplot(3, 3, i + 1)
    y_true = all_labels_per_class[i]
    y_scores = all_scores_per_class[i]
    total_gt = gt_counts_per_class[i] # 使用之前统计的 GT 总数

    if len(y_true) > 0 and total_gt > 0:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # 注意：sklearn 的 recall 是相对于已匹配样本的，我们需要手动缩放到相对于总 GT 数
        recall = recall * (sum(y_true) / total_gt) 
        ap = average_precision_score(y_true, y_scores) * (sum(y_true) / total_gt)

        plt.plot(recall, precision, color='dodgerblue', lw=2, label=f'AP: {ap:.2f}')
        plt.fill_between(recall, precision, alpha=0.2, color='dodgerblue')
        plt.title(f"Class: {class_names[i]}")
    else:
        plt.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
        plt.title(f"Class: {class_names[i]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "pr_curves_multi_class.png"))
plt.show()




# %% [4] 生成可视化对比图 (Master Grid)
vis_save_dir = os.path.join(output_dir, "custom_visualizations")
os.makedirs(vis_save_dir, exist_ok=True)

class_names = MetadataCatalog.get("tooth_val").thing_classes
num_classes = len(class_names)
max_samples = 5

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 推理阈值
predictor = DefaultPredictor(cfg)

# 定义一套固定的颜色表 (BGR格式，用于 OpenCV)
# 确保各个类别的颜色区分度高
COLORS = [
    (0, 0, 255),     # caries: 红色
    (0, 255, 255),   # white_spot_lesion: 黄色
    (255, 0, 0),     # filling_no_caries: 蓝色p
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
target_size = (640, 640) # 统一缩放尺寸，确保矩阵对齐
for cat_id, cat_name in enumerate(class_names):
    samples = category_to_samples[cat_id]
    if not samples:
        continue
        
    # 为当前类别创建一个独立的文件夹
    cat_dir = os.path.join(vis_save_dir, cat_name)
    os.makedirs(cat_dir, exist_ok=True)
    
    # 根据之前设定的随机种子抽取固定样本
    selected = random.sample(samples, min(len(samples), max_samples))
    
    row_orig, row_gt, row_pred = [], [], []

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

        # --- 3. 为矩阵拼接做准备 (缩放并存入行列表) ---
        # 统一缩放，并在图上标记 Sample 编号
        res_o = cv2.resize(img_bgr, target_size)
        res_g = cv2.resize(img_gt, target_size)
        res_p = cv2.resize(img_pred, target_size)
        
        # 在第一行（原图）上方标一下 Sample 序号
        cv2.putText(res_o, f"Sample {idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        row_orig.append(res_o)
        row_gt.append(res_g)
        row_pred.append(res_p)
    
    # --- 4. 合成大矩阵 ---
    if row_orig:
        # 横向拼接每一行 (Columns = Samples)
        combined_orig = np.hstack(row_orig)
        combined_gt = np.hstack(row_gt)
        combined_pred = np.hstack(row_pred)

        # 纵向拼接这三行 (Rows = Stages)
        summary_matrix = np.vstack([combined_orig, combined_gt, combined_pred])

        # 可选：在最左侧增加一个文字标签列（Origin/GT/Pred）
        label_w = 150
        label_col = np.zeros((summary_matrix.shape[0], label_w, 3), dtype=np.uint8)
        labels = ["ORIGIN", "GT", "PRED"]
        for i, txt in enumerate(labels):
            y_pos = i * target_size[1] + target_size[1] // 2
            cv2.putText(label_col, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        final_output = np.hstack([label_col, summary_matrix])

        # 保存该类别的汇总矩阵图
        matrix_save_path = os.path.join(cat_dir, f"00_{cat_name}_summary_matrix.jpg")
        cv2.imwrite(matrix_save_path, final_output)

print(f"所有的独立可视化图片已成功保存至: {vis_save_dir}")

# %% [5] 性能评估 (mAP)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader




# %%
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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 推理阈值
predictor = DefaultPredictor(cfg)
# 评估验证集
res_val = safe_evaluate(predictor.model, "tooth_val", cfg, os.path.join(output_dir, "pred_val"))
# 评估训练集
# %%
res_train = safe_evaluate(predictor.model, "tooth_train", cfg, os.path.join(output_dir, "pred_train"))
# %%
