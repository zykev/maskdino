# SPDX-License-Identifier: MIT
# ============================================================================
# SegmentAnyTooth Batch Processing Script
#
# Copyright (c) 2025 Khoa D. Nguyen (Original Code)
#
# This script processes a dataset of intraoral images, generates tooth segmentations
# and bounding boxes, and saves the results in a structured format.
# ============================================================================

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Literal, Optional, Dict, List, Tuple
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from sam import sam_load, sam_predict
from utils import suppress_stdout
from tqdm import tqdm
from pathlib import Path
import argparse


# --- Configuration ---
# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="SegmentAnyTooth 批量牙齿分割与处理脚本")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default=".datasets/intraoral/test", 
        help="zeyu 根目录路径"
    )
    parser.add_argument(
        "--weight_dir", 
        type=str, 
        default=".checkpoints/segtooth_model", 
        help="权重文件目录"
    )
    parser.add_argument(
        "--sam_batch_size", 
        type=int, 
        default=10, 
        help="SAM 推理的 Batch Size"
    )
    parser.add_argument(
        "--min_conf", 
        type=float, 
        default=0.35, 
        help="YOLO 检测的最小置信度阈值"
    )
    parser.add_argument(
        "--min_area", 
        type=int, 
        default=500, 
        help="过滤检测框的最小面积(像素)"
    )
    parser.add_argument(
        "--process_mask_only",
        action="store_true",
        help="只补充输出 process_mask，不重新生成 preview、tooth_bbox、single_tooth、sextant"
    )
    return parser.parse_args()

args = parse_args()

DATA_ROOT = Path(args.base_dir).resolve()
WEIGHT_DIR = Path(args.weight_dir).resolve()
SAM_BATCH_SIZE = args.sam_batch_size
MIN_CONFIDENCE = args.min_conf
MIN_AREA_PIXELS = args.min_area
PROCESS_MASK_ONLY = args.process_mask_only

# View mapping: Filename prefix -> View type required by predict()
# Assuming filenames are strictly "F.png", "U.png", "D.png", "L.png", "R.png" or similar.
# Map: D (Down/Lower), U (Upper), F (Front), L (Left), R (Right)
VIEW_MAPPING = {
    "F": "front",
    "U": "upper",
    "D": "lower", # "D" usually stands for Down/Lower occlusal
    "L": "left",
    "R": "right"
}

# --- Target Teeth Definition ---
# 定义每个视角下需要提取的牙齿 FDI 编号
# 注意：Frontal 中你写了两个 42，我推测其中一个是 41。
TARGET_TEETH_PER_VIEW = {
    "front":  [13, 12, 11, 21, 22, 23, 43, 42, 41, 31, 32, 33],
    "left":   [24, 25, 26, 27, 28, 34, 35, 36, 37, 38],
    "right":  [14, 15, 16, 17, 18, 44, 45, 46, 47, 48],
    "upper":  [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower":  [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
}

# Sextant 定义: 每个区域包含哪些 FDI 牙位
# S1: 右上后牙, S2: 上前牙, S3: 左上后牙
# S4: 左下后牙, S5: 下前牙, S6: 右下后牙
SEXTANT_DEFINITIONS = {
    "S1": [18, 17, 16, 15, 14, 13], # 包含13以辅助边界
    "S2": [13, 12, 11, 21, 22, 23],
    "S3": [23, 24, 25, 26, 27, 28], # 包含23以辅助边界
    "S4": [38, 37, 36, 35, 34, 33], # 包含33
    "S5": [33, 32, 31, 41, 42, 43],
    "S6": [43, 44, 45, 46, 47, 48]  # 包含43
}

# 视图与 Sextant 的对应关系 (即在这个视图下应该切哪些 Sextant)
VIEW_TO_SEXTANT_MAPPING = {
    "front": ["S2", "S5"],
    "right": ["S1", "S6"], # 此时 Right View 显示的是患者右侧
    "left":  ["S3", "S4"],
    "upper": [], # 通常全口咬合面不切 Sextant，或者切 S1-S3
    "lower": []  # 通常全口咬合面不切 Sextant，或者切 S4-S6
}

# --- Constants & Helpers from Original Code ---

LOGGER.setLevel("ERROR")

LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]

def get_color_map() -> Dict[int, Tuple[int, int, int]]:
    """生成 FDI 标签到 RGB 颜色的映射"""
    fdi_labels = []
    for quadrant in [1, 2, 3, 4]:
        for tooth_num in range(1, 9):
            label = quadrant * 10 + tooth_num
            fdi_labels.append(label)
    
    num_labels = len(fdi_labels)
    cmap = plt.get_cmap('gist_ncar', num_labels)
    color_map = {}
    color_map[0] = (0, 0, 0)
    
    for i, label in enumerate(fdi_labels):
        rgb_float = np.array(cmap(i)[:3])
        rgb_int = (rgb_float * 255).astype(np.uint8)
        color_map[label] = tuple(rgb_int.tolist())
    return color_map

def get_model_path(
    model: Literal["upper", "lower", "left", "right", "front", "sam"],
    weight_dir: Optional[str] = "./weight",
) -> str:
    """Returns the file path to the model weights."""
    if model == "left":
        model = "right" # Left view uses right model with flipping

    if model == "sam":
        name = "vit_tiny.pt"
    else:
        name = f"yolo11_{model}.pt"

    return os.path.join(weight_dir, f"segmentanytooth_{name}")

def predict(
    image_path: str,
    view: Literal["upper", "lower", "left", "right", "front"],
    weight_dir: Optional[str] = "./weight",
    sam_batch_size: Optional[int] = 10,
) -> Tuple[np.ndarray, List[Dict]]:
    """Predicts mask and boxes."""
    weight_dir = os.path.normpath(weight_dir)
    should_flip = view == "left"
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if should_flip:
        image = cv2.flip(image, 1)

    # Inference
    with suppress_stdout():
        sam = sam_load(get_model_path("sam", weight_dir))
        yolo = YOLO(model=get_model_path(view, weight_dir))
        r = yolo.predict(
            image,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            project=None,
        )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    names = r.names if not should_flip else LEFT_CLASSES
    boxes = r.boxes.xyxy.squeeze(0).cpu().numpy()
    clss = r.boxes.cls.squeeze(0).cpu().numpy().astype(np.int32)
    confs = r.boxes.conf.cpu().numpy()

    # --- ### NEW: FILTERING LOGIC (过滤逻辑开始) ### ---
    
    # 1. Calculate Area (Width * Height)
    # boxes format: x1, y1, x2, y2
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    # 2. Create Boolean Mask
    # Keep if Confidence > Threshold AND Area > Threshold
    valid_mask = (confs >= MIN_CONFIDENCE) & (areas >= MIN_AREA_PIXELS)

    # 3. Apply Filter
    boxes = boxes[valid_mask]
    clss = clss[valid_mask]
    confs = confs[valid_mask] # (Optional: keep if needed for debug)

    # 4. Check if anything is left after filtering
    if len(boxes) == 0:
        print(f"    (Filtered out all detections based on conf<{MIN_CONFIDENCE} or area<{MIN_AREA_PIXELS})")
        return np.zeros(image.shape[:2], dtype=np.uint8), []
    
    # --- ### NEW: Step 4: Parse FDI Names early (提前解析 FDI) ### ---
    # 我们需要先知道每个检测结果对应的真实 FDI 编号，才能进行去重
    parsed_fdi_list = []
    valid_indices_after_parse = []
    
    for idx, cls_id in enumerate(clss):
        tooth_name_str = names[cls_id]
        fdi = int(''.join(filter(str.isdigit, tooth_name_str)))
        parsed_fdi_list.append(fdi)
        valid_indices_after_parse.append(idx)
            
    # 只保留能解析出 FDI 的框
    boxes = boxes[valid_indices_after_parse]
    clss = clss[valid_indices_after_parse]
    confs = confs[valid_indices_after_parse]
    fdis = np.array(parsed_fdi_list) # 转换为 numpy array 方便操作

    # --- ### NEW: Step 5: Keep Max Confidence per Class (去重逻辑) ### ---
    # 逻辑：对于每一个唯一的 fdi 编号，找到它对应的 conf 最大的那个索引
    unique_fdis = np.unique(fdis)
    final_indices = []
    
    for fdi in unique_fdis:
        # 找到所有预测为该 FDI 的索引
        indices_for_this_fdi = np.where(fdis == fdi)[0]
        
        # 找到这些索引中 conf 最大的那个在 fdis 数组中的位置
        # np.argmax 返回的是局部位置，需要转换回 indices_for_this_fdi 中的值
        best_local_idx = np.argmax(confs[indices_for_this_fdi])
        best_global_idx = indices_for_this_fdi[best_local_idx]
        
        final_indices.append(best_global_idx)
    
    # 应用去重筛选
    boxes = boxes[final_indices]
    clss = clss[final_indices]
    confs = confs[final_indices]

    sort_ids = np.argsort(clss)
    clss = clss[sort_ids]
    boxes = boxes[sort_ids]
    confs = confs[sort_ids]

    if should_flip:
        image_width = image.shape[1]
        image = cv2.flip(image, 1)
        flipped_boxes = boxes.copy()
        flipped_boxes[:, [0, 2]] = image_width - flipped_boxes[:, [2, 0]]
        boxes = flipped_boxes

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_masks = sam_predict(
        sam=sam,
        boxes_xyxy=boxes,
        image=image_rgb,
        batch_size=sam_batch_size,
    )

    predict_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    boxes_data_list = []
    
    for i, (cls_id, current_mask) in enumerate(zip(clss, sam_masks)):
        fdi_tooth_name = int(names[cls_id][-2:])
        predict_mask[current_mask == 1] = fdi_tooth_name
        box = boxes[i]
        boxes_data_list.append({
            "fdi_tooth_name": fdi_tooth_name,
            "box": box.tolist(),
            "confidence": float(confs[i]) # 可选：把置信度也存进JSON
        })

    return predict_mask, boxes_data_list

def process_and_save(
    mask: np.ndarray, 
    boxes_data: List[Dict], 
    image_path: str,
    json_output_dir: str,
    preview_output_dir: str,
    file_stem: str # e.g. "F"
):
    """
    Saves JSON to json_output_dir and PNG to preview_output_dir.
    """
    STANDARD_FDI_COLOR_MAP = get_color_map()
    original_image = Image.open(image_path)
    
    # 1. Save JSON
    json_output = {}
    for item in boxes_data:
        class_name = str(item["fdi_tooth_name"]) # Use string for JSON keys
        box = item["box"]
        conf = item['confidence']
        if class_name not in json_output:
            json_output[class_name] = []
        json_output[class_name].append({
            "box": box,
            "confidence": conf
        })
    
    os.makedirs(json_output_dir, exist_ok=True)
    json_filename = f"{file_stem}.json"
    json_path = os.path.join(json_output_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
    # print(f"  -> JSON saved: {json_path}")

    # 2. Save Visualization PNG
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in STANDARD_FDI_COLOR_MAP.items():
        colored_mask[mask == label] = color

    # Setup Plot
    # Use output dpi=100, calculate figsize to match original resolution approximately
    fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100), dpi=100)
    
    # Visualization Config
    LABEL_FONTSIZE = 20
    TEXT_COLOR = 'cyan' 
    BOX_COLOR = 'yellow' 
    MASK_ALPHA = 0.5
    
    ax.imshow(original_image)
    ax.imshow(colored_mask, alpha=MASK_ALPHA)

    for item in boxes_data:
        class_name = str(item["fdi_tooth_name"])
        box = item.get("box")
        if box is None or len(box) != 4: continue

        xmin, ymin, xmax, ymax = box
        
        # Draw Box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=BOX_COLOR, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw Label
        ax.text(
            xmin, ymin - 10, class_name,
            color=TEXT_COLOR, fontsize=LABEL_FONTSIZE, fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')
        )
        
    ax.axis('off')
    
    os.makedirs(preview_output_dir, exist_ok=True)
    preview_filename = f"{file_stem}.png"
    preview_path = os.path.join(preview_output_dir, preview_filename)
    
    fig.savefig(preview_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    # print(f"  -> Preview saved: {preview_path}")

def save_single_tooth_images(
    image: np.ndarray,      # 原始图像 (OpenCV BGR 格式)
    mask: np.ndarray,       # 全图分割掩码 (每个像素值为 FDI 编号)
    boxes_data: List[Dict], # 包含 box 和 FDI 的列表
    output_root: str,       # single_tooth/idxxx/
    view_stem: str,         # 当前视角代码，如 "F", "L"
    view_type: str,         # 当前视角类型，如 "front", "left"
):
    """
    根据 Bounding Box 裁剪牙齿，并将掩码外的区域置黑，保存为单颗牙齿图片。
    """
    # 1. 获取当前视角允许的目标牙齿列表
    target_fdis = TARGET_TEETH_PER_VIEW.get(view_type, [])
    
    # 2. 创建该视角对应的子文件夹 (例如 single_tooth/id001/F/)
    dir_name = os.path.dirname(output_root)
    id_name = os.path.basename(output_root)
    # view_mask_output_dir = os.path.join(dir_name + "_mask", id_name, view_stem)
    view_nomask_output_dir = os.path.join(dir_name, id_name, view_stem)
    # os.makedirs(view_mask_output_dir, exist_ok=True)
    os.makedirs(view_nomask_output_dir, exist_ok=True)
    
    img_h, img_w = image.shape[:2]

    for item in boxes_data:
        fdi = item["fdi_tooth_name"]
        box = item["box"]
        
        # 3. 过滤：如果这颗牙不在当前视角的提取列表中，跳过
        if fdi not in target_fdis:
            continue
            
        # 4. 获取坐标并进行边界检查 (Clamping)
        # YOLO 输出可能是 float，转换为 int
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        
        # 检查无效框
        if x2 <= x1 or y2 <= y1:
            continue

        # 5. 裁剪图像和掩码
        # crop_img: (H, W, 3)
        crop_img = image[y1:y2, x1:x2].copy()
        # # crop_mask: (H, W) - 注意这里裁剪的是全图掩码
        # crop_mask = mask[y1:y2, x1:x2]
        
        # # 6. 应用遮罩 (Apply Mask)
        # # 我们只保留当前 FDI 编号的像素。
        # # 注意：crop_mask 中可能包含相邻牙齿的部分（例如切 11 时切到了 21），
        # # 我们必须把那些也变成黑色，只保留 11。
        
        # # 创建二值掩码：只有当前牙齿区域为 255 (白色)，其余为 0 (黑色)
        # binary_mask = np.zeros_like(crop_mask, dtype=np.uint8)
        # binary_mask[crop_mask == fdi] = 255
        
        # # 7. 保存图片
        # # --- Version A: Masked (应用遮罩，背景置黑) ---
        # masked_tooth = cv2.bitwise_and(crop_img, crop_img, mask=binary_mask)
        # cv2.imwrite(os.path.join(view_mask_output_dir, f"{fdi}.png"), masked_tooth)
        
        # --- Version B: Unmasked (不应用遮罩，只进行裁剪) ---
        # 直接保存裁剪后的原始图像
        cv2.imwrite(os.path.join(view_nomask_output_dir, f"{fdi}.png"), crop_img)

def save_process_mask_image(
    image: np.ndarray,
    mask: np.ndarray,
    output_root: str,
    view_filename: str,
):
    """
    Saves the full process image with only segmented tooth pixels visible.
    Non-tooth regions are set to black.
    """
    os.makedirs(output_root, exist_ok=True)

    binary_mask = (mask > 0).astype(np.uint8) * 255
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

    cv2.imwrite(os.path.join(output_root, view_filename), masked_image)

def save_sextant_crops(
    image: np.ndarray,      # 原始图像
    boxes_data: List[Dict], # 检测到的牙齿数据
    output_root: str,       # sextant/idxxx/
    view_stem: str,         # "F", "L", "R"
    view_type: str          # "front", "left", "right"
):
    """
    聚合多颗牙齿的 Bounding Box，生成 Sextant 区域裁剪图。
    """
    target_sextants = VIEW_TO_SEXTANT_MAPPING.get(view_type, [])
    if not target_sextants:
        return

    # 创建输出目录: sextant/id001/F/
    view_output_dir = os.path.join(output_root, view_stem)
    os.makedirs(view_output_dir, exist_ok=True)
    
    img_h, img_w = image.shape[:2]

    # 将检测数据转换为便于查找的字典: fdi -> box
    detection_map = {item["fdi_tooth_name"]: item["box"] for item in boxes_data}

    for sextant_id in target_sextants:
        target_fdis = SEXTANT_DEFINITIONS[sextant_id]
        
        # 1. 收集属于当前 Sextant 的所有牙齿的 Box
        relevant_boxes = []
        for fdi in target_fdis:
            if fdi in detection_map:
                relevant_boxes.append(detection_map[fdi])
        
        # 如果当前 Sextant 一颗牙都没检测到，跳过
        if not relevant_boxes:
            continue
            
        # 2. 计算 Union Box (合并大框)
        relevant_boxes = np.array(relevant_boxes) # Shape: [N, 4]
        # x1_min, y1_min, x2_max, y2_max
        union_x1 = np.min(relevant_boxes[:, 0])
        union_y1 = np.min(relevant_boxes[:, 1])
        union_x2 = np.max(relevant_boxes[:, 2])
        union_y2 = np.max(relevant_boxes[:, 3])
        
        # 基础宽高
        box_w = union_x2 - union_x1
        box_h = union_y2 - union_y1
        
        # 3. 应用定向 Padding (核心逻辑)
        # 初始化 padding 值
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
        
        # 定义扩充比例
        GUM_PADDING_RATIO = 0.4  # 垂直方向扩充 40% 看牙龈
        SIDE_PADDING_RATIO = 0.1 # 水平方向适当扩充 10%

        # 根据 Sextant ID 判断是上颌还是下颌
        is_upper = sextant_id in ["S1", "S2", "S3"]
        
        if is_upper:
            # 上颌牙：重点向上扩充 (y_min 减小)
            pad_top = box_h * GUM_PADDING_RATIO
            pad_bottom = box_h * 0.05 # 下方稍微留一点点
        else:
            # 下颌牙：重点向下扩充 (y_max 增加)
            pad_top = box_h * 0.05
            pad_bottom = box_h * GUM_PADDING_RATIO
            
        # 水平扩充 (左右各扩一点，防止切断边缘牙齿)
        pad_left = box_w * SIDE_PADDING_RATIO
        pad_right = box_w * SIDE_PADDING_RATIO
        
        # 特殊处理：Right View (S1, S6) 和 Left View (S3, S4) 的智齿区域
        # 假设图片左侧是口腔后方 (对于右视图通常如此，视拍摄手法而定)
        # 这里使用通用的左右扩充通常足够。如果需要特定方向（如向后方扩充），可以根据 view_type 调整
        if view_type == "right": 
             # 右视图，通常图片左边是后牙区，右边是前牙
             pad_left += box_w * 0.1 # 向后方多扩一点
        elif view_type == "left":
             # 左视图，通常图片右边是后牙区
             pad_right += box_w * 0.1

        # 4. 应用坐标变换
        final_x1 = int(union_x1 - pad_left)
        final_y1 = int(union_y1 - pad_top)
        final_x2 = int(union_x2 + pad_right)
        final_y2 = int(union_y2 + pad_bottom)
        
        # 5. 边界检查 (Clamping)
        final_x1 = max(0, final_x1)
        final_y1 = max(0, final_y1)
        final_x2 = min(img_w, final_x2)
        final_y2 = min(img_h, final_y2)
        
        # 确保有效
        if final_x2 <= final_x1 or final_y2 <= final_y1:
            continue
            
        # 6. 裁剪并保存
        crop_img = image[final_y1:final_y2, final_x1:final_x2]
        
        filename = f"{sextant_id}.png"
        save_path = os.path.join(view_output_dir, filename)
        cv2.imwrite(save_path, crop_img)




def main():
    if not DATA_ROOT.exists():
        print(f"Error: Base directory '{DATA_ROOT}' does not exist.")
        return

    # 定义错误日志路径
    error_log_path = DATA_ROOT / "process_errors.txt"

    # 1. 查找所有以 _process 结尾的日期文件夹
    date_folders = [d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.endswith("_process")]
    date_folders.sort()

    if not date_folders:
        print("No '_process' folders found in zeyu directory.")
        return

    # --- 总进度条 ---
    pbar_dates = tqdm(date_folders, desc="Overall Dates", unit="date")
    
    for date_path in pbar_dates:
        pbar_dates.set_description(f"Date: {date_path.name}")
        
        input_root = date_path / "process"
        if not input_root.exists():
            continue

        json_output_root = date_path / "tooth_bbox"
        preview_output_root = date_path / "preview"
        single_tooth_output_root = date_path / "single_tooth"
        sextant_output_root = date_path / "sextant"
        process_mask_output_root = date_path / "process_mask"

        id_dirs = [d for d in input_root.iterdir() if d.is_dir()]
        id_dirs.sort()

        # --- 子进度条 ---
        pbar_ids = tqdm(id_dirs, desc=f"  Processing {date_path.name}", leave=False, unit="id")
        
        for id_folder_path in pbar_ids:
            id_folder = id_folder_path.name
            
            id_json_path = json_output_root / id_folder
            id_preview_path = preview_output_root / id_folder
            id_single_tooth_path = single_tooth_output_root / id_folder
            id_sextant_path = sextant_output_root / id_folder
            id_process_mask_path = process_mask_output_root / id_folder

            valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG']
            
            for stem, view_name in VIEW_MAPPING.items():
                found_file = False
                for ext in valid_extensions:
                    filename = f"{stem}{ext}"
                    file_path = id_folder_path / filename
                    
                    if file_path.exists():
                        found_file = True
                        try:
                            # 预测
                            mask, boxes_data = predict(
                                image_path=str(file_path),
                                view=view_name,
                                weight_dir=WEIGHT_DIR,
                                sam_batch_size=SAM_BATCH_SIZE
                            )
                            
                            raw_image = cv2.imread(str(file_path)) 

                            save_process_mask_image(
                                image=raw_image,
                                mask=mask,
                                output_root=str(id_process_mask_path),
                                view_filename=file_path.name
                            )

                            if not PROCESS_MASK_ONLY:
                                # 保存 JSON 和 预览图
                                process_and_save(
                                    mask=mask,
                                    boxes_data=boxes_data,
                                    image_path=str(file_path),
                                    json_output_dir=str(id_json_path),
                                    preview_output_dir=str(id_preview_path),
                                    file_stem=stem
                                )

                                # 保存单颗牙齿
                                save_single_tooth_images(
                                    image=raw_image,
                                    mask=mask,
                                    boxes_data=boxes_data,
                                    output_root=str(id_single_tooth_path),
                                    view_stem=stem,
                                    view_type=view_name
                                )

                                # 保存 Sextant 区域
                                save_sextant_crops(
                                    image=raw_image,
                                    boxes_data=boxes_data,
                                    output_root=str(id_sextant_path),
                                    view_stem=stem,
                                    view_type=view_name
                                )
                        except Exception as e:
                            # --- 异常记录逻辑 ---
                            error_msg = f"ERROR | Path: {file_path} | Reason: {str(e)}\n"
                            tqdm.write(f"Skipping {file_path.name} due to error.")
                            
                            # 以追加模式写入 txt
                            with open(error_log_path, "a", encoding="utf-8") as f:
                                f.write(error_msg)
                        break
    
    if error_log_path.exists():
        print(f"\nProcessing finished. Some errors occurred, check: {error_log_path}")
    else:
        print("\n--- All tasks completed successfully without errors! ---")

if __name__ == "__main__":
    main()
