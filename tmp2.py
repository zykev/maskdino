# %%
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 配置 ---
ROOT_DIR = ".datasets/intraoral/annosample_ch/sextant"
ANNOTATION_COLOR = (255, 0, 0) 

# 指定的可视化目标 (Label: (Sample_ID, View_ID))
TARGETS_PLAQUE = {
    "p0": ("s20251105", "S5"),
    "p1": ("s20251077", "S2"),
    "p2": ("s20251111", "S6")
}

TARGETS_GINGIVITIS = {
    "g0": ("s20251157", "S2"),
    "g1": ("s20251114", "S1"),
    "g3": ("s20251047", "S2")
}

def find_image_and_json(sample_id, view_id):
    """
    穿透 F/L/R 层级寻找匹配 view_id 的图片和json
    路径结构示例: ROOT/s20251114/L/S2/24.png
    """
    sample_path = os.path.join(ROOT_DIR, sample_id)
    if not os.path.exists(sample_path):
        return None, None

    # 递归遍历样本文件夹
    for dir in os.listdir(sample_path):
        dir_path = os.path.join(sample_path, dir)
        for root, dirs, files in os.walk(dir_path):
            # 检查当前目录名是否为目标 view_id (如 "S2")
            if f"{view_id}.png" in files:
                img_p = os.path.join(root, f"{view_id}.png")
                json_p = img_p.replace(".png", ".json")
                if os.path.exists(json_p):
                    return img_p, json_p
    return None, None

def draw_polygons(img, json_path, target_label=None):
    """绘制标注"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for shape in data.get('shapes', []):
        label = str(shape.get('label'))
        # 牙菌斑模式：只画匹配的 p0, p1, p2
        if target_label and label.lower() != target_label:
            continue
        pts = np.array(shape.get('points', []), np.int32)
        if pts.size > 0:
            cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, ANNOTATION_COLOR, 3)
    return img

def visualize_specific_targets(target_dict, type_name):
    """生成单行组图"""
    keys = sorted(target_dict.keys())
    num_cols = len(keys)
    
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    if num_cols == 1: axes = [axes]

    for i, grade_key in enumerate(keys):
        sample_id, view_id = target_dict[grade_key]
        img_p, json_p = find_image_and_json(sample_id, view_id)
        
        ax = axes[i]
        if img_p:
            img = cv2.imread(img_p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 牙菌斑可视化逻辑
            if type_name == "Gingivitis":
                pass
            if type_name == "Plaque":
                # p0 通常代表健康/无菌斑，不画框；p1, p2 画对应框
                if grade_key != "p0":
                    img = draw_polygons(img, json_p, target_label=grade_key)
            
            ax.imshow(img)
            # 提取路径中的方向(F/L/R)用于标题显示
            direction = Path(img_p).parts[-2] 
            ax.set_title(f"Grade: {grade_key}\nID: {sample_id}_{direction}_{view_id}", fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"Not Found:\n{sample_id}\n{view_id}", ha='center', va='center', color='red')
        
        ax.axis('off')

    plt.tight_layout()
    output_path = f"specified_{type_name.lower()}_vis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"成功保存 {type_name} 可视化结果至: {output_path}")
    plt.show()

if __name__ == "__main__":
    # 执行可视化
    visualize_specific_targets(TARGETS_GINGIVITIS, "Gingivitis")
    visualize_specific_targets(TARGETS_PLAQUE, "Plaque")