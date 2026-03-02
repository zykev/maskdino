import json
import os
import cv2
import numpy as np
from datetime import datetime

# ================= 配置区域 =================
# 根目录路径 (你的 single_tooth 文件夹路径)
root = ".datasets/intraoral/single_ch_0225"
ROOT_DIR = os.path.join(root, "single_tooth") 
# 输出的 COCO json 文件路径
OUTPUT_JSON = os.path.join(root, "caries_sample_dataset.json")

# 定义类别映射: ori_lable_id -> coco_category_id
CATEGORY_MAP = {
    "1": 1,  # caries
    "11": 2, # white spot lesion
    "3": 3, # filling without caries
    "4": 4, # filling with caries
    "6": 5, # fissure sealant
    "10": 6, # non-caries disease(hard tissue)
    "12": 7, # staining
    "7": 8, # abnormal central cusp
    "8": 9, # palatal radicular groove
}

# 定义类别信息用于 COCO header
CATEGORIES_INFO = [
    {"id": 1, "name": "caries", "supercategory": "tooth_disease"},
    {"id": 2, "name": "white_spot_lesion", "supercategory": "tooth_disease"},
    {"id": 3, "name": "filling_no_caries", "supercategory": "treatment"},
    {"id": 4, "name": "filling_with_caries", "supercategory": "tooth_disease"},
    {"id": 5, "name": "fissure_sealant", "supercategory": "treatment"},
    {"id": 6, "name": "non_caries_hard_tissue", "supercategory": "tooth_disease"},
    {"id": 7, "name": "staining", "supercategory": "clinical_finding"},
    {"id": 8, "name": "abnormal_central_cusp", "supercategory": "anatomy_anomaly"},
    {"id": 9, "name": "palatal_radicular_groove", "supercategory": "anatomy_anomaly"}
]
# ===========================================

def calculate_polygon_area(points):
    """使用鞋带公式计算多边形面积"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def convert_to_coco(root_dir, output_file):
    coco_output = {
        "info": {
            "description": "Tooth Caries Sample Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES_INFO
    }

    image_id_counter = 1
    annotation_id_counter = 1
    
    # 遍历文件夹结构
    # 结构: root -> sample_id -> view (D,F,L,R,U) -> files
    for sample_id in os.listdir(root_dir):
        sample_path = os.path.join(root_dir, sample_id)
        if not os.path.isdir(sample_path):
            continue
            
        for view in os.listdir(sample_path):
            view_path = os.path.join(sample_path, view)
            if not os.path.isdir(view_path):
                continue
            
            # 遍历该视角下的所有文件
            files = os.listdir(view_path)
            # 找到所有的 json 文件
            json_files = [f for f in files if f.endswith('.json')]
            
            for json_file in json_files:
                json_path = os.path.join(view_path, json_file)
                
                # 对应的图片文件名 (假设是同名 .png)
                image_filename = json_file.replace('.json', '.png')
                image_path = os.path.join(view_path, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found for {json_path}, skipping.")
                    continue
                
                # 1. 读取图片获取宽高
                rel_path = os.path.join(ROOT_DIR, sample_id, view, image_filename).replace("\\", "/")
                
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error reading image: {image_path}")
                    continue
                height, width, _ = img.shape
                
                # 2. 添加到 images 列表
                image_info = {
                    "id": image_id_counter,
                    "file_name": rel_path, # 关键：这是训练时数据加载器寻找图片的路径
                    "height": height,
                    "width": width,
                    "date_captured": datetime.now().strftime("%Y-%m-%d")
                }
                coco_output["images"].append(image_info)
                
                # 3. 读取 JSON 处理标注
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查 flags
                # 根据你的描述： "1": true 表示不健康，需要提取 shapes
                is_unhealthy = False
                if "flags" in data:
                    # 容错处理：有时候json可能只写了 "1": true，或者 "0": true
                    if data["flags"].get("1") is True:
                        is_unhealthy = True
                
                # 只有不健康 且 shapes 不为空时才添加 annotation
                if is_unhealthy and "shapes" in data:
                    for shape in data["shapes"]:
                        label = shape.get("label", "")
                        points = shape.get("points", [])
                        
                        if label not in CATEGORY_MAP:
                            print(f"Skipping unknown label '{label}' in {json_path}")
                            continue
                        
                        if len(points) < 3:
                            continue # 不是多边形
                            
                        # 转换 points 格式
                        # LabelMe: [[x1, y1], [x2, y2]]
                        # COCO segmentation: [x1, y1, x2, y2, ...] (扁平列表)
                        np_points = np.array(points)
                        flat_points = np_points.flatten().tolist()
                        
                        # 计算 BBox [x, y, width, height]
                        x_min = float(np.min(np_points[:, 0]))
                        y_min = float(np.min(np_points[:, 1]))
                        x_max = float(np.max(np_points[:, 0]))
                        y_max = float(np.max(np_points[:, 1]))
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        # 计算面积
                        area = calculate_polygon_area(np_points)
                        
                        annotation = {
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": CATEGORY_MAP[label],
                            "segmentation": [flat_points],
                            "area": area,
                            "bbox": [x_min, y_min, w, h],
                            "iscrowd": 0
                        }
                        coco_output["annotations"].append(annotation)
                        annotation_id_counter += 1
                
                # 无论是否添加了 annotation，image_id 都要自增
                image_id_counter += 1

    # 4. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"Done! Processed {image_id_counter-1} images.")
    print(f"Created {annotation_id_counter-1} annotations.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    convert_to_coco(ROOT_DIR, OUTPUT_JSON)