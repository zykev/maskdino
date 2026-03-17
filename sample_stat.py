# %%
import os
import json
from collections import defaultdict, Counter
from pathlib import Path

label_dict = {"1": "Caries", "11": "White spot leison", "3": "Filling without caries", 
              "6": "Fissure sealant", "10": "Non-caries disease (hard tissue)", "12": "Staining", 
              "4": "Filling with caries", "7": "Abnormal central cusp", "8": "Palatal radicular groove",
              "loss of fissure sealant": "Loss of fissure sealant", "abnormal central cusp": "Abnormal central cusp"}


def process_dental_dataset(root_path, output_txt="dental_stats_summary.txt"):
    # 1. 基础计数器
    img_stats = {'total': 0, 'healthy': 0, 'unhealthy': 0}
    # 使用集合跟踪样本 ID 的状态
    sample_status = defaultdict(set) 
    # 统计不健康类别的频次
    label_counts = Counter()

    print(f"开始处理目录: {root_path} ...")

    # --- 遍历文件 ---
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                
                parts = Path(json_path).parts
                sample_id = parts[-3] if len(parts) >= 3 else "Unknown"

                img_stats['total'] += 1
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    flags = data.get('flags', {})
                    is_unhealthy = flags.get('1', False)

                    if is_unhealthy:
                        img_stats['unhealthy'] += 1
                        sample_status[sample_id].add('unhealthy')
                        
                        # 统计具体的不健康类别 (Label)
                        shapes = data.get('shapes', [])
                        for shape in shapes:
                            label = shape.get('label')
                            if label:
                                label_counts[label] += 1
                    else:
                        img_stats['healthy'] += 1
                        sample_status[sample_id].add('healthy')

                except Exception as e:
                    print(f"跳过损坏文件 {json_path}: {e}")

    # --- 汇总样本级统计 ---
    total_samples = len(sample_status)
    unhealthy_samples = sum(1 for s in sample_status.values() if 'unhealthy' in s)
    healthy_samples = total_samples - unhealthy_samples

    # --- 准备输出内容 ---
    stats_output = [
        "="*50,
        "           口腔数据汇总统计报告",
        "="*50,
        f"数据源路径: {root_path}",
        f"总图片数:   {img_stats['total']} (健康: {img_stats['healthy']}, 不健康: {img_stats['unhealthy']})",
        f"总样本数:   {total_samples} (健康: {healthy_samples}, 不健康: {unhealthy_samples})",
        "-"*50,
        "不健康类别详细统计 (Label Counts):"
    ]
    
    # 排序类别：按数量从高到低
    for label, count in label_counts.most_common():
        label_name = label_dict.get(label, label)  # 获取友好名称
        stats_output.append(f"  - {label_name}: {count}")
    
    if not label_counts:
        stats_output.append("  (未发现任何不健康类别标注)")
    
    stats_output.append("="*50)

    # --- 执行输出：命令行 + 写入文件 ---
    final_report = "\n".join(stats_output)
    
    print(final_report) # 打印到终端

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n统计报告已成功保存至: {output_txt}")

if __name__ == "__main__":
    
    # 设置你的路径
    DATASET_DIR = r".datasets/intraoral/single_ch_0225/single_tooth"
    
    process_dental_dataset(DATASET_DIR)

# %%
import os
import json
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
random.seed(42)


label_dict = {"1": "Caries", "11": "White spot leison", "3": "Filling without caries", 
              "6": "Fissure sealant", "10": "Non-caries disease (hard tissue)", "12": "Staining", 
              "4": "Filling with caries", "7": "Abnormal central cusp", "8": "Palatal radicular groove",
              "loss of fissure sealant": "Loss of fissure sealant", "abnormal central cusp": "Abnormal central cusp"}

def visualize_unhealthy_samples(root_path):
    # --- 1. 数据收集 ---
    # 结构: {'Label名称': [{'json': json_path, 'img': img_path}, ...]}
    label_data = defaultdict(list)
    allowed_labels = set(label_dict.keys())

    print(f"正在扫描数据: {root_path} ...")

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                # 假设图片是 png，如果是 jpg 请修改
                img_path = os.path.join(dirpath, filename.replace(".json", ".png"))
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 只有不健康的才处理
                    if data.get('flags', {}).get('1', False):
                        shapes = data.get('shapes', [])
                        for shape in shapes:
                            label = str(shape.get('label'))
                            if label in allowed_labels:
                                label_data[label].append({
                                    'json': json_path,
                                    'img': img_path
                                })
                except Exception as e:
                    continue

    if not label_data:
        print("未找到不健康样本，无法可视化。")
        return

    # --- 2. 随机采样与绘图准备 ---
    active_labels = sorted(label_data.keys())
    num_labels = len(active_labels)
    print(f"共发现 {num_labels} 个不健康类别，准备生成可视化组图...")

    # 计算网格行列 (例如 10个类别 -> 3行4列)
    cols = 5
    rows = math.ceil(num_labels / cols)

    # 创建画布
    plt.figure(figsize=(5 * cols, 5 * rows))
    # plt.suptitle(f"不健康牙齿类别随机采样可视化 (总类别数: {num_labels})", fontsize=16, y=0.98)

    # --- 3. 循环处理每个类别 ---
    for idx, label in enumerate(active_labels):
        # 随机选一个样本
        if label in label_dict.keys():
            sample_pair = random.choice(label_data[label])
            json_file = sample_pair['json']
            img_file = sample_pair['img']

            # 读取图片
            if not os.path.exists(img_file):
                print(f"警告: 图片不存在 {img_file}")
                continue
                
            # OpenCV 读取的是 BGR，需转为 RGB 以便 Matplotlib 显示
            img = cv2.imread(img_file)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 读取 JSON 获取坐标
            with open(json_file, 'r', encoding='utf-8') as f:
                js_content = json.load(f)

            # --- 4. 绘制多边形 ---
            shapes = js_content.get('shapes', [])
            found_shape = False
            
            for shape in shapes:
                # 只绘制当前关注的这个 Label，避免图片太乱
                # 如果想绘制该图上所有的病灶，去掉 `if shape['label'] == label:` 判断即可
                if shape.get('label') == label:
                    points = shape.get('points', [])
                    if points:
                        # 坐标点转为 numpy int32 数组，OpenCV 绘图需要整数
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        # 绘制轮廓 (图像, 坐标点, 是否闭合, 颜色(RGB), 线宽)
                        # 颜色: 红色 (255, 0, 0)
                        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
                        
                        found_shape = True

            # --- 5. 添加到子图 ---
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img)
            p = Path(img_file)
            sample_title = f"{p.parent.parent.name}_{p.parent.name}{p.stem}"
            label_title = label_dict[label]
            ax.set_title(label_title, fontsize=12)
            # ax.set_title(f"Label: {label_title}\nSample: {sample_title}", fontsize=10)
            ax.axis('off')  # 关闭坐标轴显示

    # --- 6. 输出与保存 ---
    plt.tight_layout()
    output_vis_path = "caries_visualization.png"
    plt.savefig(output_vis_path, dpi=150)
    print(f"可视化组图已保存至: {os.path.abspath(output_vis_path)}")
    plt.show()

if __name__ == "__main__":
    # 请替换为你的数据集路径
    DATASET_DIR = r".datasets/intraoral/single_ch_0225/single_tooth"
    
    if os.path.exists(DATASET_DIR):
        visualize_unhealthy_samples(DATASET_DIR)
    else:
        print("路径不存在")



# %%
import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
random.seed(42)

# --- 配置 ---
ROOT_DIR = ".datasets/intraoral/annosample_ch/sextant"
PERIO_GRADES = ["G0", "G1", "G3", "G4"]

def stat_periodontal(root_path):
    # 统计计数
    perio_counts = defaultdict(int)
    plaque_counts = defaultdict(int)

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                img_path = os.path.join(dirpath, filename.replace(".json", ".png"))
                
                if not os.path.exists(img_path):
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 1. 牙周炎分级统计 (Flags)
                flags = data.get('flags', {})
                for g in PERIO_GRADES:
                    if flags.get(g):
                        perio_counts[g] += 1
                        break 

                # 2. 牙菌斑分级统计 (Shapes)
                shapes = data.get('shapes', [])
                if not shapes:
                    plaque_counts["None"] += 1
                else:
                    # 获取该图片中包含的所有牙菌斑等级
                    current_img_labels = set()
                    for s in shapes:
                        label = s.get('label')
                        if label:
                            current_img_labels.add(label)
                    
                    for lbl in current_img_labels:
                        plaque_counts[lbl] += 1

    return perio_counts, plaque_counts



# --- 主程序 ---
if __name__ == "__main__":
    p_counts, plq_counts = stat_periodontal(ROOT_DIR)

    print("\n" + "="*30)
    print("牙周炎 (Periodontitis) 统计:")
    for g in PERIO_GRADES:
        print(f"  {g}: {p_counts[g]} 张")

    print("\n牙菌斑 (Plaque) 统计:")
    for lbl in sorted(plq_counts.keys()):
        print(f"  {lbl}: {plq_counts[lbl]} 张")
    print("="*30 + "\n")


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

# %%
import os
from collections import defaultdict

def analyze_dental_folders(base_paths):
    print(f"{'根目录':<20} | {'总图片数':<10} | {'样本(Sample)数':<12} | {'平均每样本张数'}")
    print("-" * 75)
    
    for root_path in base_paths:
        if not os.path.exists(root_path):
            print(f"{root_path:<20} | 路径不存在")
            continue
            
        png_count = 0
        samples = set()
        
        # 遍历根目录
        # 假设结构是 root/sample_id/view_id/xxx.png
        for entry in os.scandir(root_path):
            if entry.is_dir():
                sample_id = entry.name
                has_png_in_sample = False
                
                # 递归查找该样本下的所有 png
                for sub_root, _, files in os.walk(entry.path):
                    for file in files:
                        if file.lower().endswith('.png'):
                            png_count += 1
                            has_png_in_sample = True
                
                if has_png_in_sample:
                    samples.add(sample_id)
        
        sample_num = len(samples)
        avg = png_count / sample_num if sample_num > 0 else 0
        
        print(f"{root_path:<20} | {png_count:<12} | {sample_num:<14} | {avg:.2f}")

# 你的文件夹路径
target_dirs = [
    ".datasets/intraoral/annosample_ch/single_tooth",
    ".datasets/intraoral/annosample_ch/sextant"
]

if __name__ == "__main__":
    analyze_dental_folders(target_dirs)



# %%
import os
from pathlib import Path

def visualize_specific_targets(target_dict, type_name):
    """
    将每个等级的目标单独保存为图像文件到 tmp 文件夹
    """
    # 1. 确保目标目录存在
    save_dir = "tmp"
    os.makedirs(save_dir, exist_ok=True)
    
    keys = sorted(target_dict.keys())
    
    for grade_key in keys:
        sample_id, view_id = target_dict[grade_key]
        img_p, json_p = find_image_and_json(sample_id, view_id)
        
        # 为每一张图创建一个独立的画布
        plt.figure(figsize=(6, 6))
        
        if img_p:
            img = cv2.imread(img_p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 牙菌斑 (Plaque) 与牙周炎 (Gingivitis) 可视化逻辑区分
            if type_name == "Gingivitis":
                pass
            if type_name == "Plaque":
                # p0 通常代表健康，不画框；p1, p2 画对应框
                if grade_key != "p0":
                    img = draw_polygons(img, json_p, target_label=grade_key)
            
            plt.imshow(img)
            
            # 提取路径中的方向(F/L/R)
            # direction = Path(img_p).parts[-2] 
            # plt.title(f"{type_name} - Grade: {grade_key}\nID: {sample_id}_{direction}_{view_id}", 
            #           fontsize=10, fontweight='bold')
        else:
            plt.text(0.5, 0.5, f"Not Found:\n{sample_id}\n{view_id}", 
                     ha='center', va='center', color='red')
        
        plt.axis('off')
        
        # 2. 构造独立的保存路径，例如: tmp/plaque_p1.png
        output_filename = f"{type_name.lower()}_{grade_key}.png"
        output_path = os.path.join(save_dir, output_filename)
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close() # 必须关闭，防止内存溢出
        print(f"已保存: {output_path}")

if __name__ == "__main__":
    # 执行可视化，图像将输出到 ./tmp/ 目录下
    visualize_specific_targets(TARGETS_GINGIVITIS, "Gingivitis")
    visualize_specific_targets(TARGETS_PLAQUE, "Plaque")
# %%
