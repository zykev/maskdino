import os
import random
import torch
import torchvision.transforms as T
from PIL import Image

random.seed(5)

def augment_and_save_separately(image_path, output_dir="aug_results"):
    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 加载原始图片
    img = Image.open(image_path).convert("RGB")

    # 3. 定义变换池 (包含常见的几种增强)
    transforms_pool = [
        # ("HorizontalFlip", T.RandomHorizontalFlip(p=1.0)),
        # ("Rotation", T.RandomRotation(degrees=30)),
        ("ColorJitter", T.ColorJitter(brightness=0.5)),
        # ("GaussianBlur", T.GaussianBlur(kernel_size=5)),
        ("Brightness", T.ColorJitter(brightness=0.8)),
        # 对比度调节：参数 (0.5, 1.5) 表示对比度在原图的 50% 到 150% 之间随机
        # ("Contrast", T.ColorJitter(contrast=(0.2, 1.8))),
        # ("VerticalFlip", T.RandomVerticalFlip(p=1.0))
    ]

    # 4. 随机选取两种不同的方法
    selected_transforms = random.sample(transforms_pool, 2)

    # 5. 分别应用并保存
    for i, (name, transform) in enumerate(selected_transforms):
        # 对原图直接进行变换
        aug_img = transform(img)
        
        # 构造保存路径，例如: aug_1_Rotation.png
        save_name = f"aug_{i+1}_{name}.png"
        save_path = os.path.join(output_dir, save_name)
        
        # 保存图片
        aug_img.save(save_path)
        print(f"已保存第 {i+1} 张图: {save_path} (使用方法: {name})")

if __name__ == "__main__":
    # 替换为你的牙齿图片路径
    img_file = ".datasets/intraoral/F 2.png" 
    
    if os.path.exists(img_file):
        augment_and_save_separately(img_file)
    else:
        print(f"找不到文件: {img_file}")