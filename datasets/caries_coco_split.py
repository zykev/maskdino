#!/usr/bin/env python3
"""Stratified train/test split of the caries COCO JSON by sample id.

Expects the migrated ``images/`` + ``anno/`` layout written by
``caries_to_coco.py``, where ``file_name`` looks like
``<sample_id>/images/<view>/<name>.png``. The sample id is the path component
immediately preceding the ``images`` marker, so every image of one tooth lands
in the same split. Samples are stratified by their highest category id, which
keeps the rarer lesions represented on both sides.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

DEFAULT_INPUT_JSON = ".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset.json"
DEFAULT_TRAIN_JSON = ".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_train.json"
DEFAULT_TEST_JSON = ".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified train/test split for the caries COCO JSON."
    )
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON, type=Path, help="Input COCO JSON.")
    parser.add_argument("--train-json", default=DEFAULT_TRAIN_JSON, type=Path, help="Output train COCO JSON.")
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON, type=Path, help="Output test COCO JSON.")
    parser.add_argument("--train-ratio", default=0.8, type=float, help="Train split ratio.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    return parser.parse_args()


def sample_id_from_file_name(file_name: str) -> str:
    """Return the sample id: the path component right before the ``images`` marker."""
    parts = Path(file_name).parts
    try:
        images_index = parts.index("images")
    except ValueError:
        raise ValueError(
            f"file_name {file_name!r} has no 'images' path component; expected the "
            "migrated <sample_id>/images/<view>/<name> layout."
        ) from None
    if images_index == 0:
        raise ValueError(f"file_name {file_name!r} has no sample id before 'images'.")
    return parts[images_index - 1]


def split_coco_dataset(
    input_json: Path,
    train_out: Path,
    test_out: Path,
    ratio: float = 0.8,
    seed: int = 42,
) -> None:
    # 1. 加载原始数据
    with open(input_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # 2. 建立索引：ImageID -> SampleID 和 ImageID -> Labels
    img_id_to_sample = {}
    for img in images:
        img_id_to_sample[img['id']] = sample_id_from_file_name(img['file_name'])

    img_id_to_labels = defaultdict(set)
    for anno in annotations:
        img_id_to_labels[anno['image_id']].add(anno['category_id'])

    # 3. 汇总每个 Sample 包含的所有标签
    sample_to_labels = defaultdict(set)
    sample_to_img_ids = defaultdict(list)
    
    for img in images:
        sid = img_id_to_sample[img['id']]
        sample_to_img_ids[sid].append(img['id'])
        if img['id'] in img_id_to_labels:
            sample_to_labels[sid].update(img_id_to_labels[img['id']])
        else:
            sample_to_labels[sid].add(0) # 0 代表健康样本

    # 4. 为每个样本确定一个用于分层的“代表标签”
    # 策略：选择该样本中 category_id 最大的那个（通常稀有病灶 ID 较大），或者最少的类别
    sample_list = []
    stratify_labels = []
    
    for sid, labels in sample_to_labels.items():
        sample_list.append(sid)
        # 简单策略：取最大 label ID 保证稀有类被考虑到
        stratify_labels.append(max(labels) if labels else 0)

    # --- 关键修复：处理孤儿类别 ---
    label_counts = Counter(stratify_labels)
    
    safe_samples = []
    safe_labels = []
    orphan_samples = []

    for sid, lbl in zip(sample_list, stratify_labels):
        if label_counts[lbl] < 2:
            orphan_samples.append(sid)
            print(f"⚠️ 类别 ID {lbl} 只有 1 个样本，已强制分配至训练集: {sid}")
        else:
            safe_samples.append(sid)
            safe_labels.append(lbl)

    if not safe_samples:
        train_samples, test_samples = train_test_split(sample_list, train_size=ratio, random_state=seed)
    else:
        # 对拥有 2 个以上成员的类别进行分层拆分
        train_samples, test_samples = train_test_split(
            safe_samples, 
            train_size=ratio, 
            stratify=safe_labels, 
            random_state=seed
        )
        # 将孤儿样本合并到训练集中
        train_samples = list(train_samples) + orphan_samples

    train_sample_set = set(train_samples)
    test_sample_set = set(test_samples)

    # 6. 根据拆分结果构造新的 COCO 结构
    def create_sub_coco(selected_samples):
        new_images = []
        new_annos = []
        selected_img_ids = set()

        # 筛选图片
        for img in images:
            if img_id_to_sample[img['id']] in selected_samples:
                new_images.append(img)
                selected_img_ids.add(img['id'])

        # 筛选标注
        for anno in annotations:
            if anno['image_id'] in selected_img_ids:
                new_annos.append(anno)

        return {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": categories,
            "images": new_images,
            "annotations": new_annos
        }

    train_coco = create_sub_coco(train_sample_set)
    test_coco = create_sub_coco(test_sample_set)

    # 7. 保存文件
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2)
    with open(test_out, 'w', encoding='utf-8') as f:
        json.dump(test_coco, f, indent=2)

    # 打印结果
    print(f"拆分完成！")
    print(f"训练集: {len(train_samples)} 样本, {len(train_coco['images'])} 图片, {len(train_coco['annotations'])} 标注")
    print(f"测试集: {len(test_samples)} 样本, {len(test_coco['images'])} 图片, {len(test_coco['annotations'])} 标注")
    
    # 验证类别均衡
    train_label_dist = Counter([a['category_id'] for a in train_coco['annotations']])
    test_label_dist = Counter([a['category_id'] for a in test_coco['annotations']])
    
    print("\n类别分布验证 (Train vs Test):")
    for cat in categories:
        cid = cat['id']
        t_count = train_label_dist.get(cid, 0)
        v_count = test_label_dist.get(cid, 0)
        total = t_count + v_count
        ratio_val = t_count / total if total > 0 else 0
        print(f" - {cat['name']:<25}: Train={t_count:<4} Test={v_count:<4} (Train Ratio: {ratio_val:.1%})")

if __name__ == "__main__":
    args = parse_args()
    split_coco_dataset(
        input_json=args.input_json,
        train_out=args.train_json,
        test_out=args.test_json,
        ratio=args.train_ratio,
        seed=args.seed,
    )