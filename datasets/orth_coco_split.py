#!/usr/bin/env python3
"""Split an orthodontic COCO detection JSON into train/test sets by sample id."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-aware train/test split for COCO JSON.")
    parser.add_argument(
        "--input-json",
        default=".datasets/intraoral_anno/orth_0616/orth_detection_coco.json",
        type=Path,
        help="Input COCO JSON.",
    )
    parser.add_argument(
        "--train-json",
        default=".datasets/intraoral_anno/orth_0616/orth_detection_train.json",
        type=Path,
        help="Output train COCO JSON.",
    )
    parser.add_argument(
        "--test-json",
        default=".datasets/intraoral_anno/orth_0616/orth_detection_test.json",
        type=Path,
        help="Output test COCO JSON.",
    )
    parser.add_argument("--train-ratio", default=0.8, type=float, help="Train split ratio.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    return parser.parse_args()


def sample_id_from_file_name(file_name: str) -> str:
    parts = Path(file_name).parts
    return parts[0] if len(parts) > 1 else Path(file_name).stem


def representative_label(labels: set[int], category_counts: Counter[int]) -> int:
    if not labels:
        return 0
    return min(labels, key=lambda label: (category_counts[label], label))


def split_coco_dataset(
    input_json: Path,
    train_json: Path,
    test_json: Path,
    train_ratio: float,
    seed: int,
) -> None:
    with input_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_to_sample = {image["id"]: sample_id_from_file_name(image["file_name"]) for image in images}
    image_to_labels: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        image_to_labels[ann["image_id"]].add(ann["category_id"])

    sample_to_images: dict[str, list[int]] = defaultdict(list)
    sample_to_labels: dict[str, set[int]] = defaultdict(set)
    for image in images:
        sample_id = image_to_sample[image["id"]]
        sample_to_images[sample_id].append(image["id"])
        labels = image_to_labels.get(image["id"])
        if labels:
            sample_to_labels[sample_id].update(labels)
        else:
            sample_to_labels[sample_id].add(0)

    category_counts = Counter(
        label for labels in sample_to_labels.values() for label in labels if label != 0
    )
    sample_labels = {
        sample_id: representative_label(labels, category_counts)
        for sample_id, labels in sample_to_labels.items()
    }

    rng = random.Random(seed)
    label_to_samples: dict[int, list[str]] = defaultdict(list)
    for sample_id, label in sample_labels.items():
        label_to_samples[label].append(sample_id)

    train_samples: set[str] = set()
    test_samples: set[str] = set()
    for label, sample_ids in sorted(label_to_samples.items()):
        rng.shuffle(sample_ids)
        if len(sample_ids) < 2:
            train_samples.update(sample_ids)
            continue
        train_count = max(1, min(len(sample_ids) - 1, round(len(sample_ids) * train_ratio)))
        train_samples.update(sample_ids[:train_count])
        test_samples.update(sample_ids[train_count:])

    if not test_samples:
        all_samples = sorted(sample_to_images)
        rng.shuffle(all_samples)
        test_count = max(1, round(len(all_samples) * (1.0 - train_ratio)))
        test_samples = set(all_samples[:test_count])
        train_samples = set(all_samples[test_count:])

    selected_train_image_ids = {
        image_id for sample_id in train_samples for image_id in sample_to_images[sample_id]
    }
    selected_test_image_ids = {
        image_id for sample_id in test_samples for image_id in sample_to_images[sample_id]
    }

    def build_subset(selected_image_ids: set[int]) -> dict:
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": categories,
            "images": [image for image in images if image["id"] in selected_image_ids],
            "annotations": [ann for ann in annotations if ann["image_id"] in selected_image_ids],
        }

    train_coco = build_subset(selected_train_image_ids)
    test_coco = build_subset(selected_test_image_ids)

    train_json.parent.mkdir(parents=True, exist_ok=True)
    test_json.parent.mkdir(parents=True, exist_ok=True)
    with train_json.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2)
    with test_json.open("w", encoding="utf-8") as f:
        json.dump(test_coco, f, indent=2)

    print("Split complete")
    print(
        f"Train: {len(train_samples)} samples, "
        f"{len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations"
    )
    print(
        f"Test:  {len(test_samples)} samples, "
        f"{len(test_coco['images'])} images, {len(test_coco['annotations'])} annotations"
    )

    train_dist = Counter(ann["category_id"] for ann in train_coco["annotations"])
    test_dist = Counter(ann["category_id"] for ann in test_coco["annotations"])
    print("\nClass distribution:")
    for category in categories:
        category_id = category["id"]
        print(
            f" - {category['name']:<20} "
            f"Train={train_dist.get(category_id, 0):<4} "
            f"Test={test_dist.get(category_id, 0):<4}"
        )


if __name__ == "__main__":
    args = parse_args()
    split_coco_dataset(
        input_json=args.input_json,
        train_json=args.train_json,
        test_json=args.test_json,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
