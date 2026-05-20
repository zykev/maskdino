#!/usr/bin/env python3
"""Evaluate a trained Faster R-CNN checkpoint on the orthodontic COCO split."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from fasterrcnn_orth import build_cfg, load_categories, register_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Faster R-CNN orth checkpoint.")
    parser.add_argument(
        "--config-file",
        default="configs/default_fasterrcnn_orth_config.yaml",
        type=Path,
    )
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_test/orth_test",
        type=Path,
    )
    parser.add_argument(
        "--train-json",
        default=".datasets/intraoral_anno/orth_test/orth_detection_train.json",
        type=Path,
    )
    parser.add_argument(
        "--test-json",
        default=".datasets/intraoral_anno/orth_test/orth_detection_test.json",
        type=Path,
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to evaluate. Defaults to both train_json and test_json.",
    )
    parser.add_argument(
        "--weights",
        default="output/fasterrcnn_orth/model_final.pth",
        help="Trained checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/fasterrcnn_orth_pred",
        type=Path,
    )
    parser.add_argument("--score-thresh", default=0.05, type=float)
    parser.add_argument("--vis-score-thresh", default=0.5, type=float)
    parser.add_argument(
        "--vis-samples",
        default=32,
        type=int,
        help="Maximum GT/Pred visualization images to save per class.",
    )
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def load_coco_dicts(data_dir: Path, json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {image["id"]: image for image in coco_data["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    dataset_dicts = []
    for image_id, image_info in images.items():
        record = {
            "file_name": str(data_dir / image_info["file_name"]),
            "image_id": image_id,
            "height": image_info["height"],
            "width": image_info["width"],
            "annotations": [],
        }
        for ann in annotations_by_image.get(image_id, []):
            record["annotations"].append(
                {
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": ann["category_id"] - 1,
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
        dataset_dicts.append(record)
    return dataset_dicts


def xywh_to_xyxy(boxes: list[list[float]]) -> list[list[float]]:
    return [[x, y, x + width, y + height] for x, y, width, height in boxes]


def make_cfg(args: argparse.Namespace, num_classes: int):
    train_args = argparse.Namespace(
        config_file=args.config_file,
        num_workers=None,
        repeat_threshold=None,
        keep_negative_ratio=None,
        score_thresh=args.score_thresh,
        ims_per_batch=None,
        base_lr=None,
        max_iter=None,
        eval_period=None,
        output_dir=args.output_dir,
        weights=args.weights,
        eval_only=True,
    )
    cfg = build_cfg(train_args, num_classes=num_classes)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.OUTPUT_DIR = str(args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def run_coco_eval(cfg, dataset_name: str, output_dir: Path) -> dict:
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=str(output_dir / "inference"))
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def save_score_distribution(
    cfg,
    dataset_dicts: list[dict],
    class_names: list[str],
    output_dir: Path,
) -> None:
    cfg = cfg.clone()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    predictor = DefaultPredictor(cfg)

    scores_per_class: dict[int, list[float]] = defaultdict(list)
    gt_counts_per_class: dict[int, int] = defaultdict(int)

    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            gt_counts_per_class[ann["category_id"]] += 1

        image = cv2.imread(record["file_name"])
        if image is None:
            continue
        with torch.no_grad():
            instances = predictor(image)["instances"].to("cpu")
        for cls_id, score in zip(instances.pred_classes.tolist(), instances.scores.tolist()):
            scores_per_class[cls_id].append(score)

    cols = 4
    rows = int(np.ceil(len(class_names) / cols))
    plt.figure(figsize=(cols * 5, rows * 4))
    plt.suptitle("Confidence Score Distribution per Class", fontsize=18)

    for cls_id, class_name in enumerate(class_names):
        plt.subplot(rows, cols, cls_id + 1)
        scores = scores_per_class[cls_id]
        title = f"{class_name}\nGT={gt_counts_per_class[cls_id]} Pred={len(scores)}"
        if scores:
            plt.hist(scores, bins=30, color="skyblue", edgecolor="black", alpha=0.75)
            plt.axvline(np.mean(scores), color="red", linestyle="dashed", linewidth=1)
        else:
            plt.text(0.5, 0.5, "No Predictions", ha="center", va="center")
        plt.title(title)
        plt.xlim(0.0, 1.0)
        plt.xlabel("Confidence")
        plt.ylabel("Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "confidence_score_distribution.png")
    plt.close()


def save_iou_report(
    cfg,
    dataset_dicts: list[dict],
    class_names: list[str],
    output_dir: Path,
) -> None:
    cfg = cfg.clone()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    predictor = DefaultPredictor(cfg)

    iou_stats_per_class: dict[int, list[float]] = defaultdict(list)

    for record in dataset_dicts:
        image = cv2.imread(record["file_name"])
        if image is None:
            continue

        with torch.no_grad():
            instances = predictor(image)["instances"].to("cpu")

        pred_boxes = instances.pred_boxes
        pred_classes = instances.pred_classes.tolist()
        gt_boxes = [ann["bbox"] for ann in record["annotations"]]
        gt_classes = [ann["category_id"] for ann in record["annotations"]]
        if not gt_boxes and not pred_classes:
            continue

        for cls_id in range(len(class_names)):
            pred_idx = [idx for idx, cls in enumerate(pred_classes) if cls == cls_id]
            gt_idx = [idx for idx, cls in enumerate(gt_classes) if cls == cls_id]
            if pred_idx and gt_idx:
                p_boxes = pred_boxes[pred_idx]
                g_boxes = Boxes(
                    torch.as_tensor(
                        xywh_to_xyxy([gt_boxes[idx] for idx in gt_idx]),
                        dtype=torch.float32,
                    )
                )
                max_ious = pairwise_iou(p_boxes, g_boxes).max(dim=1)[0].tolist()
                iou_stats_per_class[cls_id].extend(max_ious)
            elif pred_idx:
                iou_stats_per_class[cls_id].extend([0.0] * len(pred_idx))

    lines = ["Class,Predictions,AvgIoU,MaxIoU,MatchRateIoU>0.1"]
    print("\nIoU summary")
    print(f"{'Class':<20} {'Pred':>6} {'AvgIoU':>8} {'MaxIoU':>8} {'Match>0.1':>10}")
    for cls_id, class_name in enumerate(class_names):
        values = iou_stats_per_class[cls_id]
        if values:
            avg_iou = float(np.mean(values))
            max_iou = float(np.max(values))
            match_rate = float(np.mean([iou > 0.1 for iou in values]))
        else:
            avg_iou = 0.0
            max_iou = 0.0
            match_rate = 0.0
        print(f"{class_name:<20} {len(values):>6} {avg_iou:>8.4f} {max_iou:>8.4f} {match_rate:>10.2%}")
        lines.append(f"{class_name},{len(values)},{avg_iou:.6f},{max_iou:.6f},{match_rate:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "iou_summary.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_samples_per_class(
    dataset_dicts: list[dict],
    num_classes: int,
    limit: int,
    seed: int,
) -> dict[int, list[dict]]:
    rng = random.Random(seed)
    records_by_class: dict[int, list[dict]] = defaultdict(list)
    for record in dataset_dicts:
        class_ids = {ann["category_id"] for ann in record.get("annotations", [])}
        for cls_id in class_ids:
            if 0 <= cls_id < num_classes:
                records_by_class[cls_id].append(record)
            else:
                print(f"  skipped invalid GT class id {cls_id} in image {record['image_id']}")

    samples_by_class: dict[int, list[dict]] = {}
    print("\nVisualization samples per class")
    for records in records_by_class.values():
        rng.shuffle(records)

    for cls_id in range(num_classes):
        candidates = records_by_class.get(cls_id, [])
        samples_by_class[cls_id] = candidates[: min(limit, len(candidates))]
        if not candidates:
            print(f"  class {cls_id}: 0 GT images, skipped")
        elif len(candidates) < limit:
            print(f"  class {cls_id}: only {len(candidates)} GT images, saved all")
        else:
            print(f"  class {cls_id}: saved {limit} of {len(candidates)} GT images")

    return samples_by_class


def safe_path_name(name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe_name.strip("._") or "class"


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    header_height = 36
    titled = np.full(
        (image_rgb.shape[0] + header_height, image_rgb.shape[1], 3),
        255,
        dtype=np.uint8,
    )
    titled[header_height:, :, :] = image_rgb
    cv2.putText(
        titled,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return titled


def save_visualizations(
    cfg,
    dataset_dicts: list[dict],
    metadata,
    class_names: list[str],
    output_dir: Path,
    limit: int,
    score_thresh: float,
    seed: int,
) -> None:
    if limit <= 0:
        return

    cfg = cfg.clone()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    predictor = DefaultPredictor(cfg)

    samples_by_class = select_samples_per_class(dataset_dicts, len(class_names), limit, seed)
    visualization_dir = output_dir / "visualizations"
    class_output_dir = visualization_dir / "by_class"
    class_output_dir.mkdir(parents=True, exist_ok=True)
    visualizations_by_image_id: dict[int, np.ndarray] = {}

    for cls_id, samples in samples_by_class.items():
        per_class_dir = class_output_dir / f"{cls_id:02d}_{safe_path_name(class_names[cls_id])}"
        per_class_dir.mkdir(parents=True, exist_ok=True)

        for index, record in enumerate(samples):
            image_id = record["image_id"]
            comparison = visualizations_by_image_id.get(image_id)
            if comparison is None:
                image = cv2.imread(record["file_name"])
                if image is None:
                    print(f"Could not read image for visualization: {record['file_name']}")
                    continue
                with torch.no_grad():
                    outputs = predictor(image)

                image_rgb = image[:, :, ::-1]
                gt_visualizer = Visualizer(image_rgb, metadata=metadata, scale=0.8)
                pred_visualizer = Visualizer(image_rgb, metadata=metadata, scale=0.8)
                gt_vis = gt_visualizer.draw_dataset_dict(record).get_image()
                pred_vis = pred_visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

                gt_panel = add_panel_title(gt_vis, "GT")
                pred_panel = add_panel_title(pred_vis, f"Pred score>={score_thresh:g}")
                separator = np.full((gt_panel.shape[0], 8, 3), 255, dtype=np.uint8)
                comparison = np.concatenate([gt_panel, separator, pred_panel], axis=1)
                visualizations_by_image_id[image_id] = comparison

            gt_class_ids = sorted(
                {
                    ann["category_id"]
                    for ann in record.get("annotations", [])
                    if 0 <= ann["category_id"] < len(class_names)
                }
            )
            class_suffix = "_".join(class_names[gt_cls_id] for gt_cls_id in gt_class_ids) or "negative"
            filename = f"orth_gt_pred_{index:03d}_image_{image_id}_{safe_path_name(class_suffix)}.jpg"
            cv2.imwrite(str(per_class_dir / filename), comparison[:, :, ::-1])

    print(f"Saved visualizations to {visualization_dir}")


def evaluate_split(
    cfg,
    split_name: str,
    dataset_name: str,
    data_dir: Path,
    json_path: Path,
    class_names: list[str],
    seed: int,
    vis_samples: int,
    vis_score_thresh: float,
) -> None:
    print(f"\nEvaluating {split_name} split: {json_path}")
    split_output_dir = Path(cfg.OUTPUT_DIR) / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dicts = load_coco_dicts(data_dir, json_path)
    metadata = MetadataCatalog.get(dataset_name)

    run_coco_eval(cfg, dataset_name, split_output_dir)
    save_score_distribution(cfg, dataset_dicts, class_names, split_output_dir)
    save_iou_report(cfg, dataset_dicts, class_names, split_output_dir)
    save_visualizations(
        cfg,
        dataset_dicts,
        metadata,
        class_names,
        split_output_dir,
        limit=vis_samples,
        score_thresh=vis_score_thresh,
        seed=seed,
    )


def main() -> None:
    setup_logger()
    args = parse_args()
    class_names = load_categories(args.train_json)
    cfg = make_cfg(args, num_classes=len(class_names))

    register_datasets(
        args.data_dir,
        args.train_json,
        args.test_json,
        keep_negative_ratio=1.0,
        seed=args.seed,
    )

    split_configs = {
        "train": ("orth_train", args.train_json),
        "test": ("orth_val", args.test_json),
    }
    for split_name in args.eval_splits:
        dataset_name, json_path = split_configs[split_name]
        evaluate_split(
            cfg,
            split_name,
            dataset_name,
            args.data_dir,
            json_path,
            class_names,
            seed=args.seed,
            vis_samples=args.vis_samples,
            vis_score_thresh=args.vis_score_thresh,
        )


if __name__ == "__main__":
    main()
