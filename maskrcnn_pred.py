#!/usr/bin/env python3
"""Evaluate unified Mask R-CNN checkpoints with saved diagnostics and visualizations."""

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
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from maskrcnn_unify import (
    add_panel_title,
    build_cfg,
    default_paths,
    load_categories,
    load_coco_dicts,
    register_datasets,
    safe_path_name,
    select_samples_per_class,
    draw_gt_panel,
    draw_pred_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unified Mask R-CNN caries/orth checkpoints.")
    parser.add_argument("--task", choices=["caries", "orth"], default="caries")
    parser.add_argument(
        "--config_file",
        default=None,
        type=Path,
        help="Training config. Defaults to config.yaml beside the checkpoint when available.",
    )
    parser.add_argument("--data_dir", default=None, type=Path)
    parser.add_argument("--train_json", default=None, type=Path)
    parser.add_argument("--test_json", default=None, type=Path)
    parser.add_argument("--weights", default="", help="Checkpoint path. Defaults to output_dir/model_final.pth.")
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to evaluate.",
    )
    parser.add_argument("--score_thresh", default=0.05, type=float)
    parser.add_argument("--vis_score_thresh", default=0.5, type=float)
    parser.add_argument("--vis_samples", default=32, type=int, help="Maximum visualizations per GT class.")
    parser.add_argument(
        "--save_raw_predictions",
        action="store_true",
        help="Save COCOEvaluator raw predictions under each split's inference directory.",
    )
    parser.add_argument("--keep_negative_ratio", default=None, type=float)
    parser.add_argument("--repeat_threshold", default=None, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def apply_prediction_paths(args: argparse.Namespace) -> argparse.Namespace:
    config_was_explicit = args.config_file is not None
    default_paths(args)

    if config_was_explicit:
        return args

    checkpoint_path = (
        Path(args.weights)
        if args.weights and "://" not in args.weights
        else args.output_dir / "model_final.pth"
    )
    saved_config = checkpoint_path.parent / "config.yaml"
    if saved_config.is_file():
        args.config_file = saved_config
    return args


def xywh_to_xyxy(boxes: list[list[float]]) -> list[list[float]]:
    return [[x, y, x + width, y + height] for x, y, width, height in boxes]


def inspect_coco_json(json_path: Path, split_name: str, output_dir: Path) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {image["id"]: image for image in coco_data["images"]}
    categories = sorted(coco_data["categories"], key=lambda item: item["id"])
    category_ids = {category["id"] for category in categories}
    annotations = coco_data["annotations"]
    images_with_annotations = {ann["image_id"] for ann in annotations}
    annotation_counts: dict[int, int] = defaultdict(int)
    invalid_category_count = 0
    invalid_bbox_count = 0
    out_of_bounds_bbox_count = 0
    missing_image_count = 0

    for ann in annotations:
        category_id = ann["category_id"]
        if category_id not in category_ids:
            invalid_category_count += 1
        else:
            annotation_counts[category_id] += 1

        image_info = images.get(ann["image_id"])
        if image_info is None:
            missing_image_count += 1
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            invalid_bbox_count += 1
            continue
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            invalid_bbox_count += 1
        if x < 0 or y < 0 or x + width > image_info["width"] or y + height > image_info["height"]:
            out_of_bounds_bbox_count += 1

    empty_categories = [
        category["name"]
        for category in categories
        if annotation_counts.get(category["id"], 0) == 0
    ]

    print(f"\n[{split_name}] COCO annotation sanity check")
    print(
        f"  images={len(images)} positives={len(images_with_annotations)} "
        f"negatives={len(images) - len(images_with_annotations)} "
        f"annotations={len(annotations)} categories={len(categories)}"
    )
    print(
        f"  classes_with_gt={len(categories) - len(empty_categories)}/{len(categories)} "
        f"empty_categories={empty_categories}"
    )
    print(
        f"  invalid_categories={invalid_category_count} "
        f"missing_images={missing_image_count} "
        f"invalid_bboxes={invalid_bbox_count} "
        f"out_of_bounds_bboxes={out_of_bounds_bbox_count}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [category["name"] for category in categories]
    counts = [annotation_counts.get(category["id"], 0) for category in categories]
    x_positions = np.arange(len(class_names))
    plt.figure(figsize=(max(12, len(class_names) * 0.45), 7))
    bars = plt.bar(x_positions, counts, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title(f"{split_name} GT Bounding Boxes per Category")
    plt.xlabel("Category")
    plt.ylabel("Bounding Box Count")
    plt.xticks(x_positions, class_names, rotation=60, ha="right")
    for bar, count in zip(bars, counts):
        if count > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    summary = (
        f"Images: {len(images)}    "
        f"Positive images: {len(images_with_annotations)}    "
        f"Negative images: {len(images) - len(images_with_annotations)}    "
        f"Annotations: {len(annotations)}"
    )
    plt.figtext(0.5, 0.01, summary, ha="center", fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_dir / "annotation_category_histogram.png")
    plt.close()


def inspect_category_consistency(train_json: Path, test_json: Path) -> None:
    with train_json.open("r", encoding="utf-8") as f:
        train_categories = sorted(json.load(f)["categories"], key=lambda item: item["id"])
    with test_json.open("r", encoding="utf-8") as f:
        test_categories = sorted(json.load(f)["categories"], key=lambda item: item["id"])

    train_pairs = [(category["id"], category["name"]) for category in train_categories]
    test_pairs = [(category["id"], category["name"]) for category in test_categories]
    if train_pairs == test_pairs:
        print("\n[category_check] train/test category ids and names match.")
    else:
        print("\n[category_check] train/test category mismatch detected.")
        print(f"  train={train_pairs}")
        print(f"  test={test_pairs}")


def run_coco_eval(
    cfg,
    dataset_name: str,
    output_dir: Path,
    save_raw_predictions: bool,
) -> dict:
    predictor = DefaultPredictor(cfg)
    evaluator_output_dir = str(output_dir / "inference") if save_raw_predictions else None
    evaluator = COCOEvaluator(dataset_name, output_dir=evaluator_output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def save_coco_results_txt(results: dict, class_names: list[str], output_dir: Path) -> None:
    lines = ["COCO evaluation"]
    for result_type, metrics in results.items():
        lines.extend(["", f"[{result_type}] Overall metrics"])
        for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
            value = metrics.get(key, float("nan"))
            try:
                lines.append(f"{key}: {float(value):.6f}")
            except (TypeError, ValueError):
                lines.append(f"{key}: {value}")

        lines.extend(["", f"[{result_type}] Per-category AP"])
        for class_name in class_names:
            key = f"AP-{class_name}"
            value = metrics.get(key, float("nan"))
            try:
                lines.append(f"{class_name}: {float(value):.6f}")
            except (TypeError, ValueError):
                lines.append(f"{class_name}: {value}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coco_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_score_distribution(cfg, dataset_dicts: list[dict], class_names: list[str], output_dir: Path) -> None:
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
            plt.axvline(np.median(scores), color="red", linestyle="dashed", linewidth=1, label="Median")
            plt.legend(fontsize=8)
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


def save_iou_report(cfg, dataset_dicts: list[dict], class_names: list[str], output_dir: Path) -> None:
    cfg = cfg.clone()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    predictor = DefaultPredictor(cfg)

    iou_stats_per_class: dict[int, list[float]] = defaultdict(list)
    gt_counts_per_class: dict[int, int] = defaultdict(int)

    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            gt_counts_per_class[ann["category_id"]] += 1

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
                g_boxes = Boxes(torch.as_tensor(xywh_to_xyxy([gt_boxes[idx] for idx in gt_idx]), dtype=torch.float32))
                max_ious = pairwise_iou(p_boxes, g_boxes).max(dim=1)[0].tolist()
                iou_stats_per_class[cls_id].extend(max_ious)
            elif pred_idx:
                iou_stats_per_class[cls_id].extend([0.0] * len(pred_idx))

    lines = ["Class,GTs,Predictions,AvgIoU,MaxIoU,MatchRateIoU>0.1"]
    print("\nIoU summary")
    print(f"{'Class':<20} {'GTs':>6} {'Pred':>6} {'AvgIoU':>8} {'MaxIoU':>8} {'Match>0.1':>10}")
    for cls_id, class_name in enumerate(class_names):
        values = iou_stats_per_class[cls_id]
        gt_count = gt_counts_per_class[cls_id]
        if values:
            avg_iou = float(np.mean(values))
            max_iou = float(np.max(values))
            match_rate = float(np.mean([iou > 0.1 for iou in values]))
        else:
            avg_iou = 0.0
            max_iou = 0.0
            match_rate = 0.0
        print(
            f"{class_name:<20} {gt_count:>6} {len(values):>6} "
            f"{avg_iou:>8.4f} {max_iou:>8.4f} {match_rate:>10.2%}"
        )
        lines.append(f"{class_name},{gt_count},{len(values)},{avg_iou:.6f},{max_iou:.6f},{match_rate:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "iou_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    samples_by_class = select_samples_per_class(dataset_dicts, class_names, limit, seed)
    draw_masks = bool(cfg.MODEL.MASK_ON)

    visualization_dir = output_dir / "visualizations" / "by_class"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    rendered_by_image_id: dict[int, np.ndarray] = {}

    for cls_id, samples in samples_by_class.items():
        class_dir = visualization_dir / f"{cls_id:02d}_{safe_path_name(class_names[cls_id])}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(samples):
            image_id = record["image_id"]
            comparison = rendered_by_image_id.get(image_id)
            if comparison is None:
                image = cv2.imread(record["file_name"])
                if image is None:
                    print(f"Could not read image for visualization: {record['file_name']}")
                    continue
                with torch.no_grad():
                    outputs = predictor(image)

                image_rgb = image[:, :, ::-1]
                gt_vis = draw_gt_panel(image_rgb, metadata, record, draw_masks)
                pred_vis = draw_pred_panel(image_rgb, metadata, outputs["instances"], draw_masks)
                gt_panel = add_panel_title(gt_vis, "GT")
                pred_panel = add_panel_title(pred_vis, f"Pred score>={score_thresh:g}")
                separator = np.full((gt_panel.shape[0], 8, 3), 255, dtype=np.uint8)
                comparison = np.concatenate([gt_panel, separator, pred_panel], axis=1)
                rendered_by_image_id[image_id] = comparison

            gt_class_ids = sorted(
                {
                    ann["category_id"]
                    for ann in record.get("annotations", [])
                    if 0 <= ann["category_id"] < len(class_names)
                }
            )
            class_suffix = "_".join(class_names[gt_cls_id] for gt_cls_id in gt_class_ids) or "negative"
            filename = f"gt_pred_{index:03d}_image_{image_id}_{safe_path_name(class_suffix)}.jpg"
            cv2.imwrite(str(class_dir / filename), comparison[:, :, ::-1])

    print(f"Saved visualizations to {visualization_dir}")


def evaluate_split(
    cfg,
    split_name: str,
    dataset_name: str,
    json_path: Path,
    class_names: list[str],
    seed: int,
    vis_samples: int,
    vis_score_thresh: float,
    save_raw_predictions: bool,
) -> None:
    print(f"\nEvaluating {split_name} split: {json_path}")
    split_output_dir = Path(cfg.OUTPUT_DIR) / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    inspect_coco_json(json_path, split_name, split_output_dir)
    dataset_dicts = list(DatasetCatalog.get(dataset_name))
    metadata = MetadataCatalog.get(dataset_name)

    coco_results = run_coco_eval(
        cfg,
        dataset_name,
        split_output_dir,
        save_raw_predictions,
    )
    save_coco_results_txt(coco_results, class_names, split_output_dir)
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
    args = apply_prediction_paths(parse_args())
    if not args.weights:
        args.weights = str(args.output_dir / "model_final.pth")

    print(f"Using inference config: {args.config_file}")
    class_names = load_categories(args.train_json, args.task)
    cfg = build_cfg(args, len(class_names))
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    register_datasets(args, class_names)
    inspect_category_consistency(args.train_json, args.test_json)

    split_configs = {
        "train": (f"{args.task}_train", args.train_json),
        "test": (f"{args.task}_val", args.test_json),
    }
    for split_name in args.eval_splits:
        dataset_name, json_path = split_configs[split_name]
        evaluate_split(
            cfg,
            split_name,
            dataset_name,
            json_path,
            class_names,
            seed=args.seed,
            vis_samples=args.vis_samples,
            vis_score_thresh=args.vis_score_thresh,
            save_raw_predictions=args.save_raw_predictions,
        )


if __name__ == "__main__":
    main()
