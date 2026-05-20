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
    parser.add_argument("--vis-samples", default=32, type=int)
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


def run_coco_eval(cfg) -> dict:
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("orth_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    val_loader = build_detection_test_loader(cfg, "orth_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def save_score_distribution(cfg, dataset_dicts: list[dict], class_names: list[str]) -> None:
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
    plt.savefig(Path(cfg.OUTPUT_DIR) / "confidence_score_distribution.png")
    plt.close()


def save_iou_report(cfg, dataset_dicts: list[dict], class_names: list[str]) -> None:
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

    report_path = Path(cfg.OUTPUT_DIR) / "iou_summary.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_visualizations(
    cfg,
    dataset_dicts: list[dict],
    metadata,
    limit: int,
    score_thresh: float,
    seed: int,
) -> None:
    if limit <= 0:
        return

    cfg = cfg.clone()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    predictor = DefaultPredictor(cfg)

    random.seed(seed)
    samples = random.sample(dataset_dicts, min(limit, len(dataset_dicts)))
    output_dir = Path(cfg.OUTPUT_DIR) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(samples):
        image = cv2.imread(record["file_name"])
        if image is None:
            continue
        with torch.no_grad():
            outputs = predictor(image)
        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
        vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(str(output_dir / f"orth_pred_{index:03d}.jpg"), vis.get_image()[:, :, ::-1])

    print(f"Saved visualizations to {output_dir}")


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

    dataset_dicts_val = load_coco_dicts(args.data_dir, args.test_json)
    metadata = MetadataCatalog.get("orth_val")

    run_coco_eval(cfg)
    save_score_distribution(cfg, dataset_dicts_val, class_names)
    save_iou_report(cfg, dataset_dicts_val, class_names)
    save_visualizations(
        cfg,
        dataset_dicts_val,
        metadata,
        limit=args.vis_samples,
        score_thresh=args.vis_score_thresh,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
