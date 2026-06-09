#!/usr/bin/env python3
"""Evaluate unified MaskDINO checkpoints for caries segmentation or orth detection."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from maskdino_unify import (
    apply_default_paths,
    configure_bbox_only_maskdino,
    load_categories,
    register_task_datasets,
)
from maskrcnn_unify import (
    add_panel_title,
    draw_gt_panel,
    draw_pred_panel,
    safe_path_name,
    select_samples_per_class,
)
from maskrcnn_pred import (
    inspect_category_consistency,
    inspect_coco_json,
    save_coco_results_txt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unified MaskDINO caries/orth checkpoints.")
    parser.add_argument("--task", choices=["caries", "orth"], default="caries")
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
    parser.add_argument("--vis_samples", default=8, type=int, help="Maximum visualizations per GT class.")
    parser.add_argument("--keep_negative_ratio", default=None, type=float)
    parser.add_argument("--repeat_threshold", default=None, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra config options in KEY VALUE form.")
    return parser.parse_args()


def xywh_to_xyxy(boxes: list[list[float]]) -> list[list[float]]:
    return [[x, y, x + width, y + height] for x, y, width, height in boxes]


def filter_instances(instances, score_thresh: float):
    instances = instances.to("cpu")
    if instances.has("scores"):
        instances = instances[instances.scores >= score_thresh]
    return instances


class MaskDINOPredictor:
    """Small predictor wrapper matching MaskDINO's meta-architecture input path."""

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        if getattr(self.cfg, "ORTH_BBOX_ONLY", False):
            configure_bbox_only_maskdino(self.model)
        self.model.eval()
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        self.input_format = self.cfg.INPUT.FORMAT
        # Match the eval-set loader transform (default DatasetMapper, is_train=False):
        # ResizeShortestEdge(MIN_SIZE_TEST, MAX_SIZE_TEST) so inputs match training-time eval.
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            return self.model([{"image": image, "height": height, "width": width}])[0]


def build_cfg(args: argparse.Namespace, num_classes: int):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.set_new_allowed(True)
    cfg.ORTH_BBOX_ONLY = args.task == "orth"

    cfg.merge_from_file(str(args.config_file))
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = (f"{args.task}_train",)
    cfg.DATASETS.TEST = (f"{args.task}_val",)
    cfg.OUTPUT_DIR = str(args.output_dir)

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if hasattr(cfg.MODEL, "MaskDINO"):
        cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
        cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = args.score_thresh
        if args.task == "orth":
            cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = True

    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


def run_coco_eval(cfg, dataset_name: str, output_dir: Path, task: str) -> dict:
    predictor = MaskDINOPredictor(cfg)
    tasks = ("bbox",) if task == "orth" else None
    evaluator = COCOEvaluator(dataset_name, tasks=tasks, output_dir=str(output_dir / "inference"))
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def save_score_distribution(
    cfg,
    dataset_dicts: list[dict],
    class_names: list[str],
    output_dir: Path,
    score_thresh: float,
) -> None:
    predictor = MaskDINOPredictor(cfg)
    scores_per_class: dict[int, list[float]] = defaultdict(list)
    gt_counts_per_class: dict[int, int] = defaultdict(int)

    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            gt_counts_per_class[ann["category_id"]] += 1

        image = cv2.imread(record["file_name"])
        if image is None:
            continue
        instances = filter_instances(predictor(image)["instances"], score_thresh)
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


def save_iou_report(
    cfg,
    dataset_dicts: list[dict],
    class_names: list[str],
    output_dir: Path,
    score_thresh: float,
) -> None:
    predictor = MaskDINOPredictor(cfg)
    iou_stats_per_class: dict[int, list[float]] = defaultdict(list)
    gt_counts_per_class: dict[int, int] = defaultdict(int)

    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            gt_counts_per_class[ann["category_id"]] += 1

        image = cv2.imread(record["file_name"])
        if image is None:
            continue
        instances = filter_instances(predictor(image)["instances"], score_thresh)

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
    draw_masks: bool,
) -> None:
    if limit <= 0:
        return

    predictor = MaskDINOPredictor(cfg)
    samples_by_class = select_samples_per_class(dataset_dicts, class_names, limit, seed)

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

                outputs = predictor(image)
                instances = filter_instances(outputs["instances"], score_thresh)

                image_rgb = image[:, :, ::-1]
                gt_vis = draw_gt_panel(image_rgb, metadata, record, draw_masks)
                pred_vis = draw_pred_panel(image_rgb, metadata, instances, draw_masks)
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
    task: str,
    split_name: str,
    dataset_name: str,
    json_path: Path,
    class_names: list[str],
    seed: int,
    vis_samples: int,
    score_thresh: float,
    vis_score_thresh: float,
) -> None:
    print(f"\nEvaluating {split_name} split: {json_path}")
    split_output_dir = Path(cfg.OUTPUT_DIR) / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    inspect_coco_json(json_path, split_name, split_output_dir)
    dataset_dicts = list(DatasetCatalog.get(dataset_name))
    metadata = MetadataCatalog.get(dataset_name)

    coco_results = run_coco_eval(cfg, dataset_name, split_output_dir, task)
    save_coco_results_txt(coco_results, class_names, split_output_dir)
    save_score_distribution(cfg, dataset_dicts, class_names, split_output_dir, score_thresh)
    save_iou_report(cfg, dataset_dicts, class_names, split_output_dir, score_thresh)
    save_visualizations(
        cfg,
        dataset_dicts,
        metadata,
        class_names,
        split_output_dir,
        limit=vis_samples,
        score_thresh=vis_score_thresh,
        seed=seed,
        draw_masks=task == "caries",
    )


def main() -> None:
    setup_logger()
    args = parse_args()
    apply_default_paths(args)
    if args.keep_negative_ratio is None:
        args.keep_negative_ratio = 0.1 if args.task == "caries" else 0.2
    if not args.weights:
        args.weights = str(args.output_dir / "model_final.pth")

    class_names = load_categories(args.train_json, args.task)
    cfg = build_cfg(args, len(class_names))
    register_task_datasets(args, class_names)
    inspect_category_consistency(args.train_json, args.test_json)

    split_configs = {
        "train": (f"{args.task}_train", args.train_json),
        "test": (f"{args.task}_val", args.test_json),
    }
    for split_name in args.eval_splits:
        dataset_name, json_path = split_configs[split_name]
        evaluate_split(
            cfg,
            args.task,
            split_name,
            dataset_name,
            json_path,
            class_names,
            seed=args.seed,
            vis_samples=args.vis_samples,
            score_thresh=args.score_thresh,
            vis_score_thresh=args.vis_score_thresh,
        )


if __name__ == "__main__":
    main()
