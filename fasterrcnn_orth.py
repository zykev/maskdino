#!/usr/bin/env python3
"""Train and evaluate Detectron2 Faster R-CNN on orth_test COCO annotations."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faster R-CNN training for tooth anomaly detection.")
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_test/orth_test",
        type=Path,
        help="Image root. COCO file_name entries are relative to this folder.",
    )
    parser.add_argument(
        "--train-json",
        default=".datasets/intraoral_anno/orth_test/orth_detection_train.json",
        type=Path,
        help="COCO train JSON.",
    )
    parser.add_argument(
        "--test-json",
        default=".datasets/intraoral_anno/orth_test/orth_detection_test.json",
        type=Path,
        help="COCO test JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/fasterrcnn_orth",
        type=Path,
        help="Training output directory.",
    )
    parser.add_argument("--max-iter", default=10000, type=int)
    parser.add_argument("--eval-period", default=500, type=int)
    parser.add_argument("--ims-per-batch", default=2, type=int)
    parser.add_argument("--base-lr", default=0.0025, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--score-thresh", default=0.5, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate --weights or output_dir/model_final.pth.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional model weights for resume/eval/predict. Defaults to COCO init for training.",
    )
    parser.add_argument(
        "--vis-samples",
        default=16,
        type=int,
        help="Number of validation predictions to save after training/eval.",
    )
    return parser.parse_args()


def load_categories(json_path: Path) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [category["name"] for category in sorted(data["categories"], key=lambda item: item["id"])]


def register_datasets(data_dir: Path, train_json: Path, test_json: Path) -> list[str]:
    class_names = load_categories(train_json)
    for name, json_path in {
        "orth_train": train_json,
        "orth_val": test_json,
    }.items():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, str(json_path), str(data_dir))
        MetadataCatalog.get(name).set(thing_classes=class_names)
    return class_names


def build_cfg(args: argparse.Namespace, num_classes: int):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("orth_train",)
    cfg.DATASETS.TEST = ("orth_val",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.1

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 800, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.7), int(args.max_iter * 0.9))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(1, args.max_iter // 20))
    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.OUTPUT_DIR = str(args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    elif args.eval_only:
        cfg.MODEL.WEIGHTS = str(args.output_dir / "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    return cfg


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def evaluate(cfg) -> dict:
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("orth_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    val_loader = build_detection_test_loader(cfg, "orth_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def save_visualizations(cfg, limit: int) -> None:
    if limit <= 0:
        return

    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get("orth_val")
    metadata = MetadataCatalog.get("orth_val")
    random.seed(42)
    samples = random.sample(dataset_dicts, min(limit, len(dataset_dicts)))
    output_dir = Path(cfg.OUTPUT_DIR) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(samples):
        image = cv2.imread(record["file_name"])
        if image is None:
            continue
        outputs = predictor(image)
        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
        vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        save_path = output_dir / f"orth_pred_{index:03d}.jpg"
        cv2.imwrite(str(save_path), vis.get_image()[:, :, ::-1])

    print(f"Saved visualizations to {output_dir}")


def main() -> None:
    setup_logger()
    args = parse_args()
    class_names = register_datasets(args.data_dir, args.train_json, args.test_json)
    cfg = build_cfg(args, num_classes=len(class_names))

    if args.eval_only:
        evaluate(cfg)
        save_visualizations(cfg, args.vis_samples)
        return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    evaluate(cfg)
    save_visualizations(cfg, args.vis_samples)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
