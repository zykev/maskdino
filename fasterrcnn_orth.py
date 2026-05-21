#!/usr/bin/env python3
"""Train and evaluate Detectron2 Faster R-CNN on orth_test COCO annotations."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faster R-CNN training for tooth anomaly detection.")
    parser.add_argument(
        "--config-file",
        default="configs/default_fasterrcnn_orth_config.yaml",
        type=Path,
        help="Local Detectron2 YAML with tunable Faster R-CNN settings.",
    )
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
        default=None,
        type=Path,
        help="Training output directory. Overrides OUTPUT_DIR in --config-file.",
    )
    parser.add_argument("--max-iter", default=None, type=int, help="Overrides SOLVER.MAX_ITER.")
    parser.add_argument("--eval-period", default=None, type=int, help="Overrides TEST.EVAL_PERIOD.")
    parser.add_argument("--ims-per-batch", default=None, type=int, help="Overrides SOLVER.IMS_PER_BATCH.")
    parser.add_argument("--base-lr", default=None, type=float, help="Overrides SOLVER.BASE_LR.")
    parser.add_argument("--num-workers", default=None, type=int, help="Overrides DATALOADER.NUM_WORKERS.")
    parser.add_argument("--score-thresh", default=None, type=float, help="Overrides MODEL.ROI_HEADS.SCORE_THRESH_TEST.")
    parser.add_argument(
        "--keep-negative-ratio",
        default=None,
        type=float,
        help=(
            "Overrides DATALOADER.KEEP_NEGATIVE_RATIO in --config-file. "
            "Controls how many empty-annotation normal images are kept."
        ),
    )
    parser.add_argument(
        "--repeat-threshold",
        default=None,
        type=float,
        help="Overrides DATALOADER.REPEAT_THRESHOLD for RepeatFactorTrainingSampler.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed for negative-sample retention.")
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


def get_orth_dicts(
    data_dir: Path,
    json_path: Path,
    is_train: bool,
    keep_negative_ratio: float,
    seed: int,
) -> list[dict]:
    if not 0.0 <= keep_negative_ratio <= 1.0:
        raise ValueError("--keep-negative-ratio must be between 0.0 and 1.0")

    with json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {image["id"]: image for image in coco_data["images"]}
    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for annotation in coco_data["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    rng = random.Random(seed)
    dataset_dicts: list[dict] = []
    negative_count = 0
    kept_negative_count = 0

    for image_id, image_info in images.items():
        anns = annotations_by_image.get(image_id, [])
        if not anns:
            negative_count += 1
            if is_train and rng.random() > keep_negative_ratio:
                continue
            kept_negative_count += 1

        record = {
            "file_name": str(data_dir / image_info["file_name"]),
            "image_id": image_id,
            "height": image_info["height"],
            "width": image_info["width"],
            "annotations": [],
        }

        for ann in anns:
            record["annotations"].append(
                {
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": ann["category_id"] - 1,
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )

        dataset_dicts.append(record)

    split_name = "train" if is_train else "val"
    print(
        f"[orth_{split_name}] loaded {len(dataset_dicts)} images from {json_path}; "
        f"kept {kept_negative_count}/{negative_count} empty negative samples"
    )
    return dataset_dicts


def register_datasets(
    data_dir: Path,
    train_json: Path,
    test_json: Path,
    keep_negative_ratio: float,
    seed: int,
) -> list[str]:
    class_names = load_categories(train_json)
    for name, json_path in {
        "orth_train": train_json,
        "orth_val": test_json,
    }.items():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)

        is_train = name == "orth_train"
        DatasetCatalog.register(
            name,
            lambda json_path=json_path, is_train=is_train: get_orth_dicts(
                data_dir=data_dir,
                json_path=json_path,
                is_train=is_train,
                keep_negative_ratio=keep_negative_ratio,
                seed=seed,
            ),
        )
        MetadataCatalog.get(name).set(
            thing_classes=class_names,
            evaluator_type="coco",
            json_file=str(json_path),
            image_root=str(data_dir),
            thing_dataset_id_to_contiguous_id={
                category_id: category_id - 1 for category_id in range(1, len(class_names) + 1)
            },
        )
    return class_names


def build_cfg(args: argparse.Namespace, num_classes: int):
    cfg = get_cfg()
    cfg.DATALOADER.KEEP_NEGATIVE_RATIO = 0.5

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    if args.config_file:
        cfg.merge_from_file(str(args.config_file))

    cfg.DATASETS.TRAIN = ("orth_train",)
    cfg.DATASETS.TEST = ("orth_val",)
    if args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    if args.repeat_threshold is not None:
        cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold
    elif cfg.DATALOADER.REPEAT_THRESHOLD <= 0:
        cfg.DATALOADER.REPEAT_THRESHOLD = 0.05
    if args.keep_negative_ratio is not None:
        cfg.DATALOADER.KEEP_NEGATIVE_RATIO = args.keep_negative_ratio

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if args.score_thresh is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh

    if args.ims_per_batch is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    if args.base_lr is not None:
        cfg.SOLVER.BASE_LR = args.base_lr
    if args.max_iter is not None:
        cfg.SOLVER.MAX_ITER = args.max_iter
        cfg.SOLVER.STEPS = (int(args.max_iter * 0.7), int(args.max_iter * 0.9))
        cfg.SOLVER.WARMUP_ITERS = min(1000, max(1, args.max_iter // 20))
    if args.eval_period is not None:
        cfg.TEST.EVAL_PERIOD = args.eval_period

    if args.output_dir is not None:
        cfg.OUTPUT_DIR = str(args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    elif args.eval_only:
        cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    elif not cfg.MODEL.WEIGHTS:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    return cfg


class LossEvalHook(HookBase):
    def __init__(self, eval_period: int, model, data_loader):
        self._period = eval_period
        self._model = model
        self._data_loader = data_loader

    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            key: value.detach().cpu().item() if isinstance(value, torch.Tensor) else float(value)
            for key, value in metrics_dict.items()
        }
        total_loss = sum(metrics_dict.values())
        return total_loss, metrics_dict

    def _do_loss_eval(self):
        losses = []
        loss_by_name: dict[str, list[float]] = defaultdict(list)

        self._model.train()
        with torch.no_grad():
            for inputs in self._data_loader:
                total_loss, metrics_dict = self._get_loss(inputs)
                losses.append(total_loss)
                for name, value in metrics_dict.items():
                    loss_by_name[name].append(value)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        for name, values in loss_by_name.items():
            self.trainer.storage.put_scalar(f"val_{name}", float(np.mean(values)))

        print(f"\n[LossEvalHook] iter {self.trainer.iter}: validation_loss = {mean_loss:.4f}\n")
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=DatasetMapper(self.cfg, is_train=True),
        )
        hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, val_loader))
        return hooks


def evaluate(cfg) -> dict:
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("orth_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    val_loader = build_detection_test_loader(cfg, "orth_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def select_samples_per_class(
    dataset_dicts: list[dict],
    num_classes: int,
    class_names: list[str],
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
        class_name = class_names[cls_id]
        if not candidates:
            print(f"  class {cls_id} ({class_name}): 0 GT images, skipped")
        elif len(candidates) < limit:
            print(f"  class {cls_id} ({class_name}): only {len(candidates)} GT images, saved all")
        else:
            print(f"  class {cls_id} ({class_name}): saved {limit} of {len(candidates)} GT images")

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


def save_visualizations(cfg, limit: int) -> None:
    if limit <= 0:
        return

    predictor = DefaultPredictor(cfg)
    dataset_name = cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = list(metadata.thing_classes)
    samples_by_class = select_samples_per_class(
        dataset_dicts,
        len(class_names),
        class_names,
        limit,
        seed=42,
    )
    visualization_dir = Path(cfg.OUTPUT_DIR) / "visualizations"
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
                pred_panel = add_panel_title(
                    pred_vis,
                    f"Pred score>={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:g}",
                )
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


def main() -> None:
    setup_logger()
    args = parse_args()
    class_names = load_categories(args.train_json)
    cfg = build_cfg(args, num_classes=len(class_names))
    register_datasets(
        args.data_dir,
        args.train_json,
        args.test_json,
        keep_negative_ratio=cfg.DATALOADER.KEEP_NEGATIVE_RATIO,
        seed=args.seed,
    )

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
