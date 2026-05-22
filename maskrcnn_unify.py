#!/usr/bin/env python3
"""Train/evaluate Mask R-CNN style models for caries segmentation or orth detection."""

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
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as comm

from datasets_coco.datasets_to_coco import CATEGORIES_INFO as CARIES_CATEGORIES_INFO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Mask R-CNN runner for caries or orth tasks.")
    parser.add_argument(
        "--task",
        choices=["caries", "orth"],
        default="caries",
        help="caries: single_tooth instance segmentation; orth: orthodontic object detection.",
    )
    parser.add_argument("--config_file", default=None, type=Path)
    parser.add_argument("--data_dir", default=None, type=Path)
    parser.add_argument("--train_json", default=None, type=Path)
    parser.add_argument("--test_json", default=None, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--weights", default="", help="Optional checkpoint for resume/eval.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_iter", default=None, type=int)
    parser.add_argument("--eval_period", default=None, type=int)
    parser.add_argument("--ims_per_batch", default=None, type=int)
    parser.add_argument("--base_lr", default=None, type=float)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--score_thresh", default=None, type=float)
    parser.add_argument("--keep_negative_ratio", default=None, type=float)
    parser.add_argument("--repeat_threshold", default=None, type=float)
    parser.add_argument("--vis_samples", default=8, type=int, help="Maximum visualizations per GT class.")
    parser.add_argument("--vis_score_thresh", default=0.5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.task == "caries":
        args.config_file = args.config_file or Path("configs/default_maskrcnn_caries_config.yaml")
        args.data_dir = args.data_dir or Path(".")
        args.train_json = args.train_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_train.json")
        args.test_json = args.test_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_test.json")
        args.output_dir = args.output_dir or Path("output/maskrcnn_caries")
    else:
        args.config_file = args.config_file or Path("configs/default_maskrcnn_orth_config.yaml")
        args.data_dir = args.data_dir or Path(".datasets/intraoral_anno/orth_test/orth_test")
        args.train_json = args.train_json or Path(".datasets/intraoral_anno/orth_test/orth_detection_train.json")
        args.test_json = args.test_json or Path(".datasets/intraoral_anno/orth_test/orth_detection_test.json")
        args.output_dir = args.output_dir or Path("output/maskrcnn_orth")
    return args


def load_categories(json_path: Path, task: str) -> list[str]:
    if task == "caries":
        return [item["name"] for item in sorted(CARIES_CATEGORIES_INFO, key=lambda item: item["id"])]

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [category["name"] for category in sorted(data["categories"], key=lambda item: item["id"])]


def resolve_image_path(data_dir: Path, file_name: str) -> str:
    path = Path(file_name)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(data_dir / file_name)


def load_coco_dicts(
    data_dir: Path,
    json_path: Path,
    *,
    include_masks: bool,
    is_train: bool,
    keep_negative_ratio: float,
    seed: int,
) -> list[dict]:
    if not 0.0 <= keep_negative_ratio <= 1.0:
        raise ValueError("--keep_negative_ratio must be between 0.0 and 1.0")

    with json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {image["id"]: image for image in coco_data["images"]}
    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for ann in coco_data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    rng = random.Random(seed)
    dataset_dicts = []
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
            "file_name": resolve_image_path(data_dir, image_info["file_name"]),
            "image_id": image_id,
            "height": image_info["height"],
            "width": image_info["width"],
            "annotations": [],
        }
        for ann in anns:
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": ann["category_id"] - 1,
                "iscrowd": ann.get("iscrowd", 0),
            }
            if include_masks and ann.get("segmentation"):
                obj["segmentation"] = ann["segmentation"]
            record["annotations"].append(obj)

        dataset_dicts.append(record)

    split_name = "train" if is_train else "val"
    print(
        f"[{split_name}] loaded {len(dataset_dicts)} images from {json_path}; "
        f"kept {kept_negative_count}/{negative_count} empty negative samples"
    )
    return dataset_dicts


def register_datasets(args: argparse.Namespace, class_names: list[str]) -> None:
    include_masks = args.task == "caries"
    for split, json_path in {"train": args.train_json, "val": args.test_json}.items():
        dataset_name = f"{args.task}_{split}"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)

        is_train = split == "train"
        DatasetCatalog.register(
            dataset_name,
            lambda json_path=json_path, is_train=is_train: load_coco_dicts(
                args.data_dir,
                json_path,
                include_masks=include_masks,
                is_train=is_train,
                keep_negative_ratio=args.keep_negative_ratio,
                seed=args.seed,
            ),
        )
        MetadataCatalog.get(dataset_name).set(
            thing_classes=class_names,
            evaluator_type="coco",
            json_file=str(json_path),
            image_root=str(args.data_dir),
            thing_dataset_id_to_contiguous_id={
                category_id: category_id - 1 for category_id in range(1, len(class_names) + 1)
            },
        )


def build_cfg(args: argparse.Namespace, num_classes: int):
    cfg = get_cfg()
    cfg.DATALOADER.KEEP_NEGATIVE_RATIO = 0.5
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    if args.config_file:
        cfg.merge_from_file(str(args.config_file))

    cfg.DATASETS.TRAIN = (f"{args.task}_train",)
    cfg.DATASETS.TEST = (f"{args.task}_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = args.task == "caries"

    if args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    if args.repeat_threshold is not None:
        cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold
    if args.keep_negative_ratio is not None:
        cfg.DATALOADER.KEEP_NEGATIVE_RATIO = args.keep_negative_ratio
    else:
        args.keep_negative_ratio = cfg.DATALOADER.KEEP_NEGATIVE_RATIO
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
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
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
        return sum(metrics_dict.values()), metrics_dict

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
        self.trainer.storage.put_scalar("validation_loss", float(np.mean(losses)) if losses else 0.0)
        for name, values in loss_by_name.items():
            self.trainer.storage.put_scalar(f"val_{name}", float(np.mean(values)))
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
    dataset_name = cfg.DATASETS.TEST[0]
    evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


def select_samples_per_class(dataset_dicts: list[dict], class_names: list[str], limit: int, seed: int):
    rng = random.Random(seed)
    records_by_class: dict[int, list[dict]] = defaultdict(list)
    for record in dataset_dicts:
        for cls_id in {ann["category_id"] for ann in record.get("annotations", [])}:
            if 0 <= cls_id < len(class_names):
                records_by_class[cls_id].append(record)

    samples_by_class = {}
    for cls_id, class_name in enumerate(class_names):
        records = records_by_class.get(cls_id, [])
        rng.shuffle(records)
        samples_by_class[cls_id] = records[: min(limit, len(records))]
        print(f"Visualization class {cls_id} ({class_name}): {len(samples_by_class[cls_id])}/{len(records)}")
    return samples_by_class


def safe_path_name(name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe_name.strip("._") or "class"


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    header_height = 36
    titled = np.full((image_rgb.shape[0] + header_height, image_rgb.shape[1], 3), 255, dtype=np.uint8)
    titled[header_height:, :, :] = image_rgb
    cv2.putText(titled, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return titled


def strip_instance_masks(record: dict) -> dict:
    stripped_record = {key: value for key, value in record.items() if key != "annotations"}
    stripped_record["annotations"] = []
    for ann in record.get("annotations", []):
        stripped_record["annotations"].append(
            {key: value for key, value in ann.items() if key != "segmentation"}
        )
    return stripped_record


def draw_gt_panel(image_rgb: np.ndarray, metadata, record: dict, draw_masks: bool) -> np.ndarray:
    draw_record = record if draw_masks else strip_instance_masks(record)
    return Visualizer(image_rgb, metadata=metadata, scale=0.8).draw_dataset_dict(draw_record).get_image()


def draw_pred_panel(image_rgb: np.ndarray, metadata, instances, draw_masks: bool) -> np.ndarray:
    pred_instances = instances.to("cpu")
    if not draw_masks and pred_instances.has("pred_masks"):
        pred_instances.remove("pred_masks")
    return Visualizer(image_rgb, metadata=metadata, scale=0.8).draw_instance_predictions(pred_instances).get_image()


def save_visualizations(cfg, limit: int, seed: int, score_thresh: float) -> None:
    if limit <= 0:
        return

    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    dataset_name = cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = list(metadata.thing_classes)
    draw_masks = bool(cfg.MODEL.MASK_ON)
    samples_by_class = select_samples_per_class(dataset_dicts, class_names, limit, seed)

    output_dir = Path(cfg.OUTPUT_DIR) / "visualizations" / "by_class"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_by_image_id: dict[int, np.ndarray] = {}

    for cls_id, samples in samples_by_class.items():
        class_dir = output_dir / f"{cls_id:02d}_{safe_path_name(class_names[cls_id])}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(samples):
            image_id = record["image_id"]
            comparison = rendered_by_image_id.get(image_id)
            if comparison is None:
                image = cv2.imread(record["file_name"])
                if image is None:
                    print(f"Could not read image: {record['file_name']}")
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

    print(f"Saved visualizations to {output_dir}")


def main() -> None:
    setup_logger()
    args = default_paths(parse_args())
    class_names = load_categories(args.train_json, args.task)
    cfg = build_cfg(args, len(class_names))
    register_datasets(args, class_names)

    if args.eval_only:
        evaluate(cfg)
        save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)
        return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    evaluate(cfg)
    save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
