#!/usr/bin/env python3
"""Unified MaskDINO training/evaluation for caries segmentation and orth detection."""

from __future__ import annotations

import argparse
import json
import os
import random
import types
from pathlib import Path

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from datasets_coco.datasets_to_coco import CATEGORIES_INFO as CARIES_CATEGORIES_INFO
from maskdino import add_maskdino_config
from maskdino_train import Trainer
from maskdino.utils import box_ops


def add_task_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--task",
        choices=["caries", "orth"],
        default="caries",
        help="caries: single_tooth instance segmentation; orth: orth_test bbox detection.",
    )
    parser.add_argument("--data-dir", default=None, type=Path)
    parser.add_argument("--train-json", default=None, type=Path)
    parser.add_argument("--test-json", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--keep-negative-ratio", default=None, type=float)
    parser.add_argument("--repeat-threshold", default=None, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def apply_default_paths(args) -> None:
    if args.task == "caries":
        if not args.config_file:
            args.config_file = "configs/default_maskdino_caries_config.yaml"
        args.data_dir = args.data_dir or Path(".")
        args.train_json = args.train_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_train.json")
        args.test_json = args.test_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_test.json")
        args.output_dir = args.output_dir or Path("output/maskdino_caries")
    else:
        if not args.config_file:
            args.config_file = "configs/default_maskdino_orth_config.yaml"
        args.data_dir = args.data_dir or Path(".datasets/intraoral_anno/orth_test/orth_test")
        args.train_json = args.train_json or Path(".datasets/intraoral_anno/orth_test/orth_detection_train.json")
        args.test_json = args.test_json or Path(".datasets/intraoral_anno/orth_test/orth_detection_test.json")
        args.output_dir = args.output_dir or Path("output/maskdino_orth")


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
    task: str,
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
            if task == "caries":
                obj["segmentation"] = ann["segmentation"]
            record["annotations"].append(obj)

        dataset_dicts.append(record)

    split_name = "train" if is_train else "val"
    print(
        f"[{task}_{split_name}] loaded {len(dataset_dicts)} images from {json_path}; "
        f"kept {kept_negative_count}/{negative_count} empty negative samples"
    )
    return dataset_dicts


def register_task_datasets(args, class_names: list[str]) -> None:
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
                task=args.task,
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


class UnifiedTrainer(Trainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if getattr(cfg, "ORTH_BBOX_ONLY", False):
            configure_bbox_only_maskdino(model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if getattr(cfg, "ORTH_BBOX_ONLY", False):
            return COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=output_folder)
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def prepare_bbox_only_targets(self, targets, images):
    new_targets = []
    for targets_per_image in targets:
        height, width = targets_per_image.image_size
        image_size_xyxy = torch.as_tensor(
            [width, height, width, height],
            dtype=torch.float,
            device=self.device,
        )
        new_targets.append(
            {
                "labels": targets_per_image.gt_classes,
                "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)
                / image_size_xyxy,
            }
        )
    return new_targets


def matcher_bbox_only_forward(self, outputs, targets, cost=("cls", "box")):
    return self.memory_efficient_forward(outputs, targets, cost=list(cost))


def configure_bbox_only_maskdino(model) -> None:
    model.prepare_targets = types.MethodType(prepare_bbox_only_targets, model)
    model.prepare_targets_detr = types.MethodType(prepare_bbox_only_targets, model)

    criterion = model.criterion
    criterion.losses = ["labels", "boxes"]
    criterion.dn_losses = ["labels", "boxes"]
    criterion.weight_dict = {
        key: value
        for key, value in criterion.weight_dict.items()
        if "loss_mask" not in key and "loss_dice" not in key
    }
    criterion.matcher.cost_mask = 0.0
    criterion.matcher.cost_dice = 0.0
    criterion.matcher.forward = types.MethodType(matcher_bbox_only_forward, criterion.matcher)


def setup(args):
    apply_default_paths(args)
    class_names = load_categories(args.train_json, args.task)

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.set_new_allowed(True)
    cfg.ORTH_BBOX_ONLY = args.task == "orth"

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = (f"{args.task}_train",)
    cfg.DATASETS.TEST = (f"{args.task}_val",)
    cfg.OUTPUT_DIR = str(args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    if args.repeat_threshold is not None:
        cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold
    elif cfg.DATALOADER.REPEAT_THRESHOLD <= 0:
        cfg.DATALOADER.REPEAT_THRESHOLD = 0.05

    if args.keep_negative_ratio is None:
        args.keep_negative_ratio = 0.1 if args.task == "caries" else 0.2

    num_classes = len(class_names)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if hasattr(cfg.MODEL, "MaskDINO"):
        cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes

    register_task_datasets(args, class_names)

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)

    if args.eval_only:
        model = UnifiedTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        res = UnifiedTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = UnifiedTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = add_task_args(default_argument_parser())
    args = parser.parse_args()
    port = random.randint(1000, 20000)
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
