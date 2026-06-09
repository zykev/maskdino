#!/usr/bin/env python3
"""Unified MaskDINO training/evaluation for caries segmentation and orth detection."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import random
import types
import warnings
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Set

import cv2
import detectron2.utils.comm as comm
import numpy as np
import torch
# PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the
# argparse.Namespace stored in our trusted Swin-MAE pretrain checkpoints.
torch.serialization.add_safe_globals([argparse.Namespace])
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    create_ddp_model,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, SemSegEvaluator, verify_results
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.structures import BoxMode
from detectron2.utils.events import EventWriter, JSONWriter, TensorboardXWriter, get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from datasets_coco.datasets_to_coco import CATEGORIES_INFO as CARIES_CATEGORIES_INFO
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    DetrDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
)
from maskdino.utils import box_ops

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)


def add_task_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config_file", dest="config_file", default=argparse.SUPPRESS)
    parser.add_argument("--num_gpus", dest="num_gpus", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_machines", dest="num_machines", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--machine_rank", dest="machine_rank", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dist_url", dest="dist_url", default=argparse.SUPPRESS)
    parser.add_argument("--eval_only", dest="eval_only", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--task",
        choices=["caries", "orth"],
        default="caries",
        help="caries: single_tooth instance segmentation; orth: orth_test bbox detection.",
    )
    parser.add_argument("--data_dir", default=None, type=Path)
    parser.add_argument("--train_json", default=None, type=Path)
    parser.add_argument("--test_json", default=None, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--keep_negative_ratio", default=None, type=float)
    parser.add_argument("--repeat_threshold", default=None, type=float)
    parser.add_argument("--vis_samples", default=8, type=int, help="Maximum visualizations per GT class.")
    parser.add_argument("--vis_score_thresh", default=0.5, type=float)
    parser.add_argument("--log_period", default=100, type=int, help="Iteration interval for concise console logs.")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable the training progress bar.")
    parser.add_argument("--seed", default=42, type=int)
    return parser


def apply_default_paths(args) -> None:
    config_file = getattr(args, "config_file", None)
    if args.task == "caries":
        if not config_file:
            args.config_file = "configs/default_maskdino_caries_config.yaml"
        args.data_dir = args.data_dir or Path(".datasets/intraoral_anno/single_ch_0225/single_tooth")
        args.train_json = args.train_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_train.json")
        args.test_json = args.test_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_test.json")
        args.output_dir = args.output_dir or Path("output/maskdino_caries")
    else:
        if not config_file:
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


class ConciseMetricPrinter(EventWriter):
    """Print a compact training summary instead of every auxiliary loss."""

    def __init__(self, max_iter: int, window_size: int = 20) -> None:
        self.max_iter = max_iter
        self.window_size = window_size
        self.logger = logging.getLogger("detectron2")

    def write(self) -> None:
        storage = get_event_storage()
        iteration = storage.iter
        latest = storage.latest_with_smoothing_hint(self.window_size)

        def latest_value(name: str):
            value = latest.get(name)
            if value is None:
                return None
            return value[0] if isinstance(value, tuple) else value

        pieces = [f"iter: {iteration}/{self.max_iter}"]
        for name in ("total_loss", "loss_ce", "loss_bbox", "loss_giou", "loss_ce_dn"):
            value = latest_value(name)
            if value is not None:
                pieces.append(f"{name}: {value:.4g}")

        lr = latest_value("lr")
        if lr is not None:
            pieces.append(f"lr: {lr:.3g}")
        if torch.cuda.is_available():
            pieces.append(f"max_mem: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}M")

        self.logger.info("  ".join(pieces))


class TqdmHook(hooks.HookBase):
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and tqdm is not None and comm.is_main_process()
        self.progress = None

    def before_train(self) -> None:
        if not self.enabled:
            return
        total = max(0, self.trainer.max_iter - self.trainer.start_iter)
        self.progress = tqdm(total=total, dynamic_ncols=True, desc="training")

    def after_step(self) -> None:
        if self.progress is None:
            return
        self.progress.update(1)
        storage = get_event_storage()
        latest = storage.latest_with_smoothing_hint(20)

        def latest_value(name: str):
            value = latest.get(name)
            if value is None:
                return None
            return value[0] if isinstance(value, tuple) else value

        postfix = {}
        total_loss = latest_value("total_loss")
        lr = latest_value("lr")
        if total_loss is not None:
            postfix["loss"] = f"{total_loss:.3g}"
        if lr is not None:
            postfix["lr"] = f"{lr:.2g}"
        if postfix:
            self.progress.set_postfix(postfix, refresh=False)

    def after_train(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None


class Trainer(DefaultTrainer):
    """Trainer adapted to MaskDINO without depending on task-specific entry files."""

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(
            model,
            broadcast_buffers=False,
            find_unused_parameters=getattr(cfg, "ORTH_BBOX_ONLY", False),
        )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {
            "trainer": weakref.proxy(self),
        }
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        hooks_list = super().build_hooks()
        log_period = max(1, int(getattr(self.cfg, "TRAIN_LOG_PERIOD", 100)))
        for hook in hooks_list:
            if isinstance(hook, hooks.PeriodicWriter) and hasattr(hook, "_period"):
                hook._period = log_period
        if getattr(self.cfg, "TRAIN_TQDM", True):
            hooks_list.insert(-1, TqdmHook(enabled=True))
        return hooks_list

    def build_writers(self):
        if not comm.is_main_process():
            return []
        return [
            ConciseMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco" or evaluator_type == "":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        if len(evaluator_list) == 0:
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg, mapper=None)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {
            "lr": cfg.SOLVER.BASE_LR,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        }

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})


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


class MaskDINOPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        if getattr(self.cfg, "ORTH_BBOX_ONLY", False):
            configure_bbox_only_maskdino(self.model)
        self.model.eval()
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        self.input_format = self.cfg.INPUT.FORMAT

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            return self.model(inputs)[0]


def select_samples_per_class(dataset_dicts: list[dict], class_names: list[str], limit: int, seed: int):
    rng = random.Random(seed)
    records_by_class: dict[int, list[dict]] = {}
    for record in dataset_dicts:
        for cls_id in {ann["category_id"] for ann in record.get("annotations", [])}:
            if 0 <= cls_id < len(class_names):
                records_by_class.setdefault(cls_id, []).append(record)

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
    stripped_record["annotations"] = [
        {key: value for key, value in ann.items() if key != "segmentation"}
        for ann in record.get("annotations", [])
    ]
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
    if limit <= 0 or not comm.is_main_process():
        return

    cfg = cfg.clone()
    cfg.defrost()
    if hasattr(cfg.MODEL, "MaskDINO"):
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
        cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = score_thresh
        if getattr(cfg, "ORTH_BBOX_ONLY", False):
            cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = True
    cfg.freeze()

    predictor = MaskDINOPredictor(cfg)
    dataset_name = cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = list(metadata.thing_classes)
    draw_masks = not getattr(cfg, "ORTH_BBOX_ONLY", False)
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
                outputs = predictor(image)
                instances = outputs["instances"].to("cpu")
                if instances.has("scores"):
                    instances = instances[instances.scores >= score_thresh]

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

    print(f"Saved visualizations to {output_dir}")


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
    cfg.TRAIN_LOG_PERIOD = args.log_period
    cfg.TRAIN_TQDM = not args.no_tqdm
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
            save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)
        return res

    trainer = UnifiedTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    train_result = trainer.train()
    comm.synchronize()

    if comm.is_main_process():
        vis_cfg = cfg.clone()
        vis_cfg.defrost()
        vis_cfg.MODEL.WEIGHTS = str(Path(vis_cfg.OUTPUT_DIR) / "model_final.pth")
        vis_cfg.freeze()
        save_visualizations(vis_cfg, args.vis_samples, args.seed, args.vis_score_thresh)
    return train_result


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
