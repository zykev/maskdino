#!/usr/bin/env python3
"""Train/evaluate Mask R-CNN style models for caries segmentation or orth detection."""

from __future__ import annotations

import argparse
import logging
import os
import random
import types
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import detectron2.engine.hooks as d2_hooks
from detectron2 import model_zoo
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.events import JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
import torch.nn.functional as F

from datasets_coco.class_balance import add_class_balance_config, compute_effective_class_weights
from datasets_coco.orth_augmentations import (
    add_orth_augmentation_config,
    build_orth_augmentations,
)
from task_paths import add_input_dir_arg, resolve_task_paths
from unify_common import (
    ConciseMetricPrinter,
    LossEvalHook,
    TqdmHook,
    WandbWriter,
    add_panel_title,
    configure_distributed_solver,
    count_visible_gpus,
    draw_gt_panel,
    draw_pred_panel,
    load_categories,
    make_checkpointer_pointer_free,
    register_datasets,
    safe_path_name,
    select_samples_per_class,
    validate_distributed_launch,
)

REFERENCE_GLOBAL_BATCH = {"caries": 4, "orth": 2}

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Mask R-CNN runner for caries or orth tasks.")
    parser.add_argument(
        "--task",
        choices=["caries", "orth"],
        default="caries",
        help="caries: single_tooth instance segmentation; orth: orthodontic object detection.",
    )
    parser.add_argument("--config_file", default=None, type=Path)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--dist_url", default=None)
    add_input_dir_arg(parser)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--wandb_name", default=None, help="Override WANDB.NAME from the config.")
    parser.add_argument("--weights", default="", help="Optional checkpoint for resume/eval.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_iter", default=None, type=int)
    parser.add_argument("--eval_period", default=None, type=int)
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        default=None,
        type=int,
        help="Per-GPU batch size. Overrides DISTRIBUTED.IMS_PER_GPU.",
    )
    parser.add_argument("--base_lr", default=None, type=float)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--score_thresh", default=None, type=float)
    parser.add_argument("--keep_negative_ratio", default=None, type=float)
    parser.add_argument("--repeat_threshold", default=None, type=float)
    parser.add_argument("--vis_samples", default=8, type=int, help="Maximum visualizations per GT class.")
    parser.add_argument("--vis_score_thresh", default=0.5, type=float)
    parser.add_argument(
        "--save_raw_predictions",
        action="store_true",
        help="Save COCOEvaluator raw predictions under the inference directory.",
    )
    parser.add_argument("--log_period", default=100, type=int, help="Iteration interval for concise console logs.")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable the training progress bar.")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    config_file = getattr(args, "config_file", None)
    resolve_task_paths(args)
    if args.task == "caries":
        args.config_file = config_file or Path("configs/default_maskrcnn_caries_config.yaml")
    else:
        args.config_file = config_file or Path("configs/default_maskrcnn_orth_config.yaml")
    return args


def add_distributed_config(cfg) -> None:
    cfg.DISTRIBUTED = CN()
    cfg.DISTRIBUTED.NUM_GPUS = 1
    cfg.DISTRIBUTED.IMS_PER_GPU = 1


def _build_base_cfg():
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_class_balance_config(cfg)
    add_orth_augmentation_config(cfg)
    add_distributed_config(cfg)
    cfg.DATALOADER.KEEP_NEGATIVE_RATIO = 0.5
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    return cfg


def load_launch_settings(args: argparse.Namespace) -> tuple[int, int]:
    default_paths(args)
    cfg = _build_base_cfg()
    if args.config_file:
        cfg.merge_from_file(str(args.config_file))

    visible_gpu_count = count_visible_gpus()
    num_gpus = (
        visible_gpu_count
        if visible_gpu_count is not None
        else int(cfg.DISTRIBUTED.NUM_GPUS)
    )
    ims_per_gpu = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg.DISTRIBUTED.IMS_PER_GPU)
    )
    if num_gpus < 1:
        raise ValueError(f"DISTRIBUTED.NUM_GPUS must be positive, got {num_gpus}.")
    if ims_per_gpu < 1:
        raise ValueError(
            f"DISTRIBUTED.IMS_PER_GPU must be positive, got {ims_per_gpu}."
        )
    return num_gpus, ims_per_gpu


def configure_maskrcnn_class_balance(model, cfg) -> None:
    class_weights = list(getattr(cfg.ORTH_CLASS_BALANCE, "CLASS_WEIGHTS", []))
    if not class_weights:
        return

    if hasattr(model, "module"):
        model = model.module
    if not hasattr(model, "roi_heads") or not hasattr(model.roi_heads, "box_predictor"):
        return

    predictor = model.roi_heads.box_predictor
    original_losses = predictor.losses
    loss_type = str(cfg.ORTH_CLASS_BALANCE.LOSS_TYPE).lower()
    focal_gamma = float(cfg.ORTH_CLASS_BALANCE.FOCAL_GAMMA)
    background_weight = float(cfg.ORTH_CLASS_BALANCE.BACKGROUND_WEIGHT)

    def balanced_losses(self, predictions, proposals):
        losses = original_losses(predictions, proposals)
        scores = predictions[0]
        gt_classes = [
            proposal.gt_classes
            for proposal in proposals
            if len(proposal) and proposal.has("gt_classes")
        ]
        if gt_classes:
            gt_classes = torch.cat(gt_classes, dim=0)
        else:
            gt_classes = torch.empty(0, dtype=torch.int64, device=scores.device)

        valid = (gt_classes >= 0) & (gt_classes < scores.shape[1])
        if not torch.any(valid):
            losses["loss_cls"] = scores.sum() * 0.0
            return losses

        scores = scores[valid]
        gt_classes = gt_classes[valid]
        weight_vector = torch.ones(scores.shape[1], dtype=scores.dtype, device=scores.device)
        positive_count = min(len(class_weights), scores.shape[1] - 1)
        weight_vector[:positive_count] = torch.as_tensor(
            class_weights[:positive_count],
            dtype=scores.dtype,
            device=scores.device,
        )
        if scores.shape[1] > positive_count:
            weight_vector[-1] = background_weight

        sample_weights = weight_vector[gt_classes]
        cls_loss = F.cross_entropy(scores, gt_classes, reduction="none")
        if loss_type == "focal":
            probabilities = F.softmax(scores, dim=1)
            p_t = probabilities.gather(1, gt_classes[:, None]).squeeze(1)
            cls_loss = cls_loss * torch.pow(1.0 - p_t, focal_gamma)
        elif loss_type != "weighted_ce":
            raise ValueError(f"Unsupported ORTH_CLASS_BALANCE.LOSS_TYPE: {loss_type}")

        losses["loss_cls"] = (cls_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        return losses

    predictor.losses = types.MethodType(balanced_losses, predictor)
    print(f"Enabled orth Mask R-CNN class-balanced ROI {loss_type} loss")


def build_cfg(args: argparse.Namespace, num_classes: int):
    cfg = _build_base_cfg()
    if args.config_file:
        cfg.merge_from_file(str(args.config_file))
    cfg.DISTRIBUTED.NUM_GPUS = int(args.num_gpus)
    if args.batch_size is not None:
        cfg.DISTRIBUTED.IMS_PER_GPU = int(args.batch_size)

    cfg.DATASETS.TRAIN = (f"{args.task}_train",)
    cfg.DATASETS.TEST = (f"{args.task}_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = args.task == "caries"

    num_workers = getattr(args, "num_workers", None)
    repeat_threshold = getattr(args, "repeat_threshold", None)
    keep_negative_ratio = getattr(args, "keep_negative_ratio", None)
    score_thresh = getattr(args, "score_thresh", None)
    base_lr = getattr(args, "base_lr", None)
    max_iter = getattr(args, "max_iter", None)
    eval_period = getattr(args, "eval_period", None)
    weights = getattr(args, "weights", "")
    eval_only = getattr(args, "eval_only", False)

    if num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    if repeat_threshold is not None:
        cfg.DATALOADER.REPEAT_THRESHOLD = repeat_threshold
    if keep_negative_ratio is not None:
        cfg.DATALOADER.KEEP_NEGATIVE_RATIO = keep_negative_ratio
    else:
        args.keep_negative_ratio = cfg.DATALOADER.KEEP_NEGATIVE_RATIO
    if score_thresh is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    if base_lr is not None:
        cfg.SOLVER.BASE_LR = base_lr
    if max_iter is not None:
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
        cfg.SOLVER.WARMUP_ITERS = min(1000, max(1, max_iter // 20))
    if eval_period is not None:
        cfg.TEST.EVAL_PERIOD = eval_period
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = str(args.output_dir)
    if getattr(args, "wandb_name", None) is not None:
        cfg.WANDB.NAME = args.wandb_name
    cfg.TRAIN_LOG_PERIOD = getattr(args, "log_period", 100)
    cfg.TRAIN_TQDM = not getattr(args, "no_tqdm", False)
    cfg.SAVE_RAW_PREDICTIONS = getattr(args, "save_raw_predictions", False)
    if args.task == "orth" and cfg.ORTH_CLASS_BALANCE.ENABLED:
        cfg.ORTH_CLASS_BALANCE.CLASS_WEIGHTS = compute_effective_class_weights(
            args.train_json,
            num_classes,
            beta=float(cfg.ORTH_CLASS_BALANCE.BETA),
            clip_min=float(cfg.ORTH_CLASS_BALANCE.CLIP_MIN),
            clip_max=float(cfg.ORTH_CLASS_BALANCE.CLIP_MAX),
        )
    configure_distributed_solver(cfg, args, reference_global_batch=REFERENCE_GLOBAL_BATCH[args.task])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if weights:
        cfg.MODEL.WEIGHTS = weights
    elif eval_only:
        cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    elif not cfg.MODEL.WEIGHTS:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return cfg


def save_resolved_config(cfg) -> Path:
    config_path = Path(cfg.OUTPUT_DIR) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(cfg.dump(), encoding="utf-8")
    print(f"Saved resolved config to {config_path}")
    return config_path


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        # PeriodicCheckpointer keeps this same object, so retrofitting its
        # class here disables the pointer file for both periodic and final saves.
        make_checkpointer_pointer_free(self.checkpointer)
        if getattr(cfg, "ORTH_CLASS_BALANCE", None) and cfg.ORTH_CLASS_BALANCE.ENABLED:
            configure_maskrcnn_class_balance(self.model, cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None and cfg.SAVE_RAW_PREDICTIONS:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if not cfg.ORTH_AUGMENTATION.ENABLED:
            return super().build_train_loader(cfg)
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=build_orth_augmentations(cfg, is_train=True),
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        log_period = max(1, int(getattr(self.cfg, "TRAIN_LOG_PERIOD", 100)))
        for hook in hooks_list:
            if isinstance(hook, d2_hooks.PeriodicWriter) and hasattr(hook, "_period"):
                hook._period = log_period
        if self.cfg.ORTH_AUGMENTATION.ENABLED:
            val_mapper = DatasetMapper(
                self.cfg,
                is_train=True,
                augmentations=build_orth_augmentations(self.cfg, is_train=False),
            )
        else:
            val_mapper = DatasetMapper(self.cfg, is_train=True)
        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=val_mapper,
        )
        # Insert right after the AP EvalHook so collective ordering is identical on every rank.
        eval_idx = next(i for i, hook in enumerate(hooks_list) if isinstance(hook, d2_hooks.EvalHook))
        hooks_list.insert(eval_idx + 1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, val_loader))
        if getattr(self.cfg, "TRAIN_TQDM", True):
            hooks_list.insert(-1, TqdmHook(enabled=True))
        return hooks_list

    def build_writers(self):
        if not comm.is_main_process():
            return []
        metric_names = ("total_loss", "loss_cls", "loss_box_reg", "loss_mask", "val/total_loss")
        writers = [
            ConciseMetricPrinter(self.max_iter, metric_names),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]
        if getattr(self.cfg, "LOGGER", "tensorboard") == "wandb":
            writers.append(WandbWriter(self.cfg))
        else:
            writers.append(TensorboardXWriter(self.cfg.OUTPUT_DIR))
        return writers


def evaluate(cfg) -> dict:
    predictor = DefaultPredictor(cfg)
    dataset_name = cfg.DATASETS.TEST[0]
    evaluator_output_dir = (
        os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.SAVE_RAW_PREDICTIONS
        else None
    )
    evaluator = COCOEvaluator(dataset_name, output_dir=evaluator_output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results


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


def main(args: argparse.Namespace) -> None:
    setup_logger()
    args = default_paths(args)
    class_names = load_categories(args.train_json, args.task)
    cfg = build_cfg(args, len(class_names))
    register_datasets(args, class_names, include_masks=args.task == "caries")

    if comm.is_main_process():
        logging.getLogger("detectron2").info(
            "Distributed training config: world_size=%d, global_batch=%d, "
            "per_gpu_batch=%d, base_lr=%g, max_iter=%d",
            comm.get_world_size(),
            cfg.SOLVER.IMS_PER_BATCH,
            cfg.DISTRIBUTED.IMS_PER_GPU,
            cfg.SOLVER.BASE_LR,
            cfg.SOLVER.MAX_ITER,
        )
        save_resolved_config(cfg)

    if args.eval_only:
        evaluate(cfg)
        comm.synchronize()
        if comm.is_main_process():
            save_visualizations(
                cfg, args.vis_samples, args.seed, args.vis_score_thresh
            )
        return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    evaluate(cfg)
    comm.synchronize()
    if comm.is_main_process():
        save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    args.num_gpus, args.batch_size = load_launch_settings(args)
    if args.dist_url is None:
        # Fill in a local default before validating so a missing --dist_url on a
        # multi-machine launch is caught below instead of silently binding to
        # this machine only.
        args.dist_url = f"tcp://127.0.0.1:{random.randint(1000, 20000)}"
    validate_distributed_launch(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
