#!/usr/bin/env python3
"""Unified MaskDINO training/evaluation for caries segmentation and orth detection."""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import os
import types
import warnings
import weakref
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
from detectron2.config import CfgNode as CN, get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import transforms as T
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
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger

from datasets_coco.class_balance import compute_effective_class_weights
from datasets_coco.orth_augmentations import build_orth_augmentations
from task_paths import add_input_dir_arg, resolve_task_paths
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    DetrDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    add_maskdino_config,
)
from maskdino.utils import box_ops
from unify_common import (
    ConciseMetricPrinter,
    LossEvalHook,
    PointerFreeDetectionCheckpointer,
    TqdmHook,
    WandbWriter,
    add_panel_title,
    configure_distributed_solver,
    count_visible_gpus,
    draw_gt_panel,
    draw_pred_panel,
    load_categories,
    load_coco_dicts,
    register_datasets,
    safe_path_name,
    select_samples_per_class,
    validate_distributed_launch,
)

# Existing solver schedules and learning rates were tuned with global batch size 2.
REFERENCE_GLOBAL_BATCH = 2

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
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
    category=UserWarning,
)


def add_task_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config_file", dest="config_file", default=argparse.SUPPRESS)
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
    add_input_dir_arg(parser)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--wandb_name", default=None, help="Override WANDB.NAME from the config.")
    parser.add_argument("--keep_negative_ratio", default=None, type=float)
    parser.add_argument("--repeat_threshold", default=None, type=float)
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        default=None,
        type=int,
        help="Per-GPU batch size. Overrides DISTRIBUTED.IMS_PER_GPU.",
    )
    parser.add_argument("--vis_samples", default=8, type=int, help="Maximum visualizations per GT class.")
    parser.add_argument("--vis_score_thresh", default=0.5, type=float)
    parser.add_argument("--log_period", default=100, type=int, help="Iteration interval for concise console logs.")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable the training progress bar.")
    parser.add_argument("--seed", default=42, type=int)
    return parser


def apply_default_paths(args) -> None:
    config_file = getattr(args, "config_file", None)
    resolve_task_paths(args)
    if args.task == "caries":
        if not config_file:
            args.config_file = "configs/default_maskdino_caries_config.yaml"
    else:
        if not config_file:
            args.config_file = "configs/default_maskdino_orth_resnet_config.yaml"


def add_distributed_config(cfg) -> None:
    cfg.DISTRIBUTED = CN()
    cfg.DISTRIBUTED.NUM_GPUS = 2
    cfg.DISTRIBUTED.IMS_PER_GPU = 1


def _build_base_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_distributed_config(cfg)
    cfg.set_new_allowed(True)
    return cfg


def load_launch_settings(args) -> tuple[int, int]:
    apply_default_paths(args)
    cfg = _build_base_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    visible_gpu_count = count_visible_gpus()
    num_gpus = (
        visible_gpu_count
        if visible_gpu_count is not None
        else int(cfg.DISTRIBUTED.NUM_GPUS)
    )
    if num_gpus < 1:
        raise ValueError(f"DISTRIBUTED.NUM_GPUS must be positive, got {num_gpus}.")
    ims_per_gpu = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg.DISTRIBUTED.IMS_PER_GPU)
    )
    if ims_per_gpu < 1:
        raise ValueError(
            f"DISTRIBUTED.IMS_PER_GPU must be positive, got {ims_per_gpu}."
        )
    return num_gpus, ims_per_gpu


class MaskDINOLossEvalHook(LossEvalHook):
    """LossEvalHook that also disables the criterion's cross-rank mask-count
    sync while computing validation loss (only relevant for MaskDINO)."""

    def __init__(self, eval_period: int, model, data_loader) -> None:
        # Never run on the final iteration: it is inserted right after the AP
        # EvalHook so both hooks issue their collective ops in the same order
        # on every rank, and the AP EvalHook already covers the final iter.
        super().__init__(eval_period, model, data_loader, run_on_final_iter=False)
        self._sync_num_masks = None

    def _begin_eval(self, model) -> None:
        criterion = getattr(model, "criterion", None)
        self._sync_num_masks = getattr(criterion, "sync_num_masks", None)
        if criterion is not None and self._sync_num_masks is not None:
            criterion.sync_num_masks = False

    def _end_eval(self, model) -> None:
        criterion = getattr(model, "criterion", None)
        if criterion is not None and self._sync_num_masks is not None:
            criterion.sync_num_masks = self._sync_num_masks


class Trainer(DefaultTrainer):
    """Trainer adapted to MaskDINO without depending on task-specific entry files."""

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        if comm.is_main_process():
            logger.info(
                "Distributed training config: world_size=%d, global_batch=%d, "
                "per_gpu_batch=%d, base_lr=%g, max_iter=%d",
                comm.get_world_size(),
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
                cfg.SOLVER.BASE_LR,
                cfg.SOLVER.MAX_ITER,
            )

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
        self.checkpointer = PointerFreeDetectionCheckpointer(
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
        if self.cfg.TEST.EVAL_PERIOD > 0:
            # Insert right after the AP EvalHook so collective ordering is identical on every rank.
            eval_idx = next(i for i, hook in enumerate(hooks_list) if isinstance(hook, hooks.EvalHook))
            hooks_list.insert(eval_idx + 1, MaskDINOLossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, self.build_loss_loader(self.cfg)))
        if getattr(self.cfg, "TRAIN_TQDM", True):
            hooks_list.insert(-1, TqdmHook(enabled=True))
        return hooks_list

    def build_writers(self):
        if not comm.is_main_process():
            return []
        metric_names = ("total_loss", "loss_ce", "loss_bbox", "loss_giou", "loss_ce_dn")
        writers = [
            ConciseMetricPrinter(self.max_iter, metric_names),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]
        if getattr(self.cfg, "LOGGER", "tensorboard") == "wandb":
            writers.append(WandbWriter(self.cfg))
        else:
            writers.append(TensorboardXWriter(self.cfg.OUTPUT_DIR))
        return writers

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if getattr(cfg, "ORTH_BBOX_ONLY", False):
            return COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=output_folder)
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if getattr(cfg, "ORTH_BBOX_ONLY", False):
            configure_bbox_only_maskdino(model)
        return model

    @staticmethod
    def build_mapper(cfg, is_train, augmentations=None):
        name = cfg.INPUT.DATASET_MAPPER_NAME
        if name == "coco_instance_lsj":
            return COCOInstanceNewBaselineDatasetMapper(cfg, is_train)
        if name == "coco_instance_detr":
            return DetrDatasetMapper(cfg, is_train, augmentations=augmentations)
        if name == "coco_panoptic_lsj":
            return COCOPanopticNewBaselineDatasetMapper(cfg, is_train)
        if name == "mask_former_semantic":
            return MaskFormerSemanticDatasetMapper(cfg, is_train)
        return None

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=cls.build_mapper(cfg, True))

    @classmethod
    def build_loss_loader(cls, cfg):
        # Keep is_train=True so the mapper yields annotations needed to compute losses.
        augmentations = None
        if cfg.ORTH_AUGMENTATION.ENABLED:
            augmentations = build_orth_augmentations(cfg, is_train=False)
        return build_detection_test_loader(
            cfg,
            cfg.DATASETS.TEST[0],
            mapper=cls.build_mapper(cfg, True, augmentations=augmentations),
        )

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
    model.focus_on_box = True
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
            inputs = [{"image": image, "height": height, "width": width}]
            return self.model(inputs)[0]


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

    cfg = _build_base_cfg()
    cfg.ORTH_BBOX_ONLY = args.task == "orth"

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DISTRIBUTED.NUM_GPUS = int(args.num_gpus)
    if args.batch_size is not None:
        cfg.DISTRIBUTED.IMS_PER_GPU = int(args.batch_size)

    cfg.DATASETS.TRAIN = (f"{args.task}_train",)
    cfg.DATASETS.TEST = (f"{args.task}_val",)
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = str(args.output_dir)
    if args.wandb_name is not None:
        cfg.WANDB.NAME = args.wandb_name
    cfg.TRAIN_LOG_PERIOD = args.log_period
    cfg.TRAIN_TQDM = not args.no_tqdm
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Negative samples are handled via KEEP_NEGATIVE_RATIO in load_coco_dicts, not
    # by dropping empty-annotation images, and RepeatFactorTrainingSampler is what
    # the class-balance tuning here assumes. Both are fixed regardless of the yaml.
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    if args.repeat_threshold is not None:
        cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold

    if args.keep_negative_ratio is None:
        args.keep_negative_ratio = float(cfg.DATALOADER.KEEP_NEGATIVE_RATIO)

    num_classes = len(class_names)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    if args.task == "orth" and cfg.ORTH_CLASS_BALANCE.ENABLED:
        cfg.ORTH_CLASS_BALANCE.CLASS_WEIGHTS = compute_effective_class_weights(
            args.train_json,
            num_classes,
            beta=float(cfg.ORTH_CLASS_BALANCE.BETA),
            clip_min=float(cfg.ORTH_CLASS_BALANCE.CLIP_MIN),
            clip_max=float(cfg.ORTH_CLASS_BALANCE.CLIP_MAX),
        )

    # instance_inference() does scores.flatten(0, 1).topk(TEST.DETECTIONS_PER_IMAGE);
    # the flattened candidate pool has NUM_OBJECT_QUERIES * num_classes entries.
    max_candidates = int(cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES) * num_classes
    if max_candidates < int(cfg.TEST.DETECTIONS_PER_IMAGE):
        raise ValueError(
            f"MODEL.MaskDINO.NUM_OBJECT_QUERIES ({cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES}) * "
            f"num_classes ({num_classes}) = {max_candidates}, which is less than "
            f"TEST.DETECTIONS_PER_IMAGE ({cfg.TEST.DETECTIONS_PER_IMAGE}); "
            "instance_inference()'s topk() would fail at eval time."
        )

    register_datasets(args, class_names, include_masks=args.task == "caries")
    configure_distributed_solver(cfg, args, reference_global_batch=REFERENCE_GLOBAL_BATCH)

    cfg.freeze()
    if comm.is_main_process():
        default_setup(cfg, args)
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    else:
        seed = getattr(cfg, "SEED", -1)
        seed_all_rng(None if seed < 0 else seed + comm.get_rank())
        setup_logger(output=None, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        PointerFreeDetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
            save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)
        return res

    trainer = Trainer(cfg)
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
    args.num_gpus, args.batch_size = load_launch_settings(args)
    validate_distributed_launch(args)
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
