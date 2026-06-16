#!/usr/bin/env python3
"""Train/evaluate Mask R-CNN style models for caries segmentation or orth detection."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import types
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import detectron2.engine.hooks as d2_hooks
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.events import EventWriter, JSONWriter, TensorboardXWriter, get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as comm

from datasets_coco.datasets_to_coco import CATEGORIES_INFO as CARIES_CATEGORIES_INFO

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)


def configure_pointer_free_checkpointer(checkpointer) -> None:
    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        return

    def latest_checkpoint(self) -> str:
        if not self.save_dir:
            return ""
        checkpoints = list(Path(self.save_dir).glob("model_*.pth"))
        if not checkpoints:
            return ""
        return str(max(checkpoints, key=lambda path: path.stat().st_mtime))

    def has_checkpoint(self) -> bool:
        return bool(latest_checkpoint(self))

    def get_checkpoint_file(self) -> str:
        return latest_checkpoint(self)

    checkpointer.tag_last_checkpoint = types.MethodType(tag_last_checkpoint, checkpointer)
    checkpointer.has_checkpoint = types.MethodType(has_checkpoint, checkpointer)
    checkpointer.get_checkpoint_file = types.MethodType(get_checkpoint_file, checkpointer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Mask R-CNN runner for caries or orth tasks.")
    parser.add_argument(
        "--task",
        choices=["caries", "orth"],
        default="caries",
        help="caries: single_tooth instance segmentation; orth: orthodontic object detection.",
    )
    parser.add_argument("--config_file", default=None, type=Path)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--dist_url", default=None)
    parser.add_argument("--data_dir", default=None, type=Path)
    parser.add_argument("--train_json", default=None, type=Path)
    parser.add_argument("--test_json", default=None, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--wandb_name", default=None, help="Override WANDB.NAME from the config.")
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
    if args.task == "caries":
        args.config_file = config_file or Path("configs/default_maskrcnn_caries_config.yaml")
        args.data_dir = args.data_dir or Path(".")
        args.train_json = args.train_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_train.json")
        args.test_json = args.test_json or Path(".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset_test.json")
        args.output_dir = args.output_dir or Path("output/maskrcnn_caries")
    else:
        args.config_file = config_file or Path("configs/default_maskrcnn_orth_config.yaml")
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
    cfg.set_new_allowed(True)
    cfg.DATALOADER.KEEP_NEGATIVE_RATIO = 0.5
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    if args.config_file:
        cfg.merge_from_file(str(args.config_file))

    cfg.DATASETS.TRAIN = (f"{args.task}_train",)
    cfg.DATASETS.TEST = (f"{args.task}_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = args.task == "caries"

    num_workers = getattr(args, "num_workers", None)
    repeat_threshold = getattr(args, "repeat_threshold", None)
    keep_negative_ratio = getattr(args, "keep_negative_ratio", None)
    score_thresh = getattr(args, "score_thresh", None)
    ims_per_batch = getattr(args, "ims_per_batch", None)
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
    if ims_per_batch is not None:
        cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
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


class LossEvalHook(HookBase):
    """Periodically compute the validation loss to monitor overfitting."""

    def __init__(self, eval_period: int, model, data_loader):
        self._period = eval_period
        self._model = model
        self._data_loader = data_loader

    def _do_loss_eval(self):
        total: dict[str, float] = {}
        count = 0
        self._model.train()
        with torch.no_grad():
            for inputs in self._data_loader:
                loss_dict = self._model(inputs)
                for key, value in loss_dict.items():
                    total[key] = total.get(key, 0.0) + value.item()
                count += 1
        # Aggregate per-worker sums with a single collective to avoid deadlocks on uneven shards.
        all_total = comm.all_gather(total)
        count_sum = sum(comm.all_gather(count))
        if count_sum and comm.is_main_process():
            keys = set().union(*[part.keys() for part in all_total])
            means = {f"val/{key}": sum(part.get(key, 0.0) for part in all_total) / count_sum for key in keys}
            means["val/total_loss"] = sum(means.values())
            self.trainer.storage.put_scalars(**means, smoothing_hint=False)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class WandbWriter(EventWriter):
    """Mirror EventStorage scalars (losses, AP, lr, ...) to Weights & Biases."""

    def __init__(self, cfg, window_size: int = 20) -> None:
        import wandb

        self._wandb = wandb
        self._window_size = window_size
        self._last_write = -1
        wandb.init(
            project=cfg.WANDB.PROJECT,
            entity=cfg.WANDB.ENTITY or None,
            name=cfg.WANDB.NAME or None,
            dir=cfg.OUTPUT_DIR,
        )

    def write(self) -> None:
        storage = get_event_storage()
        log_dict = {}
        new_last_write = self._last_write
        for key, (value, iteration) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iteration > self._last_write:
                log_dict[key] = value
                new_last_write = max(new_last_write, iteration)
        if log_dict:
            self._wandb.log(log_dict, step=new_last_write)
        self._last_write = new_last_write

    def close(self) -> None:
        self._wandb.finish()


class ConciseMetricPrinter(EventWriter):
    """Print a compact training summary while preserving full metrics on disk."""

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
        for name in ("total_loss", "loss_cls", "loss_box_reg", "loss_mask", "val/total_loss"):
            value = latest_value(name)
            if value is not None:
                pieces.append(f"{name}: {value:.4g}")

        lr = latest_value("lr")
        if lr is not None:
            pieces.append(f"lr: {lr:.3g}")
        if torch.cuda.is_available():
            pieces.append(f"max_mem: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}M")

        self.logger.info("  ".join(pieces))


class TqdmHook(HookBase):
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
    def __init__(self, cfg):
        super().__init__(cfg)
        # PeriodicCheckpointer keeps this same object, so patching it here
        # disables the pointer file for both periodic and final saves.
        configure_pointer_free_checkpointer(self.checkpointer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None and cfg.SAVE_RAW_PREDICTIONS:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        log_period = max(1, int(getattr(self.cfg, "TRAIN_LOG_PERIOD", 100)))
        for hook in hooks_list:
            if isinstance(hook, d2_hooks.PeriodicWriter) and hasattr(hook, "_period"):
                hook._period = log_period
        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=DatasetMapper(self.cfg, is_train=True),
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
        writers = [
            ConciseMetricPrinter(self.max_iter),
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


def main(args: argparse.Namespace) -> None:
    setup_logger()
    args = default_paths(args)
    class_names = load_categories(args.train_json, args.task)
    cfg = build_cfg(args, len(class_names))
    register_datasets(args, class_names)

    if args.eval_only:
        evaluate(cfg)
        save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)
        return

    save_resolved_config(cfg)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    evaluate(cfg)
    save_visualizations(cfg, args.vis_samples, args.seed, args.vis_score_thresh)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    if args.dist_url is None:
        args.dist_url = f"tcp://127.0.0.1:{random.randint(1000, 20000)}"
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
