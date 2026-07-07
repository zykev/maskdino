#!/usr/bin/env python3
"""Shared helpers for maskdino_unify.py and maskrcnn_unify.py.

Both scripts train/evaluate a detector on the same caries/orth COCO-style
datasets and share the same conventions for checkpoints, logging, and
qualitative visualization. This module holds the pieces that were
byte-for-byte (or near byte-for-byte) duplicated between the two entry
points so a fix in one place applies to both.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import hooks
from detectron2.structures import BoxMode
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.utils.visualizer import Visualizer

from datasets_coco.datasets_to_coco import CATEGORIES_INFO as CARIES_CATEGORIES_INFO

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# --------------------------------------------------------------------------
# Launch / distributed setup
# --------------------------------------------------------------------------

def count_visible_gpus() -> int | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return None
    devices = [
        device.strip()
        for device in visible_devices.split(",")
        if device.strip() and device.strip() != "-1"
    ]
    if not devices:
        raise ValueError("CUDA_VISIBLE_DEVICES does not expose any GPUs.")
    return len(devices)


def configure_distributed_solver(cfg, args, *, reference_global_batch: float) -> None:
    """Scale batch-dependent solver settings to the actual launched world size.

    `reference_global_batch` is the global batch size the shipped LR/schedule
    values were tuned for; callers pass a model-specific constant.
    """
    world_size = comm.get_world_size()
    num_gpus = int(cfg.DISTRIBUTED.NUM_GPUS)
    ims_per_gpu = int(cfg.DISTRIBUTED.IMS_PER_GPU)
    expected_world_size = num_gpus * int(args.num_machines)
    if world_size != expected_world_size:
        raise ValueError(
            f"Configured {num_gpus} GPUs per machine across {args.num_machines} "
            f"machines, but the launched world size is {world_size}."
        )

    global_batch = world_size * ims_per_gpu
    scale = global_batch / reference_global_batch

    def scale_iterations(value: int) -> int:
        return max(1, int(round(value / scale))) if value > 0 else value

    cfg.SOLVER.IMS_PER_BATCH = global_batch
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.SOLVER.BASE_LR *= scale
    cfg.SOLVER.MAX_ITER = scale_iterations(cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.WARMUP_ITERS = scale_iterations(cfg.SOLVER.WARMUP_ITERS)
    cfg.SOLVER.STEPS = tuple(scale_iterations(step) for step in cfg.SOLVER.STEPS)
    cfg.SOLVER.CHECKPOINT_PERIOD = scale_iterations(cfg.SOLVER.CHECKPOINT_PERIOD)
    cfg.TEST.EVAL_PERIOD = scale_iterations(cfg.TEST.EVAL_PERIOD)


def validate_distributed_launch(args) -> None:
    if args.num_machines <= 1:
        return
    dist_url = str(args.dist_url)
    local_urls = ("tcp://127.0.0.1", "tcp://localhost", "tcp://[::1]")
    if dist_url == "auto" or dist_url.startswith(local_urls):
        raise ValueError(
            "Multi-machine training requires all machines to use the same reachable "
            "rank-0 rendezvous URL. Pass e.g. --dist-url tcp://<rank0-host-or-ip>:29500."
        )


# --------------------------------------------------------------------------
# Dataset loading / registration
# --------------------------------------------------------------------------

def load_categories(json_path: Path, task: str) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    category_ids = [int(category["id"]) for category in categories]
    expected_ids = list(range(1, len(categories) + 1))
    if not categories or category_ids != expected_ids:
        raise ValueError(
            f"{json_path} must define contiguous COCO category IDs starting at 1; "
            f"got {category_ids}."
        )

    category_names = [str(category["name"]) for category in categories]
    if task == "caries":
        expected_names = [
            item["name"]
            for item in sorted(CARIES_CATEGORIES_INFO, key=lambda item: item["id"])
        ]
        if category_names != expected_names:
            raise ValueError(
                f"{json_path} caries categories do not match datasets_to_coco.py: "
                f"expected {expected_names}, got {category_names}."
            )
    return category_names


def remap_image_subdir(path: Path, image_subdir: str) -> Path:
    if image_subdir == "images":
        return path

    parts = list(path.parts)
    try:
        images_index = parts.index("images")
    except ValueError:
        return path

    parts[images_index] = image_subdir
    return Path(*parts)


def resolve_image_path(
    data_dir: Path,
    file_name: str,
    image_subdir: str = "images",
) -> str:
    path = remap_image_subdir(Path(file_name), image_subdir)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(data_dir / path)


def load_coco_dicts(
    data_dir: Path,
    json_path: Path,
    *,
    include_masks: bool,
    is_train: bool,
    keep_negative_ratio: float,
    seed: int,
    log_label: str,
    image_subdir: str = "images",
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
            "file_name": resolve_image_path(
                data_dir,
                image_info["file_name"],
                image_subdir,
            ),
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

    print(
        f"[{log_label}] loaded {len(dataset_dicts)} images from {json_path}; "
        f"image_subdir={image_subdir}; "
        f"kept {kept_negative_count}/{negative_count} empty negative samples"
    )
    return dataset_dicts


def register_datasets(args, class_names: list[str], *, include_masks: bool) -> None:
    image_subdir = getattr(args, "image_subdir", "images")
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
                log_label=dataset_name,
                image_subdir=image_subdir,
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


# --------------------------------------------------------------------------
# Checkpointing without a `last_checkpoint` pointer file
# --------------------------------------------------------------------------

def find_latest_checkpoint(save_dir: str) -> str:
    """Return the newest `model_*.pth` in `save_dir`, or "" if none exist."""
    if not save_dir:
        return ""
    checkpoints = list(Path(save_dir).glob("model_*.pth"))
    if not checkpoints:
        return ""
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


class PointerFreeDetectionCheckpointer(DetectionCheckpointer):
    """Resume from the newest checkpoint file without writing last_checkpoint."""

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        return

    def has_checkpoint(self) -> bool:
        return bool(find_latest_checkpoint(self.save_dir))

    def get_checkpoint_file(self) -> str:
        return find_latest_checkpoint(self.save_dir)


def make_checkpointer_pointer_free(checkpointer) -> None:
    """Patch an already-constructed DetectionCheckpointer in place.

    Used where the checkpointer is built internally by DefaultTrainer and
    can't be swapped for a PointerFreeDetectionCheckpointer instance directly.
    """
    checkpointer.__class__ = PointerFreeDetectionCheckpointer


# --------------------------------------------------------------------------
# Training hooks
# --------------------------------------------------------------------------

class LossEvalHook(hooks.HookBase):
    """Periodically compute the validation loss to monitor overfitting."""

    def __init__(self, eval_period: int, model, data_loader, *, run_on_final_iter: bool = True) -> None:
        self._period = eval_period
        self._model = model
        self._data_loader = data_loader
        self._run_on_final_iter = run_on_final_iter

    def _unwrap_model(self):
        return (
            self._model.module
            if isinstance(self._model, torch.nn.parallel.DistributedDataParallel)
            else self._model
        )

    def _begin_eval(self, model) -> None:
        """Hook point for subclasses to adjust model state before eval batches."""

    def _end_eval(self, model) -> None:
        """Hook point for subclasses to undo `_begin_eval`."""

    def _do_loss_eval(self) -> None:
        model = self._unwrap_model()
        was_training = model.training
        batch_norm_states = {
            module: module.training
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        }
        model.train()
        for module in batch_norm_states:
            module.eval()
        self._begin_eval(model)

        total: dict[str, float] = {}
        count = 0
        try:
            for inputs in self._data_loader:
                with torch.no_grad():
                    loss_dict = model(inputs)
                if not isinstance(loss_dict, dict):
                    raise TypeError(
                        "LossEvalHook expected the model to return a loss dict. "
                        f"Got {type(loss_dict).__name__} instead."
                    )
                for key, value in loss_dict.items():
                    total[key] = total.get(key, 0.0) + float(value.detach().item())
                count += 1
        finally:
            self._end_eval(model)
            model.train(was_training)
            for module, training in batch_norm_states.items():
                module.train(training)

        keys = sorted(total)
        if comm.get_world_size() > 1:
            all_keys = comm.all_gather(keys)
            keys = sorted(set().union(*all_keys))

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        stats = torch.tensor(
            [total.get(key, 0.0) for key in keys] + [float(count)],
            dtype=torch.float64,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        count_sum = float(stats[-1].item())
        if count_sum and comm.is_main_process():
            means = {
                f"val/{key}": float(stats[index].item() / count_sum)
                for index, key in enumerate(keys)
            }
            means["val/total_loss"] = sum(means.values())
            self.trainer.storage.put_scalars(**means, smoothing_hint=False)

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final:
            if self._run_on_final_iter:
                self._do_loss_eval()
            return
        if self._period > 0 and next_iter % self._period == 0:
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
    """Print a compact training summary instead of every auxiliary loss."""

    def __init__(self, max_iter: int, metric_names: tuple[str, ...], window_size: int = 20) -> None:
        self.max_iter = max_iter
        self.metric_names = metric_names
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
        for name in self.metric_names:
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


# --------------------------------------------------------------------------
# Qualitative visualization (GT vs. prediction panels)
# --------------------------------------------------------------------------

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
