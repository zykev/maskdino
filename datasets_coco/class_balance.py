"""Shared class-balance utilities for orth detection training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from detectron2.config import CfgNode as CN


def add_class_balance_config(cfg) -> None:
    """Orthodontic class-balance options shared by MaskDINO and Mask R-CNN.

    LOSS_TYPE and BACKGROUND_WEIGHT only affect the Mask R-CNN ROI
    classification loss (see maskrcnn_unify.configure_maskrcnn_class_balance).
    MaskDINO always applies its own sigmoid focal loss driven solely by
    CLASS_WEIGHTS/FOCAL_GAMMA and ignores both fields.
    """
    cfg.ORTH_CLASS_BALANCE = CN()
    cfg.ORTH_CLASS_BALANCE.ENABLED = False
    cfg.ORTH_CLASS_BALANCE.BETA = 0.99
    cfg.ORTH_CLASS_BALANCE.CLIP_MIN = 0.25
    cfg.ORTH_CLASS_BALANCE.CLIP_MAX = 2.0
    cfg.ORTH_CLASS_BALANCE.FOCAL_GAMMA = 2.0
    cfg.ORTH_CLASS_BALANCE.CLASS_WEIGHTS = []
    cfg.ORTH_CLASS_BALANCE.LOSS_TYPE = "weighted_ce"
    cfg.ORTH_CLASS_BALANCE.BACKGROUND_WEIGHT = 1.0


def _clip_preserving_mean(
    weights: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Clip positive weights while keeping their mean equal to one."""
    if not 0.0 < clip_min <= 1.0 <= clip_max:
        raise ValueError(
            "Class-weight clipping must satisfy 0 < CLIP_MIN <= 1 <= CLIP_MAX"
        )

    target_sum = float(weights.size)
    result = np.clip(weights, clip_min, clip_max)
    tolerance = 1e-10

    for _ in range(100):
        difference = target_sum - float(result.sum())
        if abs(difference) <= tolerance:
            break

        if difference > 0:
            adjustable = result < clip_max - tolerance
        else:
            adjustable = result > clip_min + tolerance
        if not np.any(adjustable):
            break

        fixed_sum = float(result[~adjustable].sum())
        adjustable_sum = float(result[adjustable].sum())
        target_adjustable_sum = target_sum - fixed_sum
        if adjustable_sum <= 0.0 or target_adjustable_sum <= 0.0:
            break

        result[adjustable] *= target_adjustable_sum / adjustable_sum
        result = np.clip(result, clip_min, clip_max)

    return result


def effective_number_weights_from_counts(
    counts: np.ndarray,
    *,
    beta: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Compute mean-one effective-number weights over visible classes only."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("counts must be a one-dimensional array")
    if not 0.0 < beta < 1.0:
        raise ValueError("BETA must be between 0 and 1")

    weights = np.ones_like(counts, dtype=np.float64)
    visible = counts > 0
    if not np.any(visible):
        return weights

    visible_counts = counts[visible]
    visible_weights = (1.0 - beta) / (
        1.0 - np.power(beta, visible_counts)
    )
    visible_weights *= visible_weights.size / visible_weights.sum()
    visible_weights = _clip_preserving_mean(
        visible_weights,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    weights[visible] = visible_weights
    return weights


def compute_effective_class_weights(
    json_path: Path,
    num_classes: int,
    *,
    beta: float,
    clip_min: float,
    clip_max: float,
) -> list[float]:
    """Read a COCO train JSON and return one foreground weight per class."""
    with json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    counts = np.zeros(num_classes, dtype=np.float64)
    for annotation in coco_data.get("annotations", []):
        class_index = int(annotation["category_id"]) - 1
        if 0 <= class_index < num_classes:
            counts[class_index] += 1.0

    weights = effective_number_weights_from_counts(
        counts,
        beta=beta,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    category_names = {
        int(category["id"]) - 1: str(category["name"])
        for category in coco_data.get("categories", [])
        if 1 <= int(category["id"]) <= num_classes
    }
    count_log = {
        category_names.get(index, str(index)): int(count)
        for index, count in enumerate(counts.tolist())
    }
    weight_log = {
        category_names.get(index, str(index)): round(float(weight), 4)
        for index, weight in enumerate(weights.tolist())
    }
    unseen = [
        category_names.get(index, str(index))
        for index, count in enumerate(counts.tolist())
        if count == 0
    ]

    print("Orth class-balance box counts:", count_log)
    print("Orth effective-number weights:", weight_log)
    if unseen:
        print(
            "Warning: orth classes with zero training boxes are excluded from "
            f"weight normalization and kept at weight 1.0: {unseen}"
        )
    return [float(weight) for weight in weights]
