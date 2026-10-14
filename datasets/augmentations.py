"""Structure-preserving augmentations for the intraoral downstream tasks.

A single ``AUGMENTATION`` config node holds one policy per task:

* ``AUGMENTATION.CARIES`` -- fine-grained instance segmentation. Conservative,
  annotation-safe geometric + photometric jitter that protects lesion texture;
  cropping, blur and JPEG compression stay off.
* ``AUGMENTATION.ORTH`` -- bbox-only detection on panoramic X-rays.

Both policies share the same transform *order* and the same config *schema*
(see ``_add_task_node``); only the ranges and probabilities differ. The task a
run uses is resolved from the dataset/task (``resolve_augmentation``), so the
entry points never hard-code one.

These policies implement ``Docs/downstream_input_augmentation_protocol.md``
sections 5.1 (caries) and 5.2 (orth). Three mirrors must stay in sync:

1. ``configs/{maskrcnn,maskdino}/caries/_base.yaml`` -- ``AUGMENTATION.CARIES``
2. ``configs/{maskrcnn,maskdino}/orth/_base.yaml``   -- ``AUGMENTATION.ORTH``
3. ``representative_orth_augmentation_transforms`` below (the montage tool).
"""

from __future__ import annotations

import cv2
import numpy as np
from detectron2.config import CfgNode as CN
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform

#: Tasks that own an ``AUGMENTATION.<TASK>`` policy node.
AUGMENTATION_TASKS = ("caries", "orth")


class GammaTransform(Transform):
    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = float(gamma)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        image_float = image.astype(np.float32) / 255.0
        corrected = np.power(np.clip(image_float, 0.0, 1.0), self.gamma)
        return np.clip(corrected * 255.0, 0.0, 255.0).astype(image.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomGamma(T.Augmentation):
    def __init__(self, gamma_min: float, gamma_max: float) -> None:
        super().__init__()
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    def get_transform(self, image: np.ndarray) -> Transform:
        gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        return GammaTransform(gamma)


class GaussianBlurTransform(Transform):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class JpegCompressionTransform(Transform):
    def __init__(self, quality: int) -> None:
        super().__init__()
        self.quality = int(quality)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return image
        if image.ndim == 3 and decoded.ndim == 2:
            decoded = np.repeat(decoded[:, :, None], image.shape[2], axis=2)
        return decoded.astype(image.dtype, copy=False)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomImageQuality(T.Augmentation):
    """Randomly apply blur or JPEG compression, never both."""

    def __init__(
        self,
        blur_sigma_min: float,
        blur_sigma_max: float,
        jpeg_quality_min: int,
        jpeg_quality_max: int,
    ) -> None:
        super().__init__()
        self.blur_sigma_min = float(blur_sigma_min)
        self.blur_sigma_max = float(blur_sigma_max)
        self.jpeg_quality_min = int(jpeg_quality_min)
        self.jpeg_quality_max = int(jpeg_quality_max)

    def get_transform(self, image: np.ndarray) -> Transform:
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(self.blur_sigma_min, self.blur_sigma_max)
            return GaussianBlurTransform(sigma)
        quality = np.random.randint(
            self.jpeg_quality_min,
            self.jpeg_quality_max + 1,
        )
        return JpegCompressionTransform(quality)


# --------------------------------------------------------------------------- #
# Config schema
# --------------------------------------------------------------------------- #
#: Every policy node gets these keys so a YAML may override only the ones it
#: cares about. Defaults mirror the orth policy; ``_TASK_OVERRIDES`` adjusts the
#: caries policy to the more conservative protocol 5.1.
_DEFAULT_NODE_VALUES = {
    "ENABLED": False,
    "HORIZONTAL_FLIP_PROB": 0.5,
    "ROTATION_PROB": 0.3,
    "ROTATION_ANGLE": 7.0,
    "BRIGHTNESS_PROB": 0.2,
    "BRIGHTNESS_MIN": 0.85,
    "BRIGHTNESS_MAX": 1.15,
    "CONTRAST_PROB": 0.2,
    "CONTRAST_MIN": 0.90,
    "CONTRAST_MAX": 1.10,
    "SATURATION_PROB": 0.2,
    "SATURATION_MIN": 0.90,
    "SATURATION_MAX": 1.10,
    "GAMMA_PROB": 0.2,
    "GAMMA_MIN": 0.90,
    "GAMMA_MAX": 1.10,
    "QUALITY_PROB": 0.2,
    "BLUR_SIGMA_MIN": 0.10,
    "BLUR_SIGMA_MAX": 0.80,
    "JPEG_QUALITY_MIN": 75,
    "JPEG_QUALITY_MAX": 100,
}

#: Per-task departures from ``_DEFAULT_NODE_VALUES``. Caries follows protocol
#: 5.1: a small rotation range, mild brightness/contrast/gamma, saturation off,
#: and no blur / JPEG (``QUALITY_PROB = 0``).
_TASK_OVERRIDES = {
    "CARIES": {
        "ROTATION_PROB": 0.2,
        "ROTATION_ANGLE": 5.0,
        "BRIGHTNESS_MIN": 0.90,
        "BRIGHTNESS_MAX": 1.10,
        "CONTRAST_MIN": 0.90,
        "CONTRAST_MAX": 1.10,
        "SATURATION_PROB": 0.0,
        "GAMMA_MIN": 0.90,
        "GAMMA_MAX": 1.10,
        "QUALITY_PROB": 0.0,
    },
    "ORTH": {},
}


def _add_task_node(parent: CN, task: str) -> CN:
    node = CN()
    values = dict(_DEFAULT_NODE_VALUES)
    values.update(_TASK_OVERRIDES.get(task, {}))
    for key, value in values.items():
        setattr(node, key, value)
    parent[task] = node
    return node


def add_augmentation_config(cfg) -> None:
    """Register the per-task augmentation policies under ``cfg.AUGMENTATION``.

    Both ``CARIES`` and ``ORTH`` always exist, so a ``_base.yaml`` only needs to
    author the node for its own task; the other keeps its defaults.
    """
    cfg.AUGMENTATION = CN()
    for task in AUGMENTATION_TASKS:
        _add_task_node(cfg.AUGMENTATION, task.upper())


def _infer_task(cfg) -> str:
    """Resolve the current task from the entry-point flag, dataset names, mask flag.

    Priority:

    1. ``cfg.ORTH_BBOX_ONLY`` — set by every entry point
       (``maskdino_unify``/``maskrcnn_unify``/``*_pred``) as
       ``args.task == "orth"``. This is the authoritative signal because it is
       derived from the CLI ``--task`` rather than from whatever YAML default
       ``DATASETS.TRAIN`` happens to carry at build time.
    2. The ``DATASETS.TRAIN`` name prefix (``{task}_train``).
    3. ``MODEL.MASK_ON`` as a last resort for callers that only know the task
       shape (e.g. the config validator).
    """
    orth_bbox_only = getattr(cfg, "ORTH_BBOX_ONLY", None)
    if orth_bbox_only is not None:
        return "orth" if bool(orth_bbox_only) else "caries"
    for name in getattr(cfg.DATASETS, "TRAIN", ()) or ():
        lowered = str(name).lower()
        for task in AUGMENTATION_TASKS:
            if lowered.startswith(task):
                return task
    return "caries" if bool(getattr(cfg.MODEL, "MASK_ON", False)) else "orth"


def resolve_augmentation(cfg, task: str | None = None):
    """Return the ``AUGMENTATION`` policy node for ``task`` (or ``None``)."""
    node = getattr(cfg, "AUGMENTATION", None)
    if node is None:
        return None
    task = (task or _infer_task(cfg)).lower()
    return getattr(node, task.upper(), None)


def augmentation_enabled(cfg, task: str | None = None) -> bool:
    node = resolve_augmentation(cfg, task)
    return bool(node is not None and node.ENABLED)


def build_augmentations(cfg, is_train: bool, task: str | None = None) -> list[T.Augmentation]:
    """Build the augmentation list for the resolved task.

    In eval mode this collapses to the deterministic test-time resize, matching
    the plain Detectron2 pipeline. In train mode every transform whose
    probability is zero is skipped, so a policy can disable a stage simply by
    setting its probability to ``0`` (e.g. caries disables quality/blur/JPEG).
    """
    if not is_train:
        return [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                "choice",
            )
        ]

    node = resolve_augmentation(cfg, task)
    if node is None or not node.ENABLED:
        raise ValueError(
            "build_augmentations() called for training but the resolved "
            "AUGMENTATION policy is missing or disabled."
        )

    augmentations: list[T.Augmentation] = []

    if float(node.HORIZONTAL_FLIP_PROB) > 0:
        augmentations.append(
            T.RandomFlip(
                prob=float(node.HORIZONTAL_FLIP_PROB),
                horizontal=True,
                vertical=False,
            )
        )

    augmentations.append(
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    )

    if float(node.ROTATION_PROB) > 0 and float(node.ROTATION_ANGLE) > 0:
        augmentations.append(
            T.RandomApply(
                T.RandomRotation(
                    angle=[-float(node.ROTATION_ANGLE), float(node.ROTATION_ANGLE)],
                    expand=False,
                    sample_style="range",
                ),
                prob=float(node.ROTATION_PROB),
            )
        )

    if float(node.BRIGHTNESS_PROB) > 0:
        augmentations.append(
            T.RandomApply(
                T.RandomBrightness(
                    float(node.BRIGHTNESS_MIN),
                    float(node.BRIGHTNESS_MAX),
                ),
                prob=float(node.BRIGHTNESS_PROB),
            )
        )

    if float(node.CONTRAST_PROB) > 0:
        augmentations.append(
            T.RandomApply(
                T.RandomContrast(
                    float(node.CONTRAST_MIN),
                    float(node.CONTRAST_MAX),
                ),
                prob=float(node.CONTRAST_PROB),
            )
        )

    if float(node.SATURATION_PROB) > 0:
        augmentations.append(
            T.RandomApply(
                T.RandomSaturation(
                    float(node.SATURATION_MIN),
                    float(node.SATURATION_MAX),
                ),
                prob=float(node.SATURATION_PROB),
            )
        )

    if float(node.GAMMA_PROB) > 0:
        augmentations.append(
            T.RandomApply(
                RandomGamma(
                    float(node.GAMMA_MIN),
                    float(node.GAMMA_MAX),
                ),
                prob=float(node.GAMMA_PROB),
            )
        )

    if float(node.QUALITY_PROB) > 0:
        augmentations.append(
            T.RandomApply(
                RandomImageQuality(
                    blur_sigma_min=float(node.BLUR_SIGMA_MIN),
                    blur_sigma_max=float(node.BLUR_SIGMA_MAX),
                    jpeg_quality_min=int(node.JPEG_QUALITY_MIN),
                    jpeg_quality_max=int(node.JPEG_QUALITY_MAX),
                ),
                prob=float(node.QUALITY_PROB),
            )
        )

    return augmentations


def representative_orth_augmentation_transforms() -> list[tuple[str, object]]:
    """Deterministic transforms used by the orth montage script.

    Values mirror the strongest end of each ``AUGMENTATION.ORTH`` range in
    ``configs/maskrcnn/orth/_base.yaml`` and ``configs/maskdino/orth/_base.yaml``;
    keep the three in sync.
    """
    return [
        ("Horizontal flip", T.RandomFlip(prob=1.0, horizontal=True, vertical=False)),
        (
            "Rotation +7 deg",
            T.RandomRotation(
                angle=[7.0, 7.0],
                expand=False,
                sample_style="range",
            ),
        ),
        ("Resize short=640", T.ResizeShortestEdge([640], 1333, "choice")),
        ("Resize short=800", T.ResizeShortestEdge([800], 1333, "choice")),
        ("Resize short=1024", T.ResizeShortestEdge([1024], 1333, "choice")),
        ("Brightness 1.15", T.RandomBrightness(1.15, 1.15)),
        ("Contrast 1.10", T.RandomContrast(1.10, 1.10)),
        ("Saturation 1.10", T.RandomSaturation(1.10, 1.10)),
        ("Gamma 1.10", RandomGamma(1.10, 1.10)),
        ("Gaussian blur 0.80", GaussianBlurTransform(0.80)),
        ("JPEG quality 75", JpegCompressionTransform(75)),
    ]
