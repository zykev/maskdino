"""Structure-preserving augmentations for orthodontic detection images."""

from __future__ import annotations

import cv2
import numpy as np
from detectron2.config import CfgNode as CN
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform


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


def add_orth_augmentation_config(cfg) -> None:
    cfg.ORTH_AUGMENTATION = CN()
    cfg.ORTH_AUGMENTATION.ENABLED = False
    cfg.ORTH_AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.5
    cfg.ORTH_AUGMENTATION.ROTATION_PROB = 0.3
    cfg.ORTH_AUGMENTATION.ROTATION_ANGLE = 7.0
    # Each color transform is sampled independently. At 0.2 each, the chance
    # of applying at least one color transform is about 49%.
    cfg.ORTH_AUGMENTATION.COLOR_PROB = 0.2
    cfg.ORTH_AUGMENTATION.BRIGHTNESS_MIN = 0.85
    cfg.ORTH_AUGMENTATION.BRIGHTNESS_MAX = 1.15
    cfg.ORTH_AUGMENTATION.CONTRAST_MIN = 0.90
    cfg.ORTH_AUGMENTATION.CONTRAST_MAX = 1.10
    cfg.ORTH_AUGMENTATION.SATURATION_MIN = 0.90
    cfg.ORTH_AUGMENTATION.SATURATION_MAX = 1.10
    cfg.ORTH_AUGMENTATION.GAMMA_PROB = 0.2
    cfg.ORTH_AUGMENTATION.GAMMA_MIN = 0.90
    cfg.ORTH_AUGMENTATION.GAMMA_MAX = 1.10
    cfg.ORTH_AUGMENTATION.QUALITY_PROB = 0.2
    cfg.ORTH_AUGMENTATION.BLUR_SIGMA_MIN = 0.10
    cfg.ORTH_AUGMENTATION.BLUR_SIGMA_MAX = 0.80
    cfg.ORTH_AUGMENTATION.JPEG_QUALITY_MIN = 75
    cfg.ORTH_AUGMENTATION.JPEG_QUALITY_MAX = 100


def build_orth_augmentations(cfg, is_train: bool) -> list[T.Augmentation]:
    if not is_train:
        return [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                "choice",
            )
        ]

    aug = cfg.ORTH_AUGMENTATION
    augmentations: list[T.Augmentation] = [
        T.RandomFlip(
            prob=float(aug.HORIZONTAL_FLIP_PROB),
            horizontal=True,
            vertical=False,
        ),
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        ),
        T.RandomApply(
            T.RandomRotation(
                angle=[-float(aug.ROTATION_ANGLE), float(aug.ROTATION_ANGLE)],
                expand=False,
                sample_style="range",
            ),
            prob=float(aug.ROTATION_PROB),
        ),
        T.RandomApply(
            T.RandomBrightness(
                float(aug.BRIGHTNESS_MIN),
                float(aug.BRIGHTNESS_MAX),
            ),
            prob=float(aug.COLOR_PROB),
        ),
        T.RandomApply(
            T.RandomContrast(
                float(aug.CONTRAST_MIN),
                float(aug.CONTRAST_MAX),
            ),
            prob=float(aug.COLOR_PROB),
        ),
        T.RandomApply(
            T.RandomSaturation(
                float(aug.SATURATION_MIN),
                float(aug.SATURATION_MAX),
            ),
            prob=float(aug.COLOR_PROB),
        ),
        T.RandomApply(
            RandomGamma(
                float(aug.GAMMA_MIN),
                float(aug.GAMMA_MAX),
            ),
            prob=float(aug.GAMMA_PROB),
        ),
        T.RandomApply(
            RandomImageQuality(
                blur_sigma_min=float(aug.BLUR_SIGMA_MIN),
                blur_sigma_max=float(aug.BLUR_SIGMA_MAX),
                jpeg_quality_min=int(aug.JPEG_QUALITY_MIN),
                jpeg_quality_max=int(aug.JPEG_QUALITY_MAX),
            ),
            prob=float(aug.QUALITY_PROB),
        ),
    ]
    return augmentations


def representative_augmentation_transforms() -> list[tuple[str, object]]:
    """Deterministic transforms used by the montage script.

    Values mirror the strongest end of each ORTH_AUGMENTATION range in
    default_maskdino_orth_base_config.yaml; keep the two in sync.
    """
    return [
        ("Horizontal flip", T.RandomFlip(prob=1.0, horizontal=True, vertical=False)),
        (
            "Rotation +12 deg",
            T.RandomRotation(
                angle=[12.0, 12.0],
                expand=False,
                sample_style="range",
            ),
        ),
        ("Resize short=640", T.ResizeShortestEdge([640], 1333, "choice")),
        ("Resize short=800", T.ResizeShortestEdge([800], 1333, "choice")),
        ("Resize short=1024", T.ResizeShortestEdge([1024], 1333, "choice")),
        ("Brightness 1.28", T.RandomBrightness(1.28, 1.28)),
        ("Contrast 1.28", T.RandomContrast(1.28, 1.28)),
        ("Saturation 1.28", T.RandomSaturation(1.28, 1.28)),
        ("Gamma 1.20", RandomGamma(1.20, 1.20)),
        ("Gaussian blur 1.75", GaussianBlurTransform(1.75)),
        ("JPEG quality 55", JpegCompressionTransform(55)),
    ]
