#!/usr/bin/env python3
"""Build the COCO JSONs consumed by training from raw LabelMe annotations.

One entry point covers both tasks, and each task's pipeline has two steps:

    1. to-coco   raw <data_dir>   ->  <input_dir>/<coco>.json
    2. split     <coco>.json      ->  <input_dir>/<train>.json + <test>.json

Both steps are idempotent. Before running anything the script scans the
``.datasets`` tree and decides, per step, whether the outputs are missing or
older than their inputs; up-to-date steps are reported and skipped. Pass
``--overwrite`` to force a rebuild and ``--dry-run`` to only report::

    python datasets/prepare_coco.py --task caries
    python datasets/prepare_coco.py --task orth --input-dir .datasets/intraoral_anno/orth_0616
    python datasets/prepare_coco.py --task orth --overwrite

Both tasks assume the ``images/`` + ``anno/`` layout (see
``migrate_to_images_anno_layout.py`` for converting the old co-located one)::

    <data_dir>/<sample_id>/images/<any/depth>/<name>.<suffix>
    <data_dir>/<sample_id>/anno/<any/depth>/<name>.json

An image and its annotation are paired by their path relative to ``data_dir``
with the ``images`` / ``anno`` component removed and the file suffix dropped, so
``2207/images/D/41.png`` and ``2207/anno/D/41.json`` (or ``2207/anno/41.json``)
both key to ``(2207, "D", "41")``. Consequences:

* the annotation folder depth is free — any view nesting (or none) works,
* any image suffix in ``IMAGE_SUFFIXES`` is accepted, not only ``.png``/``.jpg``,
* images without an annotation are kept as negative samples; pass
  ``--drop-empty-images`` to drop them instead.

Task-specific rules:

* **caries** — shapes are emitted only when the LabelMe ``flags["1"]`` marker is
  true (unhealthy tooth), and only polygons of at least three points count.
  ``--allow-missing-images`` keeps records whose image file is absent (the
  ``file_name`` is mirrored from the annotation path); a record whose size
  cannot be determined falls back to ``0 x 0``.
* **orth** — ``rectangle`` and ``polygon`` shapes become boxes. The categories
  come from the built-in IOTN/DHC table; ``--label-mode supercategory``
  collapses the 32 codes into anomaly groups while ``code`` keeps them as-is,
  and ``--labels`` overrides both with an explicit order. Every annotation
  carries ``orth_code``/``severity_raw`` metadata. A record whose size cannot
  be determined is an error.

``file_name`` is stored relative to ``data_dir`` so the JSONs stay valid
regardless of the working directory; the trainer joins it back with
``data_dir`` (see ``utils/unify_common.py``).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

# ``utils/`` sits one level up; this keeps the module usable both as a plain
# script (``python datasets/prepare_coco.py``) and as a package module.
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.task_paths import resolve_task_paths  # noqa: E402


TASKS = ("caries", "orth")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

#: Intermediate COCO JSON produced by the to-coco step, per task.
COCO_JSON_NAME = {
    "caries": "caries_sample_dataset.json",
    "orth": "orth_detection_coco.json",
}

#: Suffix used when an annotation's image file is missing and no candidate in
#: the mirrored ``images/`` folder exists.
FALLBACK_IMAGE_SUFFIX = {"caries": ".png", "orth": ".jpg"}


# ---------------------------------------------------------------------------
# Caries categories
# ---------------------------------------------------------------------------

# Raw LabelMe label id -> COCO category id.
CARIES_CATEGORY_MAP = {
    "1": 1,   # caries
    "11": 2,  # white spot lesion
    "3": 3,   # filling without caries
    "4": 4,   # filling with caries
    "6": 5,   # fissure sealant
    "10": 6,  # non-caries disease (hard tissue)
    "12": 7,  # staining
    "7": 8,   # abnormal central cusp
    "8": 9,   # palatal radicular groove
}

CARIES_CATEGORIES_INFO = [
    {"id": 1, "name": "caries", "supercategory": "tooth_disease"},
    {"id": 2, "name": "white_spot_lesion", "supercategory": "tooth_disease"},
    {"id": 3, "name": "filling_no_caries", "supercategory": "treatment"},
    {"id": 4, "name": "filling_with_caries", "supercategory": "tooth_disease"},
    {"id": 5, "name": "fissure_sealant", "supercategory": "treatment"},
    {"id": 6, "name": "non_caries_hard_tissue", "supercategory": "tooth_disease"},
    {"id": 7, "name": "staining", "supercategory": "clinical_finding"},
    {"id": 8, "name": "abnormal_central_cusp", "supercategory": "anatomy_anomaly"},
    {"id": 9, "name": "palatal_radicular_groove", "supercategory": "anatomy_anomaly"},
]


# ---------------------------------------------------------------------------
# Orth categories: the fixed IOTN/DHC table
# ---------------------------------------------------------------------------

# LabelMe shape labels use the short code, e.g. "3a" or "4d".
ORTH_CATEGORIES_INFO = [
    {"id": 1, "name": "1", "supercategory": "minor_malocclusion",
     "description": "Extremely minor malocclusions including contact point displacement less than 1 mm"},
    {"id": 2, "name": "2a", "supercategory": "increased_overjet",
     "description": "Increased overjet greater than 3.5 mm but up to 6 mm with competent lips"},
    {"id": 3, "name": "3a", "supercategory": "increased_overjet",
     "description": "Increased overjet greater than 3.5 mm but up to 6 mm with incompetent lips"},
    {"id": 4, "name": "4a", "supercategory": "increased_overjet",
     "description": "Increased overjet greater than 6 mm but up to 9 mm"},
    {"id": 5, "name": "5a", "supercategory": "increased_overjet",
     "description": "Increased overjet greater than 9 mm"},
    {"id": 6, "name": "2b", "supercategory": "reverse_overjet",
     "description": "Reverse overjet greater than 0 mm but up to 1 mm"},
    {"id": 7, "name": "3b", "supercategory": "reverse_overjet",
     "description": "Reverse overjet greater than 1 mm but up to 3.5 mm"},
    {"id": 8, "name": "4b", "supercategory": "reverse_overjet",
     "description": "Reverse overjet greater than 3.5 mm with no masticatory or speech difficulties"},
    {"id": 9, "name": "4m", "supercategory": "reverse_overjet",
     "description": "Reverse overjet greater than 1 mm but less than 3.5 mm with recorded difficulties"},
    {"id": 10, "name": "5m", "supercategory": "reverse_overjet",
     "description": "Reverse overjet greater than 3.5 mm with reported masticatory and speech difficulties"},
    {"id": 11, "name": "2c", "supercategory": "crossbite",
     "description": "Crossbite with up to 1 mm discrepancy"},
    {"id": 12, "name": "3c", "supercategory": "crossbite",
     "description": "Crossbite with greater than 1 mm but up to 2 mm discrepancy"},
    {"id": 13, "name": "4c", "supercategory": "crossbite",
     "description": "Crossbite with greater than 2 mm discrepancy between retruded and intercuspal position"},
    {"id": 14, "name": "4i", "supercategory": "crossbite",
     "description": "Posterior lingual crossbite with no functional occlusal contact"},
    {"id": 15, "name": "2d", "supercategory": "displacement_of_contact_point",
     "description": "Contact point displacement greater than 1 mm but up to 2 mm"},
    {"id": 16, "name": "3d", "supercategory": "displacement_of_contact_point",
     "description": "Contact point displacement greater than 2 mm but up to 4 mm"},
    {"id": 17, "name": "4d", "supercategory": "displacement_of_contact_point",
     "description": "Severe contact point displacement greater than 4 mm"},
    {"id": 18, "name": "4t", "supercategory": "displacement_of_contact_point",
     "description": "Partially erupted, tipped, and impacted teeth"},
    {"id": 19, "name": "4x", "supercategory": "displacement_of_contact_point",
     "description": "Presence of supernumerary teeth"},
    {"id": 20, "name": "2e", "supercategory": "open_bite",
     "description": "Anterior or posterior open bite greater than 1 mm but up to 2 mm"},
    {"id": 21, "name": "3e", "supercategory": "open_bite",
     "description": "Lateral or anterior open bite greater than 2 mm but up to 4 mm"},
    {"id": 22, "name": "4e", "supercategory": "open_bite",
     "description": "Extreme lateral or anterior open bite greater than 4 mm"},
    {"id": 23, "name": "2f", "supercategory": "deepbite",
     "description": "Increased overbite at least 3.5 mm without gingival contact"},
    {"id": 24, "name": "3f", "supercategory": "deepbite",
     "description": "Deep overbite complete on gingival or palatal tissues but no trauma"},
    {"id": 25, "name": "4f", "supercategory": "deepbite",
     "description": "Increased and complete overbite with gingival or palatal trauma"},
    {"id": 26, "name": "2g", "supercategory": "molar_relationship",
     "description": "Pre-normal or post-normal occlusions with no other anomalies"},
    {"id": 27, "name": "4g", "supercategory": "molar_relationship",
     "description": "Severe pre-normal or post-normal occlusion"},
    {"id": 28, "name": "4h", "supercategory": "missing_teeth",
     "description": "Less extensive hypodontia requiring pre-restorative orthodontics or space closure"},
    {"id": 29, "name": "5h", "supercategory": "missing_teeth",
     "description": "Extensive hypodontia with restorative implications"},
    {"id": 30, "name": "5i", "supercategory": "missing_teeth",
     "description": "Impeded eruption of teeth except third molars"},
    {"id": 31, "name": "5s", "supercategory": "missing_teeth",
     "description": "Submerged deciduous teeth"},
    {"id": 32, "name": "5p", "supercategory": "cleft_lip_palate",
     "description": "Defects of cleft lip and palate and other craniofacial anomalies"},
]

ORTH_CATEGORY_MAP = {category["name"]: category["id"] for category in ORTH_CATEGORIES_INFO}
ORTH_CATEGORY_INFO_BY_NAME = {category["name"]: category for category in ORTH_CATEGORIES_INFO}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sample_key(path: Path, data_dir: Path, marker: str) -> tuple[str, ...] | None:
    """Identify the image/annotation pair a file belongs to.

    Drops the ``marker`` path component (``"images"`` or ``"anno"``) and the
    file suffix, so ``<sample>/images/D/41.png`` and ``<sample>/anno/D/41.json``
    both key to ``(<sample>, "D", "41")``.
    """
    rel_parts = list(path.relative_to(data_dir).parts)
    if marker not in rel_parts:
        return None
    marker_index = len(rel_parts) - 1 - rel_parts[::-1].index(marker)
    stem = Path(rel_parts[-1]).stem
    return tuple(rel_parts[:marker_index] + rel_parts[marker_index + 1 : -1] + [stem])


def image_path_for_json(json_path: Path, data_dir: Path, default_suffix: str) -> Path:
    """Mirror an ``anno/...json`` path into its ``images/...`` counterpart.

    Prefers whichever suffix in ``IMAGE_SUFFIXES`` actually exists on disk and
    falls back to ``default_suffix``. Only used for records whose image file is
    missing (``--allow-missing-images``), and only ever called with a
    ``json_path`` that has ``anno`` as a path component.
    """
    rel_parts = list(json_path.relative_to(data_dir).parts)
    marker_index = len(rel_parts) - 1 - rel_parts[::-1].index("anno")
    rel_parts[marker_index] = "images"
    base = (data_dir / Path(*rel_parts)).with_suffix("")
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return base.with_suffix(default_suffix)


def image_size_from_file_or_json(
    image_path: Path | None,
    data: dict | None,
    *,
    required: bool,
) -> tuple[int, int]:
    """Return ``(width, height)``.

    Reads the image file first, then falls back to the LabelMe ``imageWidth`` /
    ``imageHeight`` fields and finally to an inlined ``imageData`` blob. When
    nothing is available, raise if ``required`` else return ``(0, 0)``.
    """
    if image_path and image_path.exists():
        image = cv2.imread(str(image_path))
        if image is not None:
            height, width = image.shape[:2]
            return width, height

    if data:
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width > 0 and height > 0:
            return width, height

        image_data = data.get("imageData")
        if image_data:
            raw = base64.b64decode(image_data)
            with Image.open(io.BytesIO(raw)) as image:
                return image.size

    if required:
        raise ValueError(f"Cannot determine image size for {image_path}")
    return 0, 0


def collect_image_records(data_dir: Path) -> dict[tuple[str, ...], dict]:
    """Pair every image under ``images/`` with the annotation under ``anno/``.

    Seeded from the images, so an image with no annotation still produces a
    record (a negative sample).
    """
    records: dict[tuple[str, ...], dict] = {}

    for image_path in sorted(data_dir.rglob("*")):
        if not (image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES):
            continue
        key = sample_key(image_path, data_dir, "images")
        if key is None:
            continue
        records.setdefault(key, {"image_path": None, "json_path": None, "data": None})
        records[key]["image_path"] = image_path

    for json_path in sorted(data_dir.rglob("*.json")):
        key = sample_key(json_path, data_dir, "anno")
        if key is None:
            continue
        data = load_json(json_path)
        if {"images", "annotations", "categories"}.issubset(data):
            continue
        if not any(field in data for field in ("shapes", "imagePath", "imageData", "imageWidth")):
            continue
        record = records.setdefault(key, {"image_path": None, "json_path": None, "data": None})
        record["json_path"] = json_path
        record["data"] = data

    return dict(sorted(records.items()))


# ---------------------------------------------------------------------------
# Caries: shapes and to-coco
# ---------------------------------------------------------------------------

def caries_polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a LabelMe polygon."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def caries_shape_annotations(
    data: dict | None,
    json_path: Path | None,
    label_to_category_id: dict[str, int],
    label_metadata: dict[str, dict],
    severity_cutoffs: list[int],
) -> tuple[list[dict], int, bool]:
    """Build caries annotations for one record.

    Returns ``(annotations, skipped_shapes, is_unhealthy)``. ``flags["1"]`` is
    true only for unhealthy teeth; healthy ones stay in the dataset without
    annotations (negative samples).
    """
    flags = (data or {}).get("flags") or {}
    if flags.get("1") is not True:
        return [], 0, False

    annotations: list[dict] = []
    skipped = 0
    for shape in (data or {}).get("shapes") or []:
        label = str(shape.get("label", "")).strip()
        points = shape.get("points") or []

        if label not in label_to_category_id:
            print(f"Skipping unknown label '{label}' in {json_path}")
            skipped += 1
            continue

        if len(points) < 3:
            skipped += 1
            continue  # not a polygon

        np_points = np.asarray(points, dtype=float)
        x_min = float(np.min(np_points[:, 0]))
        y_min = float(np.min(np_points[:, 1]))
        x_max = float(np.max(np_points[:, 0]))
        y_max = float(np.max(np_points[:, 1]))

        annotations.append(
            {
                "category_id": label_to_category_id[label],
                "segmentation": [np_points.flatten().tolist()],
                "area": caries_polygon_area(np_points),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "iscrowd": 0,
            }
        )
    return annotations, skipped, True


# ---------------------------------------------------------------------------
# Orth: shapes and to-coco
# ---------------------------------------------------------------------------

def build_orth_supercategory_info() -> tuple[dict[str, int], list[dict], dict[str, dict]]:
    """Collapse the 32 IOTN/DHC codes into their anomaly supercategories."""
    supercategory_to_codes: dict[str, list[str]] = defaultdict(list)
    for category in ORTH_CATEGORIES_INFO:
        supercategory_to_codes[category["supercategory"]].append(category["name"])

    categories: list[dict] = []
    label_to_category_id: dict[str, int] = {}
    label_metadata: dict[str, dict] = {}
    for category in ORTH_CATEGORIES_INFO:
        label = category["name"]
        supercategory = category["supercategory"]
        if supercategory not in label_to_category_id:
            category_id = len(categories) + 1
            label_to_category_id[supercategory] = category_id
            categories.append(
                {
                    "id": category_id,
                    "name": supercategory,
                    "supercategory": "orthodontic_anomaly",
                    "source_codes": supercategory_to_codes[supercategory],
                }
            )
        label_to_category_id[label] = label_to_category_id[supercategory]
        label_metadata[label] = {
            "orth_code": label,
            "orth_code_id": category["id"],
            "orth_supercategory": supercategory,
        }
    return label_to_category_id, categories, label_metadata


def orth_categories_from_labels(
    labels: list[str] | None,
    label_mode: str,
) -> tuple[dict[str, int], list[dict], dict[str, dict]]:
    if label_mode == "supercategory" and labels is None:
        return build_orth_supercategory_info()

    if labels is None:
        label_metadata = {
            category["name"]: {
                "orth_code": category["name"],
                "orth_code_id": category["id"],
                "orth_supercategory": category["supercategory"],
            }
            for category in ORTH_CATEGORIES_INFO
        }
        return ORTH_CATEGORY_MAP, ORTH_CATEGORIES_INFO, label_metadata

    if label_mode == "supercategory":
        raise ValueError("--labels can only be used with --label-mode code")

    categories = [
        {"id": index + 1, "name": label, "supercategory": "orthodontic_anomaly"}
        for index, label in enumerate(labels)
    ]
    label_metadata = {
        category["name"]: {
            "orth_code": category["name"],
            "orth_code_id": ORTH_CATEGORY_INFO_BY_NAME.get(category["name"], {}).get("id"),
            "orth_supercategory": ORTH_CATEGORY_INFO_BY_NAME.get(category["name"], {}).get("supercategory"),
        }
        for category in categories
    }
    return {category["name"]: category["id"] for category in categories}, categories, label_metadata


def parse_severity(label: str) -> int | None:
    match = re.match(r"^(\d+)", label)
    return int(match.group(1)) if match else None


def parse_severity_cutoffs(cutoffs: str | None) -> list[int]:
    if not cutoffs:
        return []
    values = []
    for item in cutoffs.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return sorted(values)


def severity_group(severity_raw: int | None, cutoffs: list[int]) -> int | None:
    if severity_raw is None:
        return None
    if not cutoffs:
        return severity_raw
    return 1 + sum(severity_raw > cutoff for cutoff in cutoffs)


def orth_valid_shape(shape: dict) -> bool:
    label = str(shape.get("label", "")).strip()
    points = shape.get("points") or []
    shape_type = shape.get("shape_type", "rectangle")
    return bool(label) and len(points) >= 2 and shape_type in {"rectangle", "polygon"}


def normalize_box(points: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [float(point[0]) for point in pts]
    ys = [float(point[1]) for point in pts]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(flat_points: list[float]) -> float:
    if len(flat_points) < 6:
        return 0.0
    points = list(zip(flat_points[0::2], flat_points[1::2]))
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def orth_shape_annotations(
    data: dict | None,
    json_path: Path | None,
    label_to_category_id: dict[str, int],
    label_metadata: dict[str, dict],
    severity_cutoffs: list[int],
) -> tuple[list[dict], int, bool]:
    """Build orthodontic annotations for one record.

    Returns ``(annotations, skipped_shapes, is_unhealthy)``; orth has no
    healthy/unhealthy gate so the last flag is always ``False``.
    """
    annotations: list[dict] = []
    skipped = 0
    for shape in (data or {}).get("shapes", []):
        if not orth_valid_shape(shape):
            skipped += 1
            continue

        label = str(shape["label"]).strip()
        if label not in label_to_category_id:
            skipped += 1
            continue

        points = shape["points"]
        x1, y1, x2, y2 = normalize_box(points)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            skipped += 1
            continue

        flat_points = [float(value) for point in points for value in point[:2]]
        area = polygon_area(flat_points)
        if area <= 0:
            area = width * height

        metadata = label_metadata.get(label, {})
        severity_raw = parse_severity(label)
        annotation = {
            "category_id": label_to_category_id[label],
            "bbox": [x1, y1, width, height],
            "area": area,
            "iscrowd": 0,
            "orth_code": metadata.get("orth_code", label),
            "orth_code_id": metadata.get("orth_code_id"),
            "orth_supercategory": metadata.get("orth_supercategory"),
            "severity_raw": severity_raw,
            "severity_id": severity_group(severity_raw, severity_cutoffs),
        }
        if len(points) >= 3:
            annotation["segmentation"] = [flat_points]
        annotations.append(annotation)
    return annotations, skipped, False


# ---------------------------------------------------------------------------
# to-coco
# ---------------------------------------------------------------------------

def _coco_header(task: str, categories: list[dict]) -> dict:
    if task == "caries":
        info = {
            "description": "Tooth Caries Sample Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        }
    else:
        info = {
            "description": "Intraoral orthodontic anomaly object detection dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        }
    return {
        "info": info,
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def convert_to_coco(
    task: str,
    data_dir: Path,
    output_json: Path,
    *,
    labels: list[str] | None = None,
    label_mode: str = "supercategory",
    severity_cutoffs: str | None = None,
    drop_empty_images: bool = False,
    allow_missing_images: bool = False,
) -> dict:
    """Convert raw LabelMe annotations into a COCO JSON. Returns the COCO dict."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")

    data_dir = data_dir.resolve()
    if task == "caries":
        label_to_category_id = CARIES_CATEGORY_MAP
        categories_info = CARIES_CATEGORIES_INFO
        label_metadata: dict[str, dict] = {}
        shape_fn: Callable = caries_shape_annotations
    else:
        label_to_category_id, categories_info, label_metadata = orth_categories_from_labels(
            labels, label_mode
        )
        shape_fn = orth_shape_annotations

    severity_cutoff_values = parse_severity_cutoffs(severity_cutoffs)
    coco = _coco_header(task, categories_info)

    image_id = 1
    annotation_id = 1
    skipped_shapes = 0
    unhealthy_images = 0
    negative_images = 0
    unpaired_json: list[Path] = []

    for record in collect_image_records(data_dir).values():
        image_path: Path | None = record["image_path"]
        json_path: Path | None = record["json_path"]
        data: dict | None = record["data"]

        if image_path is None:
            if json_path is not None:
                unpaired_json.append(json_path)
            if not allow_missing_images or json_path is None:
                continue
        elif json_path is None:
            data = None
        elif data is None:
            data = load_json(json_path)

        annotations, skipped, is_unhealthy = shape_fn(
            data, json_path, label_to_category_id, label_metadata, severity_cutoff_values
        )
        skipped_shapes += skipped
        if is_unhealthy:
            unhealthy_images += 1

        if drop_empty_images and not annotations:
            continue
        if not annotations:
            negative_images += 1

        width, height = image_size_from_file_or_json(image_path, data, required=(task == "orth"))

        if image_path is not None:
            file_name = image_path.resolve().relative_to(data_dir).as_posix()
        elif json_path is not None:
            mirrored = image_path_for_json(json_path, data_dir, FALLBACK_IMAGE_SUFFIX[task])
            file_name = mirrored.relative_to(data_dir).as_posix()
        else:
            raise ValueError("Record has neither image nor JSON path")

        coco["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "height": height,
                "width": width,
                "date_captured": datetime.now().strftime("%Y-%m-%d"),
            }
        )
        for annotation in annotations:
            annotation["id"] = annotation_id
            annotation["image_id"] = image_id
            coco["annotations"].append(annotation)
            annotation_id += 1
        image_id += 1

    write_json(coco, output_json)

    print(f"Saved COCO JSON to {output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])} -> {[c['name'] for c in coco['categories']]}")
    if task == "caries":
        print(f"Unhealthy images (flags['1'] is true): {unhealthy_images}")
        print(f"Negative images (no annotations): {negative_images}")
    if skipped_shapes:
        print(f"Skipped invalid/unknown shapes: {skipped_shapes}")
    if unpaired_json:
        verb = "Included" if allow_missing_images else "Skipped"
        print(f"{verb} {len(unpaired_json)} annotation(s) with no image file:")
        for path in unpaired_json[:20]:
            print(f"  {path}")
        if len(unpaired_json) > 20:
            print(f"  ... and {len(unpaired_json) - 20} more")

    return coco


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def _caries_sample_id(file_name: str) -> str:
    """The path component right before the ``images`` marker."""
    parts = Path(file_name).parts
    try:
        images_index = parts.index("images")
    except ValueError:
        raise ValueError(
            f"file_name {file_name!r} has no 'images' path component; expected "
            "the migrated <sample_id>/images/<view>/<name> layout."
        ) from None
    if images_index == 0:
        raise ValueError(f"file_name {file_name!r} has no sample id before 'images'.")
    return parts[images_index - 1]


def _orth_sample_id(file_name: str) -> str:
    parts = Path(file_name).parts
    return parts[0] if len(parts) > 1 else Path(file_name).stem


def _representative_label(labels: set[int], category_counts: Counter) -> int:
    if not labels:
        return 0
    return min(labels, key=lambda label: (category_counts[label], label))


def _split_caries(
    coco_data: dict,
    *,
    ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Stratified split by sample id, stratified on each sample's rarest class."""
    from sklearn.model_selection import train_test_split

    images = coco_data["images"]
    annotations = coco_data["annotations"]

    img_id_to_sample = {img["id"]: _caries_sample_id(img["file_name"]) for img in images}
    img_id_to_labels: dict[int, set[int]] = defaultdict(set)
    for anno in annotations:
        img_id_to_labels[anno["image_id"]].add(anno["category_id"])

    # The sample id is the component before the "images" marker, so every image
    # of one tooth lands in the same split.
    sample_to_labels: dict[str, set[int]] = defaultdict(set)
    for img in images:
        sample_id = img_id_to_sample[img["id"]]
        if img["id"] in img_id_to_labels:
            sample_to_labels[sample_id].update(img_id_to_labels[img["id"]])
        else:
            sample_to_labels[sample_id].add(0)  # 0 marks a healthy sample

    sample_list: list[str] = []
    stratify_labels: list[int] = []
    for sample_id, labels in sample_to_labels.items():
        sample_list.append(sample_id)
        # The highest id is usually the rarest lesion, so it keeps the rare
        # classes represented on both sides.
        stratify_labels.append(max(labels) if labels else 0)

    # A label with a single sample cannot be stratified; force those to train.
    label_counts = Counter(stratify_labels)
    safe_samples: list[str] = []
    safe_labels: list[int] = []
    orphan_samples: list[str] = []
    for sample_id, label in zip(sample_list, stratify_labels):
        if label_counts[label] < 2:
            orphan_samples.append(sample_id)
            print(f"Warning: class id {label} has a single sample; forced into train: {sample_id}")
        else:
            safe_samples.append(sample_id)
            safe_labels.append(label)

    if not safe_samples:
        train_samples, test_samples = train_test_split(sample_list, train_size=ratio, random_state=seed)
    else:
        train_samples, test_samples = train_test_split(
            safe_samples,
            train_size=ratio,
            stratify=safe_labels,
            random_state=seed,
        )
        train_samples = list(train_samples) + orphan_samples

    return set(train_samples), set(test_samples)


def _split_orth(
    coco_data: dict,
    *,
    ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Group-aware split by sample id, spread per rarest class."""
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    image_to_sample = {image["id"]: _orth_sample_id(image["file_name"]) for image in images}
    image_to_labels: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        image_to_labels[ann["image_id"]].add(ann["category_id"])

    sample_to_images: dict[str, list[int]] = defaultdict(list)
    sample_to_labels: dict[str, set[int]] = defaultdict(set)
    for image in images:
        sample_id = image_to_sample[image["id"]]
        sample_to_images[sample_id].append(image["id"])
        labels = image_to_labels.get(image["id"])
        if labels:
            sample_to_labels[sample_id].update(labels)
        else:
            sample_to_labels[sample_id].add(0)

    category_counts = Counter(
        label for labels in sample_to_labels.values() for label in labels if label != 0
    )
    sample_labels = {
        sample_id: _representative_label(labels, category_counts)
        for sample_id, labels in sample_to_labels.items()
    }

    rng = random.Random(seed)
    label_to_samples: dict[int, list[str]] = defaultdict(list)
    for sample_id, label in sample_labels.items():
        label_to_samples[label].append(sample_id)

    train_samples: set[str] = set()
    test_samples: set[str] = set()
    for label, sample_ids in sorted(label_to_samples.items()):
        # Every sample is assigned within its own label bucket, so each label
        # reaches both sides without the rare ones disappearing.
        rng.shuffle(sample_ids)
        if len(sample_ids) < 2:
            train_samples.update(sample_ids)
            continue
        train_count = max(1, min(len(sample_ids) - 1, round(len(sample_ids) * ratio)))
        train_samples.update(sample_ids[:train_count])
        test_samples.update(sample_ids[train_count:])

    if not test_samples:
        # Degenerate fallback: a random split over all samples.
        all_samples = sorted(sample_to_images)
        rng.shuffle(all_samples)
        test_count = max(1, round(len(all_samples) * (1.0 - ratio)))
        test_samples = set(all_samples[:test_count])
        train_samples = set(all_samples[test_count:])

    return train_samples, test_samples


def split_coco_dataset(
    task: str,
    input_json: Path,
    train_json: Path,
    test_json: Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Split a COCO JSON into train/test by sample id and write both files."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")

    coco_data = load_json(input_json)
    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    if task == "caries":
        train_samples, test_samples = _split_caries(coco_data, ratio=train_ratio, seed=seed)
    else:
        train_samples, test_samples = _split_orth(coco_data, ratio=train_ratio, seed=seed)

    sample_id_of = _caries_sample_id if task == "caries" else _orth_sample_id
    train_image_ids = {img["id"] for img in images if sample_id_of(img["file_name"]) in train_samples}
    test_image_ids = {img["id"] for img in images if sample_id_of(img["file_name"]) in test_samples}

    def build_subset(image_ids: set[int]) -> dict:
        return {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": categories,
            "images": [img for img in images if img["id"] in image_ids],
            "annotations": [ann for ann in annotations if ann["image_id"] in image_ids],
        }

    train_coco = build_subset(train_image_ids)
    test_coco = build_subset(test_image_ids)

    write_json(train_coco, train_json)
    write_json(test_coco, test_json)

    print(f"Split complete: {input_json} -> {train_json.name} / {test_json.name}")
    print(
        f"Train: {len(train_samples)} samples, "
        f"{len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations"
    )
    print(
        f"Test:  {len(test_samples)} samples, "
        f"{len(test_coco['images'])} images, {len(test_coco['annotations'])} annotations"
    )

    train_dist = Counter(ann["category_id"] for ann in train_coco["annotations"])
    test_dist = Counter(ann["category_id"] for ann in test_coco["annotations"])
    print("\nClass distribution (train vs test):")
    for category in categories:
        category_id = category["id"]
        train_count = train_dist.get(category_id, 0)
        test_count = test_dist.get(category_id, 0)
        total = train_count + test_count
        share = train_count / total if total else 0.0
        print(
            f" - {category['name']:<25} Train={train_count:<4} Test={test_count:<4} "
            f"(train ratio: {share:.1%})"
        )


# ---------------------------------------------------------------------------
# Paths and staleness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskPaths:
    input_dir: Path
    data_dir: Path
    coco_json: Path
    train_json: Path
    test_json: Path


def resolve_paths(task: str, input_dir: Path | None = None) -> TaskPaths:
    """Resolve every path this task touches.

    ``data_dir`` / ``train_json`` / ``test_json`` come from
    ``utils.task_paths.resolve_task_paths`` so this pipeline and the training
    entry points can never disagree about where a dataset lives.
    """
    resolved = resolve_task_paths(argparse.Namespace(task=task, input_dir=input_dir))
    return TaskPaths(
        input_dir=resolved.input_dir,
        data_dir=resolved.data_dir,
        coco_json=resolved.input_dir / COCO_JSON_NAME[task],
        train_json=resolved.train_json,
        test_json=resolved.test_json,
    )


def _newest_mtime(paths: Iterable[Path]) -> float:
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def _raw_files(data_dir: Path) -> list[Path]:
    return [path for path in data_dir.rglob("*") if path.is_file()]


def decide_to_coco(paths: TaskPaths, overwrite: bool) -> tuple[bool, str]:
    if overwrite:
        return True, "--overwrite"
    if not paths.coco_json.exists():
        return True, f"{paths.coco_json.name} is missing"
    if not paths.data_dir.exists():
        return False, f"{paths.data_dir} not found; keeping the existing COCO JSON"
    if _newest_mtime(_raw_files(paths.data_dir)) > paths.coco_json.stat().st_mtime:
        return True, "raw annotations are newer than the COCO JSON"
    return False, f"{paths.coco_json.name} is up to date"


def decide_split(paths: TaskPaths, overwrite: bool) -> tuple[bool, str]:
    if overwrite:
        return True, "--overwrite"
    missing = [path.name for path in (paths.train_json, paths.test_json) if not path.exists()]
    if missing:
        return True, f"missing {' and '.join(missing)}"
    if not paths.coco_json.exists():
        return True, f"{paths.coco_json.name} is missing"
    oldest_split = min(paths.train_json.stat().st_mtime, paths.test_json.stat().st_mtime)
    if paths.coco_json.stat().st_mtime > oldest_split:
        return True, "the COCO JSON is newer than the train/test splits"
    return False, "train/test splits are up to date"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the caries/orth COCO JSONs from raw LabelMe annotations. "
            "Steps whose outputs are already up to date are skipped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=list(TASKS), help="Which dataset to build.")
    parser.add_argument(
        "--input-dir",
        default=None,
        type=Path,
        help="Dataset root; defaults to the task's path in utils/task_paths.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even when the existing outputs are up to date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which steps would run, writing nothing.",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Do not include images without valid annotations.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="[caries] Also emit records whose image file is missing.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="[orth] Explicit label order; overrides the built-in IOTN/DHC table.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["code", "supercategory"],
        default="supercategory",
        help=(
            "[orth] code keeps the 32 IOTN/DHC classes; supercategory collapses "
            "them into broader orthodontic anomaly groups."
        ),
    )
    parser.add_argument(
        "--severity-cutoffs",
        default=None,
        help="[orth] Comma-separated severity cutoffs, e.g. 3,4.",
    )
    parser.add_argument("--train-ratio", default=0.8, type=float, help="Train split ratio.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    return parser.parse_args(argv)


def _validate_task_args(args: argparse.Namespace) -> None:
    if args.task == "caries":
        if args.labels is not None or args.label_mode != "supercategory" or args.severity_cutoffs:
            raise SystemExit("--labels / --label-mode / --severity-cutoffs are orth-only.")
    elif args.allow_missing_images:
        raise SystemExit("--allow-missing-images is caries-only.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_task_args(args)
    paths = resolve_paths(args.task, args.input_dir)

    print(f"task       : {args.task}")
    print(f"input_dir  : {paths.input_dir}")
    print(f"data_dir   : {paths.data_dir}")
    print(f"coco_json  : {paths.coco_json}")
    print(f"train/test : {paths.train_json.name} / {paths.test_json.name}")

    run_coco, coco_reason = decide_to_coco(paths, args.overwrite)
    run_split, split_reason = decide_split(paths, args.overwrite)
    if run_coco:
        # A freshly built COCO JSON invalidates any existing split.
        run_split, split_reason = True, "the COCO JSON is being rebuilt"

    print(f"[1/2] to-coco : {'RUN ' if run_coco else 'SKIP'} ({coco_reason})")
    print(f"[2/2] split   : {'RUN ' if run_split else 'SKIP'} ({split_reason})")

    if args.dry_run:
        print("dry run: nothing written.")
        return
    if not run_coco and not run_split:
        print("Everything is up to date; nothing to do.")
        return

    if run_coco:
        if not paths.data_dir.exists():
            raise SystemExit(f"Raw data directory not found: {paths.data_dir}")
        convert_to_coco(
            args.task,
            paths.data_dir,
            paths.coco_json,
            labels=args.labels,
            label_mode=args.label_mode,
            severity_cutoffs=args.severity_cutoffs,
            drop_empty_images=args.drop_empty_images,
            allow_missing_images=args.allow_missing_images,
        )

    if run_split:
        if not paths.coco_json.exists():
            raise SystemExit(f"COCO JSON not found: {paths.coco_json}")
        split_coco_dataset(
            args.task,
            paths.coco_json,
            paths.train_json,
            paths.test_json,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )

    print("Done.")


if __name__ == "__main__":
    main()
