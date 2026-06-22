#!/usr/bin/env python3
"""Convert orth_test LabelMe annotations to COCO object-detection format.

The input folder is expected to look like:

    .datasets/intraoral_anno/orth_test/
      22012/
        D.jpg
        R.jpg
        R.json

JSON files are LabelMe annotations. Shapes with ``rectangle`` or ``polygon``
points become COCO bbox annotations. Images without JSON annotations are kept as
negative samples, which is useful for Faster R-CNN training.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Fixed IOTN/DHC categories read from tmp/a.jpg and tmp/b.jpg.
# LabelMe shape labels should use the short code, e.g. "3a" or "4d".
CATEGORIES_INFO = [
    {
        "id": 1,
        "name": "1",
        "supercategory": "minor_malocclusion",
        "description": "Extremely minor malocclusions including contact point displacement less than 1 mm",
    },
    {
        "id": 2,
        "name": "2a",
        "supercategory": "increased_overjet",
        "description": "Increased overjet greater than 3.5 mm but up to 6 mm with competent lips",
    },
    {
        "id": 3,
        "name": "2b",
        "supercategory": "reverse_overjet",
        "description": "Reverse overjet greater than 0 mm but up to 1 mm",
    },
    {
        "id": 4,
        "name": "2c",
        "supercategory": "crossbite",
        "description": "Crossbite with up to 1 mm discrepancy",
    },
    {
        "id": 5,
        "name": "2d",
        "supercategory": "displacement_of_contact_point",
        "description": "Contact point displacement greater than 1 mm but up to 2 mm",
    },
    {
        "id": 6,
        "name": "2e",
        "supercategory": "open_bite",
        "description": "Anterior or posterior open bite greater than 1 mm but up to 2 mm",
    },
    {
        "id": 7,
        "name": "2f",
        "supercategory": "deepbite",
        "description": "Increased overbite at least 3.5 mm without gingival contact",
    },
    {
        "id": 8,
        "name": "2g",
        "supercategory": "molar_relationship",
        "description": "Pre-normal or post-normal occlusions with no other anomalies",
    },
    {
        "id": 9,
        "name": "3a",
        "supercategory": "increased_overjet",
        "description": "Increased overjet greater than 3.5 mm but up to 6 mm with incompetent lips",
    },
    {
        "id": 10,
        "name": "3b",
        "supercategory": "reverse_overjet",
        "description": "Reverse overjet greater than 1 mm but up to 3.5 mm",
    },
    {
        "id": 11,
        "name": "3c",
        "supercategory": "crossbite",
        "description": "Crossbite with greater than 1 mm but up to 2 mm discrepancy",
    },
    {
        "id": 12,
        "name": "3d",
        "supercategory": "displacement_of_contact_point",
        "description": "Contact point displacement greater than 2 mm but up to 4 mm",
    },
    {
        "id": 13,
        "name": "3e",
        "supercategory": "open_bite",
        "description": "Lateral or anterior open bite greater than 2 mm but up to 4 mm",
    },
    {
        "id": 14,
        "name": "3f",
        "supercategory": "deepbite",
        "description": "Deep overbite complete on gingival or palatal tissues but no trauma",
    },
    {
        "id": 15,
        "name": "4a",
        "supercategory": "increased_overjet",
        "description": "Increased overjet greater than 6 mm but up to 9 mm",
    },
    {
        "id": 16,
        "name": "4b",
        "supercategory": "reverse_overjet",
        "description": "Reverse overjet greater than 3.5 mm with no masticatory or speech difficulties",
    },
    {
        "id": 17,
        "name": "4c",
        "supercategory": "crossbite",
        "description": "Crossbite with greater than 2 mm discrepancy between retruded and intercuspal position",
    },
    {
        "id": 18,
        "name": "4d",
        "supercategory": "displacement_of_contact_point",
        "description": "Severe contact point displacement greater than 4 mm",
    },
    {
        "id": 19,
        "name": "4e",
        "supercategory": "open_bite",
        "description": "Extreme lateral or anterior open bite greater than 4 mm",
    },
    {
        "id": 20,
        "name": "4f",
        "supercategory": "deepbite",
        "description": "Increased and complete overbite with gingival or palatal trauma",
    },
    {
        "id": 21,
        "name": "4g",
        "supercategory": "molar_relationship",
        "description": "Severe pre-normal or post-normal occlusion",
    },
    {
        "id": 22,
        "name": "4h",
        "supercategory": "missing_teeth",
        "description": "Less extensive hypodontia requiring pre-restorative orthodontics or space closure",
    },
    {
        "id": 23,
        "name": "4l",
        "supercategory": "crossbite",
        "description": "Posterior lingual crossbite with no functional occlusal contact",
    },
    {
        "id": 24,
        "name": "4m",
        "supercategory": "reverse_overjet",
        "description": "Reverse overjet greater than 1 mm but less than 3.5 mm with recorded difficulties",
    },
    {
        "id": 25,
        "name": "4t",
        "supercategory": "displacement_of_contact_point",
        "description": "Partially erupted, tipped, and impacted teeth",
    },
    {
        "id": 26,
        "name": "4x",
        "supercategory": "displacement_of_contact_point",
        "description": "Presence of supernumerary teeth",
    },
    {
        "id": 27,
        "name": "5a",
        "supercategory": "increased_overjet",
        "description": "Increased overjet greater than 9 mm",
    },
    {
        "id": 28,
        "name": "5h",
        "supercategory": "missing_teeth",
        "description": "Extensive hypodontia with restorative implications",
    },
    {
        "id": 29,
        "name": "5i",
        "supercategory": "missing_teeth",
        "description": "Impeded eruption of teeth except third molars",
    },
    {
        "id": 30,
        "name": "5m",
        "supercategory": "reverse_overjet",
        "description": "Reverse overjet greater than 3.5 mm with reported masticatory and speech difficulties",
    },
    {
        "id": 31,
        "name": "5p",
        "supercategory": "cleft_lip_palate",
        "description": "Defects of cleft lip and palate and other craniofacial anomalies",
    },
    {
        "id": 32,
        "name": "5s",
        "supercategory": "missing_teeth",
        "description": "Submerged deciduous teeth",
    },
]

CATEGORY_MAP = {category["name"]: category["id"] for category in CATEGORIES_INFO}
CATEGORY_INFO_BY_NAME = {category["name"]: category for category in CATEGORIES_INFO}


def build_supercategory_info() -> tuple[dict[str, int], list[dict], dict[str, dict]]:
    supercategory_to_codes: dict[str, list[str]] = defaultdict(list)
    for category in CATEGORIES_INFO:
        supercategory_to_codes[category["supercategory"]].append(category["name"])

    categories = []
    label_to_category_id = {}
    label_metadata = {}
    for category in CATEGORIES_INFO:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert orth_test LabelMe annotations to COCO detection JSON."
    )
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_0616/orth_0616",
        type=Path,
        help="Root folder containing patient/sample subfolders.",
    )
    parser.add_argument(
        "--output-json",
        default=".datasets/intraoral_anno/orth_0616/orth_detection_coco.json",
        type=Path,
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "Optional custom label order. If omitted, the fixed IOTN/DHC "
            "CATEGORIES_INFO table in this file is used."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["code", "supercategory"],
        default="supercategory",
        help=(
            "code keeps the original 32 IOTN/DHC classes; supercategory maps "
            "annotation category_id values to broader orthodontic anomaly groups."
        ),
    )
    parser.add_argument(
        "--severity-cutoffs",
        default=None,
        help=(
            "Optional comma-separated severity cutoffs for annotation metadata. "
            "For example, 3,4 maps raw severities to <=3, 4, and >4 groups."
        ),
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Do not include images without valid annotations.",
    )
    return parser.parse_args()


def categories_from_labels(
    labels: list[str] | None,
    label_mode: str,
) -> tuple[dict[str, int], list[dict], dict[str, dict]]:
    if label_mode == "supercategory" and labels is None:
        return build_supercategory_info()

    if labels is None:
        label_metadata = {
            category["name"]: {
                "orth_code": category["name"],
                "orth_code_id": category["id"],
                "orth_supercategory": category["supercategory"],
            }
            for category in CATEGORIES_INFO
        }
        return CATEGORY_MAP, CATEGORIES_INFO, label_metadata

    if label_mode == "supercategory":
        raise ValueError("--labels can only be used with --label-mode code")

    categories = [
        {"id": index + 1, "name": label, "supercategory": "orthodontic_anomaly"}
        for index, label in enumerate(labels)
    ]
    label_metadata = {
        category["name"]: {
            "orth_code": category["name"],
            "orth_code_id": CATEGORY_INFO_BY_NAME.get(category["name"], {}).get("id"),
            "orth_supercategory": CATEGORY_INFO_BY_NAME.get(category["name"], {}).get("supercategory"),
        }
        for category in categories
    }
    return {category["name"]: category["id"] for category in categories}, categories, label_metadata


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_size_from_file_or_json(image_path: Path | None, data: dict | None) -> tuple[int, int]:
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

    raise ValueError("Cannot determine image size")


def resolve_image_path(json_path: Path, data: dict) -> Path | None:
    candidates: list[Path] = []
    image_path = data.get("imagePath")
    if image_path:
        candidates.append(json_path.parent / str(image_path))

    for suffix in IMAGE_SUFFIXES:
        candidates.append(json_path.with_suffix(suffix))
        candidates.append(json_path.with_suffix(suffix.upper()))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    stem = json_path.stem.strip().lower()
    for candidate in json_path.parent.iterdir():
        if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.stem.strip().lower() == stem:
            return candidate
    return None


def normalize_box(points: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [float(point[0]) for point in pts]
    ys = [float(point[1]) for point in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2, y2


def polygon_area(flat_points: list[float]) -> float:
    if len(flat_points) < 6:
        return 0.0
    points = list(zip(flat_points[0::2], flat_points[1::2]))
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def valid_shape(shape: dict) -> bool:
    label = str(shape.get("label", "")).strip()
    points = shape.get("points") or []
    shape_type = shape.get("shape_type", "rectangle")
    return bool(label) and len(points) >= 2 and shape_type in {"rectangle", "polygon"}


def infer_labels(data_dir: Path) -> list[str]:
    labels = set()
    for json_path in sorted(data_dir.rglob("*.json")):
        data = load_json(json_path)
        for shape in data.get("shapes", []):
            if valid_shape(shape):
                labels.add(str(shape["label"]).strip())
    return sorted(labels, key=natural_key)


def collect_image_records(data_dir: Path) -> dict[Path, dict]:
    records: dict[Path, dict] = {}

    for image_path in sorted(data_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            records[image_path.resolve()] = {"image_path": image_path, "json_path": None, "data": None}

    for json_path in sorted(data_dir.rglob("*.json")):
        data = load_json(json_path)
        if {"images", "annotations", "categories"}.issubset(data):
            continue
        if not any(key in data for key in ("shapes", "imagePath", "imageData", "imageWidth")):
            continue
        image_path = resolve_image_path(json_path, data)
        if image_path is None:
            key = json_path.resolve()
            records[key] = {"image_path": None, "json_path": json_path, "data": data}
        else:
            key = image_path.resolve()
            records[key] = {"image_path": image_path, "json_path": json_path, "data": data}

    return records


def convert_to_coco(
    data_dir: Path,
    output_json: Path,
    labels: list[str] | None,
    label_mode: str,
    severity_cutoffs: str | None,
    drop_empty_images: bool,
) -> None:
    data_dir = data_dir.resolve()
    label_to_category_id, categories_info, label_metadata = categories_from_labels(labels, label_mode)
    severity_cutoff_values = parse_severity_cutoffs(severity_cutoffs)

    coco = {
        "info": {
            "description": "Intraoral orthodontic anomaly object detection dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories_info,
    }

    image_id = 1
    annotation_id = 1
    skipped_shapes = 0

    for record in collect_image_records(data_dir).values():
        image_path: Path | None = record["image_path"]
        json_path: Path | None = record["json_path"]
        data: dict | None = record["data"]
        if json_path and data is None:
            data = load_json(json_path)

        annotations: list[dict] = []
        if data:
            for shape in data.get("shapes", []):
                if not valid_shape(shape):
                    skipped_shapes += 1
                    continue

                label = str(shape["label"]).strip()
                if label not in label_to_category_id:
                    skipped_shapes += 1
                    continue

                points = shape["points"]
                x1, y1, x2, y2 = normalize_box(points)
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0 or height <= 0:
                    skipped_shapes += 1
                    continue

                flat_points = [float(value) for point in points for value in point[:2]]
                area = polygon_area(flat_points)
                if area <= 0:
                    area = width * height

                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_category_id[label],
                    "bbox": [x1, y1, width, height],
                    "area": area,
                    "iscrowd": 0,
                }
                metadata = label_metadata.get(label, {})
                severity_raw = parse_severity(label)
                ann.update(
                    {
                        "orth_code": metadata.get("orth_code", label),
                        "orth_code_id": metadata.get("orth_code_id"),
                        "orth_supercategory": metadata.get("orth_supercategory"),
                        "severity_raw": severity_raw,
                        "severity_id": severity_group(severity_raw, severity_cutoff_values),
                    }
                )
                if len(points) >= 3:
                    ann["segmentation"] = [flat_points]
                annotations.append(ann)
                annotation_id += 1

        if drop_empty_images and not annotations:
            continue

        width, height = image_size_from_file_or_json(image_path, data)
        if image_path:
            file_name = image_path.resolve().relative_to(data_dir).as_posix()
        elif json_path:
            file_name = json_path.resolve().relative_to(data_dir).with_suffix(".jpg").as_posix()
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
        coco["annotations"].extend(annotations)
        image_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved COCO JSON to {output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])} -> {[c['name'] for c in coco['categories']]}")
    if skipped_shapes:
        print(f"Skipped invalid/unknown shapes: {skipped_shapes}")


if __name__ == "__main__":
    args = parse_args()
    convert_to_coco(
        data_dir=args.data_dir,
        output_json=args.output_json,
        labels=args.labels,
        label_mode=args.label_mode,
        severity_cutoffs=args.severity_cutoffs,
        drop_empty_images=args.drop_empty_images,
    )
