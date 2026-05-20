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
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert orth_test LabelMe annotations to COCO detection JSON."
    )
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_test",
        type=Path,
        help="Root folder containing patient/sample subfolders.",
    )
    parser.add_argument(
        "--output-json",
        default=".datasets/intraoral_anno/orth_test/orth_detection_coco.json",
        type=Path,
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "Optional fixed label order. If omitted, labels are inferred from "
            "all JSON files and naturally sorted."
        ),
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Do not include images without valid annotations.",
    )
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_size_from_file_or_json(image_path: Path | None, data: dict | None) -> tuple[int, int]:
    if image_path and image_path.exists():
        with Image.open(image_path) as image:
            return image.size

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
    drop_empty_images: bool,
) -> None:
    data_dir = data_dir.resolve()
    labels = labels or infer_labels(data_dir)
    label_to_category_id = {label: index + 1 for index, label in enumerate(labels)}

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
        "categories": [
            {"id": category_id, "name": label, "supercategory": "orthodontic_anomaly"}
            for label, category_id in label_to_category_id.items()
        ],
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
    print(f"Categories: {len(coco['categories'])} -> {labels}")
    if skipped_shapes:
        print(f"Skipped invalid/unknown shapes: {skipped_shapes}")


if __name__ == "__main__":
    args = parse_args()
    convert_to_coco(
        data_dir=args.data_dir,
        output_json=args.output_json,
        labels=args.labels,
        drop_empty_images=args.drop_empty_images,
    )
