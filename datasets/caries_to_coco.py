#!/usr/bin/env python3
"""Convert caries LabelMe annotations to a COCO instance-segmentation JSON.

The input layout and the image/annotation pairing rules are identical to
``orth_to_coco.py``::

    <data_dir>/<sample_id>/images/<any/depth>/<name>.png
    <data_dir>/<sample_id>/anno/<any/depth>/<name>.json

An image and its annotation are paired by their path relative to ``--data-dir``
with the ``images`` / ``anno`` component removed and the file suffix dropped, so
``2207/images/D/41.png`` and ``2207/anno/D/41.json`` both key to
``(2207, "D", "41")``. That means:

* the annotation folder depth is free (``anno/D/41.json`` or ``anno/41.json``),
* any suffix in ``IMAGE_SUFFIXES`` is accepted, not just ``.png``,
* images without an annotation are kept as negative samples; pass
  ``--drop-empty-images`` to remove them instead.

Caries-specific annotation rule: shapes are emitted **only** when the LabelMe
``flags["1"]`` marker is true (unhealthy tooth), so images flagged healthy are
negative samples. Shape labels must appear in ``CATEGORY_MAP``.

``file_name`` is stored relative to ``--data-dir`` so the JSON stays valid
regardless of the training working directory; the training data loader joins it
back with ``data_dir`` (see ``utils/unify_common.py``).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_DATA_DIR = ".datasets/intraoral_anno/single_ch_0225/single_tooth"
DEFAULT_OUTPUT_JSON = ".datasets/intraoral_anno/single_ch_0225/caries_sample_dataset.json"

# 定义类别映射: ori_lable_id -> coco_category_id
CATEGORY_MAP = {
    "1": 1,  # caries
    "11": 2, # white spot lesion
    "3": 3, # filling without caries
    "4": 4, # filling with caries
    "6": 5, # fissure sealant
    "10": 6, # non-caries disease(hard tissue)
    "12": 7, # staining
    "7": 8, # abnormal central cusp
    "8": 9, # palatal radicular groove
}

# 定义类别信息用于 COCO header
CATEGORIES_INFO = [
    {"id": 1, "name": "caries", "supercategory": "tooth_disease"},
    {"id": 2, "name": "white_spot_lesion", "supercategory": "tooth_disease"},
    {"id": 3, "name": "filling_no_caries", "supercategory": "treatment"},
    {"id": 4, "name": "filling_with_caries", "supercategory": "tooth_disease"},
    {"id": 5, "name": "fissure_sealant", "supercategory": "treatment"},
    {"id": 6, "name": "non_caries_hard_tissue", "supercategory": "tooth_disease"},
    {"id": 7, "name": "staining", "supercategory": "clinical_finding"},
    {"id": 8, "name": "abnormal_central_cusp", "supercategory": "anatomy_anomaly"},
    {"id": 9, "name": "palatal_radicular_groove", "supercategory": "anatomy_anomaly"}
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert caries LabelMe annotations to a COCO instance-segmentation JSON."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Root folder whose subfolders are sample ids, each holding images/ and anno/.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        type=Path,
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Do not include images without valid annotations.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help=(
            "Also emit records for annotations whose image file is missing, "
            "using the mirrored images/ path (orth_to_coco.py behaviour). Off "
            "by default so every emitted file_name exists on disk."
        ),
    )
    return parser.parse_args()


def calculate_polygon_area(points: np.ndarray) -> float:
    """使用鞋带公式计算多边形面积（与原 caries 转换脚本完全一致）。"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_key(path: Path, data_dir: Path, marker: str) -> tuple[str, ...] | None:
    """Identify the image/annotation pair a file belongs to.

    Drops the ``marker`` path component (``"images"`` or ``"anno"``) and the
    file suffix, so ``<sample>/images/D/41.png`` and ``<sample>/anno/D/41.json``
    both key to ``(<sample>, "D", "41")``. Mirrors ``orth_to_coco.sample_key``.
    """
    rel_parts = list(path.relative_to(data_dir).parts)
    if marker not in rel_parts:
        return None
    marker_index = len(rel_parts) - 1 - rel_parts[::-1].index(marker)
    stem = Path(rel_parts[-1]).stem
    return tuple(rel_parts[:marker_index] + rel_parts[marker_index + 1 : -1] + [stem])


def image_path_for_json(json_path: Path, data_dir: Path) -> Path:
    """Mirror an ``anno/...json`` path into its ``images/...`` counterpart.

    Prefers whichever suffix in ``IMAGE_SUFFIXES`` actually exists on disk and
    falls back to ``.png``, the caries convention. Only used for records whose
    image file is missing (``--allow-missing-images``).
    """
    rel_parts = list(json_path.relative_to(data_dir).parts)
    marker_index = len(rel_parts) - 1 - rel_parts[::-1].index("anno")
    rel_parts[marker_index] = "images"
    base = (data_dir / Path(*rel_parts)).with_suffix("")
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return base.with_suffix(".png")


def image_size_from_file_or_json(image_path: Path | None, data: dict | None) -> tuple[int, int]:
    """Return ``(width, height)``, reading the image and falling back to LabelMe fields."""
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

    return 0, 0


def collect_image_records(data_dir: Path) -> dict[tuple[str, ...], dict]:
    """Pair every image under ``images/`` with the annotation under ``anno/``.

    Seeded from the images, so an image with no annotation still produces a
    record (a negative sample). Mirrors ``orth_to_coco.collect_image_records``.
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


def convert_to_coco(
    data_dir: Path,
    output_json: Path,
    drop_empty_images: bool = False,
    allow_missing_images: bool = False,
) -> None:
    data_dir = data_dir.resolve()
    coco_output = {
        "info": {
            "description": "Tooth Caries Sample Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES_INFO,
    }

    image_id_counter = 1
    annotation_id_counter = 1
    
    # Same discovery rules as orth_to_coco.py: any depth under images/ and
    # anno/, any image suffix, and images without an annotation kept as
    # negative samples.
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

        # flags["1"] is true only for unhealthy teeth; healthy ones stay in the
        # dataset without annotations (negative samples).
        flags = (data or {}).get("flags") or {}
        is_unhealthy = flags.get("1") is True

        annotations: list[dict] = []
        if is_unhealthy:
            unhealthy_images += 1
            for shape in (data or {}).get("shapes") or []:
                label = str(shape.get("label", "")).strip()
                points = shape.get("points") or []

                if label not in CATEGORY_MAP:
                    print(f"Skipping unknown label '{label}' in {json_path}")
                    skipped_shapes += 1
                    continue

                if len(points) < 3:
                    skipped_shapes += 1
                    continue  # 不是多边形

                # LabelMe: [[x1, y1], [x2, y2]] -> COCO segmentation: [x1, y1, ...]
                np_points = np.asarray(points, dtype=float)
                flat_points = np_points.flatten().tolist()

                x_min = float(np.min(np_points[:, 0]))
                y_min = float(np.min(np_points[:, 1]))
                x_max = float(np.max(np_points[:, 0]))
                y_max = float(np.max(np_points[:, 1]))
                box_width = x_max - x_min
                box_height = y_max - y_min

                annotations.append(
                    {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": CATEGORY_MAP[label],
                        "segmentation": [flat_points],
                        "area": calculate_polygon_area(np_points),
                        "bbox": [x_min, y_min, box_width, box_height],
                        "iscrowd": 0,
                    }
                )
                annotation_id_counter += 1

        if drop_empty_images and not annotations:
            continue

        if not annotations:
            negative_images += 1

        width, height = image_size_from_file_or_json(image_path, data)
        if image_path is not None:
            file_name = image_path.resolve().relative_to(data_dir).as_posix()
        else:
            file_name = image_path_for_json(json_path, data_dir).relative_to(data_dir).as_posix()

        coco_output["images"].append(
            {
                "id": image_id_counter,
                "file_name": file_name,  # 训练时数据加载器据此寻找图片
                "height": height,
                "width": width,
                "date_captured": datetime.now().strftime("%Y-%m-%d"),
            }
        )
        coco_output["annotations"].extend(annotations)
        image_id_counter += 1

    # 4. 保存结果
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)

    print(f"Saved COCO JSON to {output_json}")
    print(f"Images: {len(coco_output['images'])}")
    print(f"Annotations: {len(coco_output['annotations'])}")
    print(
        f"Categories: {len(coco_output['categories'])} -> "
        f"{[category['name'] for category in coco_output['categories']]}"
    )
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


if __name__ == "__main__":
    args = parse_args()
    convert_to_coco(
        data_dir=args.data_dir,
        output_json=args.output_json,
        drop_empty_images=args.drop_empty_images,
        allow_missing_images=args.allow_missing_images,
    )