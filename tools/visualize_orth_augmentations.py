#!/usr/bin/env python3
"""Create per-sample montages for individual orth augmentation strategies.

Expects the orth raw-data layout used by datasets_coco/orth_to_coco.py:

    <data_dir>/<sample_id>/images/{D,F,L,R,U}.jpg
    <data_dir>/<sample_id>/anno/{D,F,L,R,U}.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform
from PIL import Image, ImageDraw, ImageFont

from datasets_coco.orth_augmentations import representative_augmentation_transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_VIEWS = ("D", "F", "L", "R", "U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize orth augmentations as two five-view montages."
    )
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_0616/orth_0616",
        type=Path,
        help="Root folder containing sample-id subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/orth_augmentation_visualizations",
        type=Path,
        help="Output directory for montage images.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional sample IDs. If omitted, two complete samples are selected.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--tile-width", default=320, type=int)
    parser.add_argument("--tile-height", default=240, type=int)
    return parser.parse_args()


def safe_name(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    return safe.strip("._") or "sample"


def normalize_box(points: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [float(point[0]) for point in pts]
    ys = [float(point[1]) for point in pts]
    return min(xs), min(ys), max(xs), max(ys)


def find_view_images(sample_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    images_dir = sample_dir / "images"
    if not images_dir.is_dir():
        return images
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            view = path.stem.strip().upper()
            if view in DEFAULT_VIEWS:
                images[view] = path
    return images


def find_complete_samples(data_dir: Path) -> dict[str, dict[str, Path]]:
    complete = {}
    for sample_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        images = find_view_images(sample_dir)
        if all(view in images for view in DEFAULT_VIEWS):
            complete[sample_dir.name] = images
    return complete


def choose_samples(
    complete: dict[str, dict[str, Path]],
    requested_ids: list[str] | None,
    seed: int,
) -> list[str]:
    if requested_ids:
        if len(requested_ids) != 2:
            raise ValueError("--sample-ids must contain exactly two IDs")
        missing = [sample_id for sample_id in requested_ids if sample_id not in complete]
        if missing:
            raise ValueError(
                f"Samples do not contain all D/F/L/R/U images: {missing}"
            )
        return list(requested_ids)

    candidates = sorted(complete)
    if len(candidates) < 2:
        raise ValueError("Need at least two samples with complete D/F/L/R/U images")
    return random.Random(seed).sample(candidates, 2)


def json_path_for_image(image_path: Path) -> Path:
    """Mirror <sample_id>/images/<view>.ext to <sample_id>/anno/<view>.json."""
    return image_path.parents[1] / "anno" / f"{image_path.stem}.json"


def load_boxes(image_path: Path) -> np.ndarray:
    json_path = json_path_for_image(image_path)
    if not json_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        shape_type = shape.get("shape_type", "rectangle")
        if len(points) < 2 or shape_type not in {"rectangle", "polygon"}:
            continue
        x1, y1, x2, y2 = normalize_box(points)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def apply_strategy(
    image: np.ndarray,
    boxes: np.ndarray,
    strategy: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy is None:
        return image.copy(), boxes.copy()

    if isinstance(strategy, Transform):
        transformed_image = strategy.apply_image(image.copy())
        transformed_boxes = strategy.apply_box(boxes.copy()) if boxes.size else boxes.copy()
        return transformed_image, transformed_boxes

    aug_input = T.AugInput(image.copy(), boxes=boxes.copy())
    strategy(aug_input)
    return aug_input.image, aug_input.boxes


def draw_boxes(image: np.ndarray, boxes: np.ndarray) -> Image.Image:
    pil_image = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    line_width = max(2, round(min(width, height) / 300))
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = min(max(x1, 0.0), width - 1.0)
        x2 = min(max(x2, 0.0), width - 1.0)
        y1 = min(max(y1, 0.0), height - 1.0)
        y2 = min(max(y2, 0.0), height - 1.0)
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2, y2), outline=(230, 45, 45), width=line_width)
    return pil_image


def render_tile(
    image: np.ndarray,
    boxes: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    annotated = draw_boxes(image, boxes)
    original_size = annotated.size
    caption_height = 24
    available_height = tile_height - caption_height
    scale = min(tile_width / annotated.width, available_height / annotated.height)
    resized = annotated.resize(
        (
            max(1, round(annotated.width * scale)),
            max(1, round(annotated.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )

    tile = Image.new("RGB", (tile_width, tile_height), "white")
    offset_x = (tile_width - resized.width) // 2
    offset_y = (available_height - resized.height) // 2
    tile.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    caption = f"{original_size[0]}x{original_size[1]}"
    draw.text((8, tile_height - 19), caption, fill=(30, 30, 30), font=font)
    return tile


def create_montage(
    sample_id: str,
    view_images: dict[str, Path],
    out_path: Path,
    tile_width: int,
    tile_height: int,
) -> None:
    strategies = [("Original", None)] + representative_augmentation_transforms()
    label_width = 190
    header_height = 42
    montage = Image.new(
        "RGB",
        (
            label_width + len(DEFAULT_VIEWS) * tile_width,
            header_height + len(strategies) * tile_height,
        ),
        (245, 246, 248),
    )
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()

    draw.text((10, 14), f"Sample {sample_id}", fill=(20, 20, 20), font=font)
    for column, view in enumerate(DEFAULT_VIEWS):
        x = label_width + column * tile_width + tile_width // 2 - 4
        draw.text((x, 14), view, fill=(20, 20, 20), font=font)

    source_data = {}
    for view in DEFAULT_VIEWS:
        image_path = view_images[view]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        source_data[view] = (image, load_boxes(image_path))

    for row, (strategy_name, strategy) in enumerate(strategies):
        y = header_height + row * tile_height
        draw.text((10, y + 12), strategy_name, fill=(20, 20, 20), font=font)
        for column, view in enumerate(DEFAULT_VIEWS):
            image, boxes = source_data[view]
            transformed_image, transformed_boxes = apply_strategy(
                image,
                boxes,
                strategy,
            )
            tile = render_tile(
                transformed_image,
                transformed_boxes,
                tile_width,
                tile_height,
            )
            montage.paste(tile, (label_width + column * tile_width, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    complete = find_complete_samples(data_dir)
    selected = choose_samples(complete, args.sample_ids, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in selected:
        output_path = args.out_dir / f"{safe_name(sample_id)}_augmentations.jpg"
        create_montage(
            sample_id,
            complete[sample_id],
            output_path,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
        print(f"Saved {output_path}")

    (args.out_dir / "selected_samples.json").write_text(
        json.dumps({"sample_ids": selected}, indent=2),
        encoding="utf-8",
    )
    print(f"Selected samples: {selected}")


if __name__ == "__main__":
    main()
