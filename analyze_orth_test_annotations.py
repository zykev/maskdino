#!/usr/bin/env python3
"""Visualize LabelMe bounding-box annotations grouped by class.

The script scans an orth_test-style folder, groups rectangle annotations by
label, saves one or more montage images per class, and writes class statistics.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (0, 119, 182),
    (106, 153, 78),
    (214, 40, 40),
    (123, 44, 191),
]


@dataclass(frozen=True)
class Box:
    label: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Record:
    json_path: Path
    image_path: Path | None
    image_size: tuple[int, int]
    boxes: tuple[Box, ...]

    @property
    def sample_id(self) -> str:
        if self.image_path:
            return str(self.image_path)
        return str(self.json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize LabelMe rectangle annotations by class."
    )
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_test",
        type=Path,
        help="Folder containing images and LabelMe JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default="orth_test_annotation_analysis",
        type=Path,
        help="Output folder for montages and statistics.",
    )
    parser.add_argument(
        "--tile-size",
        default=360,
        type=int,
        help="Maximum image side length in each montage tile.",
    )
    parser.add_argument(
        "--cols",
        default=4,
        type=int,
        help="Number of columns in each montage.",
    )
    parser.add_argument(
        "--max-tiles-per-page",
        default=32,
        type=int,
        help="Split a class into multiple montage pages after this many images.",
    )
    parser.add_argument(
        "--draw-other-labels",
        action="store_true",
        help="Also draw boxes from other labels in gray on each class montage.",
    )
    return parser.parse_args()


def safe_name(label: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", label.strip())
    return name or "empty_label"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_box(points: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


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


def load_image(record: Record, data: dict | None = None) -> Image.Image:
    if record.image_path and record.image_path.exists():
        return Image.open(record.image_path).convert("RGB")

    if data is None:
        data = load_json(record.json_path)
    image_data = data.get("imageData")
    if not image_data:
        raise FileNotFoundError(f"Cannot find image for {record.json_path}")
    raw = base64.b64decode(image_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def scan_records(data_dir: Path) -> list[Record]:
    records: list[Record] = []
    for json_path in sorted(data_dir.rglob("*.json")):
        data = load_json(json_path)
        boxes: list[Box] = []
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            points = shape.get("points") or []
            if not label or len(points) < 2:
                continue
            if shape.get("shape_type", "rectangle") not in {"rectangle", "polygon"}:
                continue
            boxes.append(Box(label=label, xyxy=normalize_box(points)))

        if not boxes:
            continue

        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        image_path = resolve_image_path(json_path, data)
        records.append(
            Record(
                json_path=json_path,
                image_path=image_path,
                image_size=(width, height),
                boxes=tuple(boxes),
            )
        )
    return records


def stats_by_label(records: list[Record]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    labels = sorted({box.label for record in records for box in record.boxes})
    for label in labels:
        sample_ids = {
            record.sample_id
            for record in records
            if any(box.label == label for box in record.boxes)
        }
        box_count = sum(
            1 for record in records for box in record.boxes if box.label == label
        )
        stats[label] = {"samples": len(sample_ids), "boxes": box_count}
    return stats


def draw_record_tile(
    record: Record,
    target_label: str,
    tile_size: int,
    draw_other_labels: bool,
    label_color: tuple[int, int, int],
) -> Image.Image:
    image = load_image(record)
    width, height = image.size
    scale = min(tile_size / width, tile_size / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (tile_size, tile_size + 34), (248, 249, 250))
    offset = ((tile_size - new_size[0]) // 2, (tile_size - new_size[1]) // 2)
    canvas.paste(image, offset)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box in record.boxes:
        is_target = box.label == target_label
        if not is_target and not draw_other_labels:
            continue
        color = label_color if is_target else (155, 155, 155)
        line_width = 5 if is_target else 2
        x1, y1, x2, y2 = box.xyxy
        scaled = (
            offset[0] + x1 * scale,
            offset[1] + y1 * scale,
            offset[0] + x2 * scale,
            offset[1] + y2 * scale,
        )
        draw.rectangle(scaled, outline=color, width=line_width)
        text_pos = (scaled[0] + 3, max(offset[1], scaled[1] - 14))
        draw.text(text_pos, box.label, fill=color, font=font)

    case_name = record.json_path.parent.name
    view_name = record.json_path.stem
    caption = f"{case_name}/{view_name}"
    draw.rectangle((0, tile_size, tile_size, tile_size + 34), fill=(248, 249, 250))
    draw.text((8, tile_size + 10), caption[:52], fill=(30, 30, 30), font=font)
    return canvas


def save_montages(
    records: list[Record],
    stats: dict[str, dict[str, int]],
    out_dir: Path,
    tile_size: int,
    cols: int,
    max_tiles_per_page: int,
    draw_other_labels: bool,
) -> dict[str, list[Path]]:
    montage_dir = out_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[Path]] = defaultdict(list)
    labels = sorted(stats)

    for label_index, label in enumerate(labels):
        class_records = [
            record for record in records if any(box.label == label for box in record.boxes)
        ]
        color = COLORS[label_index % len(COLORS)]
        pages = [
            class_records[i : i + max_tiles_per_page]
            for i in range(0, len(class_records), max_tiles_per_page)
        ]
        for page_idx, page_records in enumerate(pages, start=1):
            rows = math.ceil(len(page_records) / cols)
            header_h = 54
            tile_h = tile_size + 34
            canvas = Image.new(
                "RGB", (cols * tile_size, header_h + rows * tile_h), (255, 255, 255)
            )
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            header = (
                f"Class {label} | samples: {stats[label]['samples']} | "
                f"boxes: {stats[label]['boxes']} | page {page_idx}/{len(pages)}"
            )
            draw.text((12, 18), header, fill=(20, 20, 20), font=font)

            for i, record in enumerate(page_records):
                tile = draw_record_tile(
                    record, label, tile_size, draw_other_labels, color
                )
                x = (i % cols) * tile_size
                y = header_h + (i // cols) * tile_h
                canvas.paste(tile, (x, y))

            output_path = montage_dir / f"class_{safe_name(label)}_page_{page_idx:03d}.jpg"
            canvas.save(output_path, quality=92)
            saved[label].append(output_path)
    return saved


def write_stats(out_dir: Path, stats: dict[str, dict[str, int]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "class_statistics.csv"
    json_path = out_dir / "class_statistics.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "samples", "boxes"])
        writer.writeheader()
        for label, row in sorted(stats.items()):
            writer.writerow({"label": label, **row})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    records = scan_records(args.data_dir)
    if not records:
        raise SystemExit(f"No valid JSON annotations found in {args.data_dir}")

    stats = stats_by_label(records)
    write_stats(args.out_dir, stats)
    saved = save_montages(
        records=records,
        stats=stats,
        out_dir=args.out_dir,
        tile_size=args.tile_size,
        cols=args.cols,
        max_tiles_per_page=args.max_tiles_per_page,
        draw_other_labels=args.draw_other_labels,
    )

    print(f"Scanned JSON files with boxes: {len(records)}")
    print(f"Classes: {len(stats)}")
    print(f"Statistics saved to: {args.out_dir / 'class_statistics.csv'}")
    print(f"Montages saved to: {args.out_dir / 'montages'}")
    print()
    print("label,samples,boxes,montage_pages")
    for label, row in sorted(stats.items()):
        print(f"{label},{row['samples']},{row['boxes']},{len(saved[label])}")


if __name__ == "__main__":
    main()
