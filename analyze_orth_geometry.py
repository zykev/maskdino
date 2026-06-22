#!/usr/bin/env python3
"""Analyze orth image resolution and LabelMe bounding-box geometry.

The script scans the raw orth folder layout:

    .datasets/intraoral_anno/orth_0616/orth_0616/
      23025/
        D.png
        F.png
        L.png
        R.png
        U.json
        U.png

It writes CSV summaries plus plots that are useful for choosing Detectron2
input resize settings and anchor sizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from datasets_coco.orth_to_coco import CATEGORY_INFO_BY_NAME


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NICE_ANCHOR_SIZES = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


@dataclass
class ImageStat:
    sample_id: str
    view: str
    image_path: str
    width: int
    height: int
    short_side: int
    long_side: int
    aspect_ratio: float
    area: int
    has_json: bool


@dataclass
class BBoxStat:
    sample_id: str
    view: str
    json_path: str
    image_path: str
    label: str
    supercategory: str
    image_width: int
    image_height: int
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    sqrt_area: float
    area: float
    area_fraction: float
    rel_width: float
    rel_height: float
    rel_sqrt_area: float
    aspect_ratio: float


@dataclass
class SkippedShape:
    json_path: str
    shape_index: int
    label: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze orth image sizes and LabelMe bbox sizes."
    )
    parser.add_argument(
        "--data-dir",
        default=".datasets/intraoral_anno/orth_0616/orth_0616",
        type=Path,
        help="Root folder containing sample subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/orth_geometry_stats",
        type=Path,
        help="Output folder for CSV files, plots, and summaries.",
    )
    parser.add_argument(
        "--target-short-sides",
        nargs="*",
        default=[512, 640, 800, 1024],
        type=int,
        help="Candidate INPUT.MIN_SIZE values used to estimate resized bbox sizes.",
    )
    parser.add_argument(
        "--max-size",
        default=1333,
        type=int,
        help="Candidate INPUT.MAX_SIZE used when estimating resized bbox sizes.",
    )
    parser.add_argument("--bins", default=40, type=int, help="Histogram bin count.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    return safe.strip("._") or "unknown"


def sample_id_from_path(path: Path, data_dir: Path) -> str:
    try:
        rel = path.relative_to(data_dir)
    except ValueError:
        return path.parent.name
    return rel.parts[0] if rel.parts else path.parent.name


def view_from_path(path: Path) -> str:
    return path.stem


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


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
    return min(xs), min(ys), max(xs), max(ys)


def known_supercategory(label: str) -> str:
    info = CATEGORY_INFO_BY_NAME.get(label)
    if info:
        return str(info["supercategory"])
    if label in {item["supercategory"] for item in CATEGORY_INFO_BY_NAME.values()}:
        return label
    return "unknown"


def valid_shape(shape: dict) -> tuple[bool, str]:
    label = str(shape.get("label", "")).strip()
    points = shape.get("points") or []
    shape_type = shape.get("shape_type", "rectangle")
    if not label:
        return False, "empty_label"
    if len(points) < 2:
        return False, "too_few_points"
    if shape_type not in {"rectangle", "polygon"}:
        return False, f"unsupported_shape_type:{shape_type}"
    return True, ""


def scan_images(data_dir: Path) -> dict[Path, ImageStat]:
    stats: dict[Path, ImageStat] = {}
    json_stems = {path.with_suffix("").resolve() for path in data_dir.rglob("*.json")}

    for image_path in sorted(data_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            width, height = image_size(image_path)
        except Exception as exc:
            print(f"Skip unreadable image {image_path}: {exc}")
            continue
        short_side = min(width, height)
        long_side = max(width, height)
        stats[image_path.resolve()] = ImageStat(
            sample_id=sample_id_from_path(image_path, data_dir),
            view=view_from_path(image_path),
            image_path=str(image_path),
            width=width,
            height=height,
            short_side=short_side,
            long_side=long_side,
            aspect_ratio=width / height if height else 0.0,
            area=width * height,
            has_json=image_path.with_suffix("").resolve() in json_stems,
        )
    return stats


def scan_bboxes(
    data_dir: Path,
    image_stats: dict[Path, ImageStat],
) -> tuple[list[BBoxStat], list[SkippedShape]]:
    boxes: list[BBoxStat] = []
    skipped: list[SkippedShape] = []

    for json_path in sorted(data_dir.rglob("*.json")):
        data = load_json(json_path)
        if {"images", "annotations", "categories"}.issubset(data):
            continue
        image_path = resolve_image_path(json_path, data)
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if image_path and image_path.resolve() in image_stats:
            image_stat = image_stats[image_path.resolve()]
            width = image_stat.width
            height = image_stat.height
        elif width <= 0 or height <= 0:
            skipped.append(
                SkippedShape(str(json_path), -1, "", "missing_image_size")
            )
            continue

        sample_id = sample_id_from_path(json_path, data_dir)
        view = view_from_path(image_path) if image_path else view_from_path(json_path)
        image_path_text = str(image_path) if image_path else ""

        for shape_index, shape in enumerate(data.get("shapes", [])):
            label = str(shape.get("label", "")).strip()
            is_valid, reason = valid_shape(shape)
            if not is_valid:
                skipped.append(SkippedShape(str(json_path), shape_index, label, reason))
                continue

            x1, y1, x2, y2 = normalize_box(shape["points"])
            x1_clipped = min(max(x1, 0.0), float(width))
            y1_clipped = min(max(y1, 0.0), float(height))
            x2_clipped = min(max(x2, 0.0), float(width))
            y2_clipped = min(max(y2, 0.0), float(height))
            box_width = max(0.0, x2_clipped - x1_clipped)
            box_height = max(0.0, y2_clipped - y1_clipped)
            if box_width <= 0.0 or box_height <= 0.0:
                skipped.append(SkippedShape(str(json_path), shape_index, label, "zero_area_box"))
                continue

            area = box_width * box_height
            boxes.append(
                BBoxStat(
                    sample_id=sample_id,
                    view=view,
                    json_path=str(json_path),
                    image_path=image_path_text,
                    label=label,
                    supercategory=known_supercategory(label),
                    image_width=width,
                    image_height=height,
                    x1=x1_clipped,
                    y1=y1_clipped,
                    x2=x2_clipped,
                    y2=y2_clipped,
                    width=box_width,
                    height=box_height,
                    sqrt_area=math.sqrt(area),
                    area=area,
                    area_fraction=area / (width * height) if width and height else 0.0,
                    rel_width=box_width / width if width else 0.0,
                    rel_height=box_height / height if height else 0.0,
                    rel_sqrt_area=math.sqrt(area / (width * height)) if width and height else 0.0,
                    aspect_ratio=box_width / box_height if box_height else 0.0,
                )
            )

    return boxes, skipped


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def values(rows: list[object], attr: str) -> np.ndarray:
    return np.array([float(getattr(row, attr)) for row in rows], dtype=np.float64)


def percentile_dict(array: np.ndarray) -> dict[str, float]:
    if array.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES}


def nearest_nice_sizes(raw_sizes: Iterable[float]) -> list[int]:
    chosen = []
    for size in raw_sizes:
        if not np.isfinite(size) or size <= 0:
            continue
        nearest = int(NICE_ANCHOR_SIZES[np.argmin(np.abs(NICE_ANCHOR_SIZES - size))])
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def resize_scale(width: float, height: float, target_short_side: int, max_size: int) -> float:
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side <= 0 or long_side <= 0:
        return 1.0
    scale = target_short_side / short_side
    if long_side * scale > max_size:
        scale = max_size / long_side
    return scale


def resized_bbox_sqrt_areas(boxes: list[BBoxStat], target_short_side: int, max_size: int) -> np.ndarray:
    resized = []
    for box in boxes:
        scale = resize_scale(box.image_width, box.image_height, target_short_side, max_size)
        resized.append(box.sqrt_area * scale)
    return np.array(resized, dtype=np.float64)


def summarize(
    image_rows: list[ImageStat],
    bbox_rows: list[BBoxStat],
    skipped_rows: list[SkippedShape],
    target_short_sides: list[int],
    max_size: int,
) -> dict:
    image_short = values(image_rows, "short_side")
    image_long = values(image_rows, "long_side")
    bbox_sqrt = values(bbox_rows, "sqrt_area")
    bbox_width = values(bbox_rows, "width")
    bbox_height = values(bbox_rows, "height")
    bbox_area_fraction = values(bbox_rows, "area_fraction")

    resized = {}
    for short_side in target_short_sides:
        arr = resized_bbox_sqrt_areas(bbox_rows, short_side, max_size)
        selected = [np.percentile(arr, p) for p in [10, 25, 50, 75, 90]] if arr.size else []
        resized[str(short_side)] = {
            "sqrt_area_percentiles": percentile_dict(arr),
            "suggested_anchor_sizes": nearest_nice_sizes(selected),
        }

    return {
        "num_images": len(image_rows),
        "num_images_with_json_stem": sum(row.has_json for row in image_rows),
        "num_json_boxes": len(bbox_rows),
        "num_skipped_shapes": len(skipped_rows),
        "image_short_side": percentile_dict(image_short),
        "image_long_side": percentile_dict(image_long),
        "bbox_width": percentile_dict(bbox_width),
        "bbox_height": percentile_dict(bbox_height),
        "bbox_sqrt_area": percentile_dict(bbox_sqrt),
        "bbox_area_fraction": percentile_dict(bbox_area_fraction),
        "bbox_counts_by_label": dict(Counter(row.label for row in bbox_rows).most_common()),
        "bbox_counts_by_supercategory": dict(
            Counter(row.supercategory for row in bbox_rows).most_common()
        ),
        "resized_bbox_estimates": resized,
    }


def markdown_table_percentiles(title: str, stats: dict[str, float]) -> list[str]:
    if not stats:
        return [f"### {title}", "", "No data.", ""]
    keys = [f"p{p}" for p in PERCENTILES]
    lines = [f"### {title}", "", "| percentile | value |", "|---:|---:|"]
    for key in keys:
        lines.append(f"| {key} | {stats[key]:.2f} |")
    lines.append("")
    return lines


def write_summary_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Orth Geometry Statistics",
        "",
        f"- Images: {summary['num_images']}",
        f"- Images with matching JSON stem: {summary['num_images_with_json_stem']}",
        f"- Valid annotation boxes: {summary['num_json_boxes']}",
        f"- Skipped shapes: {summary['num_skipped_shapes']}",
        "",
    ]
    lines += markdown_table_percentiles("Image short side", summary["image_short_side"])
    lines += markdown_table_percentiles("Image long side", summary["image_long_side"])
    lines += markdown_table_percentiles("BBox width (original pixels)", summary["bbox_width"])
    lines += markdown_table_percentiles("BBox height (original pixels)", summary["bbox_height"])
    lines += markdown_table_percentiles("BBox sqrt area (original pixels)", summary["bbox_sqrt_area"])
    lines += markdown_table_percentiles("BBox area fraction", summary["bbox_area_fraction"])

    lines += ["## Suggested Anchor Sizes After Resize", ""]
    lines += [
        "Each row estimates bbox sqrt-area after Detectron2-style resizing "
        "with the given `MIN_SIZE` and configured `MAX_SIZE`.",
        "",
        "| MIN_SIZE | suggested anchor sizes |",
        "|---:|---|",
    ]
    for short_side, data in summary["resized_bbox_estimates"].items():
        anchors = ", ".join(str(item) for item in data["suggested_anchor_sizes"])
        lines.append(f"| {short_side} | [{anchors}] |")
    lines.append("")

    lines += ["## Counts By Supercategory", "", "| supercategory | boxes |", "|---|---:|"]
    for key, count in summary["bbox_counts_by_supercategory"].items():
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines += ["## Counts By Label", "", "| label | boxes |", "|---|---:|"]
    for key, count in summary["bbox_counts_by_label"].items():
        lines.append(f"| {key} | {count} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_hist_grid(path: Path, arrays: list[tuple[str, np.ndarray]], bins: int) -> None:
    plt = import_pyplot()
    cols = 2
    rows = math.ceil(len(arrays) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (title, array) in zip(axes, arrays):
        clean = array[np.isfinite(array)]
        ax.hist(clean, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
        if clean.size:
            ax.axvline(np.median(clean), color="#E45756", linestyle="--", label="median")
            ax.legend()
        ax.set_title(title)
        ax.set_ylabel("count")
    for ax in axes[len(arrays):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_counts_bar(path: Path, counts: Counter, title: str) -> None:
    plt = import_pyplot()
    items = counts.most_common()
    labels = [item[0] for item in items]
    values_ = [item[1] for item in items]
    fig_width = max(10, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(range(len(labels)), values_, color="#59A14F", edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("box count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    for index, value in enumerate(values_):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_bbox_scatter(path: Path, bbox_rows: list[BBoxStat]) -> None:
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    if bbox_rows:
        x = values(bbox_rows, "width")
        y = values(bbox_rows, "height")
        categories = sorted({row.supercategory for row in bbox_rows})
        cmap = plt.get_cmap("tab20")
        for idx, category in enumerate(categories):
            mask = np.array([row.supercategory == category for row in bbox_rows])
            ax.scatter(
                x[mask],
                y[mask],
                s=18,
                alpha=0.65,
                label=category,
                color=cmap(idx % 20),
            )
        max_side = max(float(np.max(x)), float(np.max(y)), 1.0)
        for anchor in NICE_ANCHOR_SIZES:
            if anchor <= max_side * 1.05:
                ax.axvline(anchor, color="gray", linewidth=0.6, alpha=0.25)
                ax.axhline(anchor, color="gray", linewidth=0.6, alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title("BBox width vs height (original pixels)")
    ax.set_xlabel("bbox width")
    ax.set_ylabel("bbox height")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_image_scatter(path: Path, image_rows: list[ImageStat]) -> None:
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 7))
    if image_rows:
        ax.scatter(values(image_rows, "width"), values(image_rows, "height"), s=14, alpha=0.5)
    ax.set_title("Image resolution scatter")
    ax.set_xlabel("image width")
    ax.set_ylabel("image height")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_supercategory_boxplot(path: Path, bbox_rows: list[BBoxStat]) -> None:
    plt = import_pyplot()
    grouped: dict[str, list[float]] = {}
    for row in bbox_rows:
        grouped.setdefault(row.supercategory, []).append(row.sqrt_area)
    labels = []
    data = []
    for label, items in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        labels.append(label)
        data.append(items)
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(labels)), 6))
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("BBox sqrt area by supercategory")
    ax.set_ylabel("sqrt(width * height)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_plots(
    out_dir: Path,
    image_rows: list[ImageStat],
    bbox_rows: list[BBoxStat],
    bins: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_hist_grid(
        plot_dir / "image_side_histograms.png",
        [
            ("image width", values(image_rows, "width")),
            ("image height", values(image_rows, "height")),
            ("image short side", values(image_rows, "short_side")),
            ("image long side", values(image_rows, "long_side")),
        ],
        bins,
    )
    save_hist_grid(
        plot_dir / "image_area_aspect_histograms.png",
        [
            ("image area", values(image_rows, "area")),
            ("image aspect ratio (w / h)", values(image_rows, "aspect_ratio")),
        ],
        bins,
    )
    save_hist_grid(
        plot_dir / "bbox_size_histograms.png",
        [
            ("bbox width", values(bbox_rows, "width")),
            ("bbox height", values(bbox_rows, "height")),
            ("bbox sqrt area", values(bbox_rows, "sqrt_area")),
            ("bbox area fraction", values(bbox_rows, "area_fraction")),
        ],
        bins,
    )
    save_hist_grid(
        plot_dir / "bbox_relative_histograms.png",
        [
            ("relative bbox width", values(bbox_rows, "rel_width")),
            ("relative bbox height", values(bbox_rows, "rel_height")),
            ("relative bbox sqrt area", values(bbox_rows, "rel_sqrt_area")),
            ("bbox aspect ratio (w / h)", values(bbox_rows, "aspect_ratio")),
        ],
        bins,
    )
    save_image_scatter(plot_dir / "image_resolution_scatter.png", image_rows)
    save_bbox_scatter(plot_dir / "bbox_width_height_scatter.png", bbox_rows)
    save_supercategory_boxplot(plot_dir / "bbox_sqrt_area_by_supercategory.png", bbox_rows)
    save_counts_bar(
        plot_dir / "bbox_counts_by_label.png",
        Counter(row.label for row in bbox_rows),
        "BBox counts by original label",
    )
    save_counts_bar(
        plot_dir / "bbox_counts_by_supercategory.png",
        Counter(row.supercategory for row in bbox_rows),
        "BBox counts by supercategory",
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_stats = scan_images(data_dir)
    image_rows = list(image_stats.values())
    bbox_rows, skipped_rows = scan_bboxes(data_dir, image_stats)

    write_csv(out_dir / "image_stats.csv", image_rows)
    write_csv(out_dir / "bbox_stats.csv", bbox_rows)
    write_csv(out_dir / "skipped_shapes.csv", skipped_rows)

    summary = summarize(
        image_rows,
        bbox_rows,
        skipped_rows,
        args.target_short_sides,
        args.max_size,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_markdown(out_dir / "summary.md", summary)
    write_plots(out_dir, image_rows, bbox_rows, args.bins)

    print(f"Scanned images: {len(image_rows)}")
    print(f"Valid annotation boxes: {len(bbox_rows)}")
    print(f"Skipped shapes: {len(skipped_rows)}")
    print(f"Wrote outputs to: {out_dir}")
    print(f"Open summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
