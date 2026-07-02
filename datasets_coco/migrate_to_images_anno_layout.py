#!/usr/bin/env python3
"""One-off migration: split a co-located raw dataset into images/ + anno/.

Old layout (any depth of subfolders under a sample_id is fine, e.g. orth's
flat ``<sample_id>/D.jpg`` or caries' ``<sample_id>/<view>/<tooth_id>.png``):

    <root>/<sample_id>/<...>/<name>.jpg
    <root>/<sample_id>/<...>/<name>.json

New layout, matching what orth_to_coco.py / datasets_to_coco.py now expect:

    <root>/<sample_id>/images/<...>/<name>.jpg
    <root>/<sample_id>/anno/<...>/<name>.json

Run with --dry-run first to preview what would move. Re-running is safe:
files already under images/ or anno/ are left alone.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a co-located raw dataset into sibling images/ and anno/ folders."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help=(
            "Root folder containing sample_id subfolders, e.g. "
            ".datasets/intraoral_anno/orth_0616/orth_0616 or "
            ".datasets/intraoral_anno/single_ch_0225/single_tooth"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="move (default, fast rename within the same filesystem) or copy the originals.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without touching files.")
    return parser.parse_args()


def plan_migration(root: Path) -> tuple[list[tuple[Path, Path, str]], list[Path]]:
    """Return (moves, skipped) where moves is a list of (src, dest, kind)."""
    moves: list[tuple[Path, Path, str]] = []
    skipped: list[Path] = []

    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for sample_dir in sample_dirs:
        # Materialize the file list before planning any moves so that later
        # mutation of the tree (creating images/anno subfolders) can't affect
        # this walk.
        files = sorted(p for p in sample_dir.rglob("*") if p.is_file())
        for path in files:
            rel_within_sample = path.relative_to(sample_dir)
            if rel_within_sample.parts[0] in ("images", "anno"):
                continue  # already migrated

            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                marker = "images"
            elif suffix == ".json":
                marker = "anno"
            else:
                skipped.append(path)
                continue

            dest = sample_dir / marker / rel_within_sample
            moves.append((path, dest, marker))

    return moves, skipped


def apply_migration(moves: list[tuple[Path, Path, str]], mode: str, dry_run: bool) -> None:
    for src, dest, _marker in moves:
        if dest.exists():
            raise FileExistsError(f"Destination already exists, refusing to overwrite: {dest}")
        if dry_run:
            print(f"[dry-run] {mode} {src} -> {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == "move":
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--root does not exist or is not a directory: {root}")

    moves, skipped = plan_migration(root)
    apply_migration(moves, args.mode, args.dry_run)

    image_count = sum(1 for _, _, marker in moves if marker == "images")
    anno_count = sum(1 for _, _, marker in moves if marker == "anno")
    verb = "Would move" if args.dry_run else ("Moved" if args.mode == "move" else "Copied")
    print(f"{verb} {image_count} images and {anno_count} annotation files under {root}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with an unrecognized extension:")
        for path in skipped[:20]:
            print(f"  {path}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
