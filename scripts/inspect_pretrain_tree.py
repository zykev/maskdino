#!/usr/bin/env python
"""Print the top-level structure of each SSL pretrain checkpoint.

Usage:
    conda run -n intraoral python scripts/inspect_pretrain_tree.py
    conda run -n intraoral python scripts/inspect_pretrain_tree.py --segment model.backbone
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from maskdino.modeling.backbone.ssl_common import (  # noqa: E402
    extract_state_dict,
    load_checkpoint,
)

PRETRAIN_DIR = REPO_ROOT / ".checkpoints" / "pretrain"
DEFAULT_FILES = [
    PRETRAIN_DIR / "simclr" / "resnet50_checkpoint-050.pth",
    PRETRAIN_DIR / "swinmae" / "swins_checkpoint-50.pth",
    PRETRAIN_DIR / "dentmap" / "swins_highres_checkpoint-50.pth",
    PRETRAIN_DIR / "dinov2" / "dinov2_vits_train249999_teacher_checkpoint.pth",
]


def describe(tree, depth, max_depth, prefix=""):
    """Yield ``prefix -> type`` lines up to ``max_depth``."""
    if depth >= max_depth:
        return
    if isinstance(tree, dict):
        entries = sorted(tree.items())
    else:
        items = list(tree)
        entries = [(str(i), items[i]) for i in range(len(items))]
    for key, value in entries:
        if isinstance(value, dict):
            print(f"{prefix}{key}/  ({len(value)} entries)")
            describe(value, depth + 1, max_depth, prefix=f"{prefix}{key}/")
        elif torch.is_tensor(value):
            print(f"{prefix}{key}  tensor{tuple(value.shape)}")
        else:
            print(f"{prefix}{key}  {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", default=None, help="Checkpoint to inspect.")
    parser.add_argument("--depth", type=int, default=2, help="Recursion depth to print.")
    parser.add_argument(
        "--segment",
        default=None,
        help="Dotted path to descend into before printing (e.g. 'model.backbone').",
    )
    args = parser.parse_args()

    files = [Path(f) for f in args.file] if args.file else DEFAULT_FILES
    for path in files:
        path = path if path.is_absolute() else (Path.cwd() / path)
        try:
            label = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            label = path
        print(f"===== {label} =====")
        checkpoint = load_checkpoint(path)
        print(f"top-level keys: {list(checkpoint.keys())}")
        tree, selected = extract_state_dict(checkpoint)
        print(f"auto-detected state dict key: {selected!r} ({len(tree)} entries)")
        if args.segment:
            for segment in args.segment.split("."):
                tree = tree[segment]
            print(f"-- {args.segment} --")
        describe(tree, 0, args.depth)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
