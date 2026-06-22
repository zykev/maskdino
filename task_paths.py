#!/usr/bin/env python3
"""Shared dataset path resolution for caries and orth tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT_DIRS = {
    "caries": Path(".datasets/intraoral_anno/single_ch_0225"),
    "orth": Path(".datasets/intraoral_anno/orth_test"),
}


def add_input_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input_dir",
        default=None,
        type=Path,
        help=(
            "Dataset root. For caries, expects single_tooth/ plus "
            "caries_sample_dataset_{train,test}.json. For orth, expects a "
            "subdirectory matching the root name plus orth_detection_{train,test}.json."
        ),
    )


def resolve_task_paths(args: argparse.Namespace) -> argparse.Namespace:
    input_dir = getattr(args, "input_dir", None) or DEFAULT_INPUT_DIRS[args.task]
    args.input_dir = input_dir

    if args.task == "caries":
        args.data_dir = input_dir / "single_tooth"
        args.train_json = input_dir / "caries_sample_dataset_train.json"
        args.test_json = input_dir / "caries_sample_dataset_test.json"
    else:
        args.data_dir = input_dir / input_dir.name
        args.train_json = input_dir / "orth_detection_train.json"
        args.test_json = input_dir / "orth_detection_test.json"

    return args
