#!/usr/bin/env python
"""Load every config in the organised tree and assert the invariants that make
the SSL comparison meaningful.

Run from the repository root (all paths are resolved relative to this file):

    conda run -n intraoral python scripts/validate_ssl_configs.py

The script never trains. It only checks that

1. each YAML parses (including its ``_BASE_`` chain),
2. the SSL schema keys survive the merge,
3. the backbone builder name matches the SSL method's family,
4. every ``SSL_WEIGHTS`` file exists and the checkpoint key is the documented
   one,
5. detector-level shape constants are consistent (``SIZE_DIVISIBILITY``,
   ``FPN.IN_FEATURES``, ``MODEL.MASK_ON``).
"""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detectron2.config import get_cfg  # noqa: E402
from detectron2.modeling.backbone import BACKBONE_REGISTRY  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

from maskdino.config import add_maskdino_config  # noqa: E402

# --------------------------------------------------------------------------- #
# Expectations
# --------------------------------------------------------------------------- #
#: variant -> (SSL method, checkpoint key, expected weights basename)
SSL_EXPECTATIONS = {
    "simclr": {
        "method": "simclr",
        "checkpoint_key": "model.backbone",
        "weights_basename": "resnet50_checkpoint-050.pth",
        "builder": "build_resnet_ssl_backbone",
        "fpn_builder": "build_resnet_ssl_fpn_backbone",
        "size_divisibility": 32,
        "family_node": "RESNET_SSL",
    },
    "swinmae": {
        "method": "swinmae",
        "checkpoint_key": "model",
        "weights_basename": "swins_checkpoint-50.pth",
        "builder": "build_swin_ssl_backbone",
        "fpn_builder": "build_swin_ssl_fpn_backbone",
        "size_divisibility": 224,
        "family_node": "SWIN_SSL",
    },
    "dentmag_swin": {
        "method": "dentmag_swin",
        "checkpoint_key": "encoder_teacher",
        "weights_basename": "swins_highres_checkpoint-50.pth",
        "builder": "build_swin_ssl_backbone",
        "fpn_builder": "build_swin_ssl_fpn_backbone",
        "size_divisibility": 224,
        "family_node": "SWIN_SSL",
    },
    "dinov2": {
        "method": "dinov2",
        "checkpoint_key": "teacher",
        "weights_basename": "dinov2_vits_train249999_teacher_checkpoint.pth",
        "builder": "build_vit_ssl_backbone",
        "fpn_builder": "build_vit_ssl_fpn_backbone",
        "size_divisibility": 224,
        "family_node": "VIT_SSL",
    },
}
VARIANTS = ["no_ssl"] + list(SSL_EXPECTATIONS)

#: ``(detector, task, num_classes)``
COMBINATIONS = [
    ("maskrcnn", "caries", 9),
    ("maskrcnn", "orth", 5),
    ("maskdino", "caries", 9),
    ("maskdino", "orth", 5),
]

TREE = REPO_ROOT / "configs"


def config_paths(detector, task):
    folder = TREE / detector / task
    return [folder / f"{variant}.yaml" for variant in VARIANTS]


def load_cfg(path, detector, task, num_classes, swin_depths):
    """Reproduce the entry-point merge order without touching the dataset."""
    if detector == "maskdino":
        from maskdino_unify import _build_base_cfg
    else:
        from maskrcnn_unify import _build_base_cfg

    # Uses the exact base cfg (including ``set_new_allowed(True)``) that the
    # real entry points merge the config file into.
    cfg = _build_base_cfg()

    # Swin depth lives in MODEL.SWIN for the legacy backbones and MODEL.SWIN_SSL
    # for the unified ones; both nodes exist after add_maskdino_config. The
    # entry points inject it before merging the file, so do the same.
    cfg.MODEL.SWIN.DEPTHS = swin_depths
    cfg.merge_from_file(str(path))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = task == "caries"
    if detector == "maskdino":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
        # ``ORTH_CLASS_BALANCE.CLASS_WEIGHTS`` is deliberately *not* written in
        # the YAML: both entry points recompute it from the training
        # annotations (``compute_effective_class_weights``) because the weights
        # are per-dataset class frequencies. ``SetCriterion`` nevertheless
        # requires exactly ``num_classes`` entries whenever the balance is
        # enabled, so emulate what the entry point will inject.
        if task == "orth" and cfg.ORTH_CLASS_BALANCE.ENABLED:
            cfg.ORTH_CLASS_BALANCE.CLASS_WEIGHTS = [1.0] * num_classes
    return cfg


def check_yaml_fragment(path, detector):
    """Validate the raw YAML fragment before any base merging happens."""
    with open(path) as handle:
        raw = yaml.safe_load(handle)

    problems = []
    if raw is None:
        problems.append("empty file")
        return problems

    expected_base = path.parent / "_base.yaml"
    if raw.get("_BASE_") != "_base.yaml":
        problems.append(f"_BASE_ is {raw.get('_BASE_')!r}, expected '_base.yaml'")
    if not expected_base.exists():
        problems.append(f"missing sibling {expected_base.name}")

    variant = path.stem
    if variant in SSL_EXPECTATIONS:
        spec = SSL_EXPECTATIONS[variant]
        backbone = (raw.get("MODEL") or {}).get("BACKBONE") or {}
        for key, want in (
            ("SSL_METHOD", spec["method"]),
            ("SSL_CHECKPOINT_KEY", spec["checkpoint_key"]),
            ("USE_SSL_PRETRAINED", True),
            ("SSL_STRICT_LOAD", True),
        ):
            if backbone.get(key) != want:
                problems.append(f"MODEL.BACKBONE.{key} = {backbone.get(key)!r}, expected {want!r}")
        # Mask R-CNN wraps the encoder in the Detectron2 FPN; MaskDINO consumes
        # the raw res2-res5 features, so it uses the un-wrapped builder.
        want_builder = spec["fpn_builder"] if detector == "maskrcnn" else spec["builder"]
        if backbone.get("NAME") != want_builder:
            problems.append(
                f"MODEL.BACKBONE.NAME = {backbone.get('NAME')!r}, expected {want_builder!r}"
            )
        if spec["family_node"] not in (raw.get("MODEL") or {}):
            problems.append(f"missing MODEL.{spec['family_node']} block")
        if (raw.get("MODEL") or {}).get("WEIGHTS") != "":
            problems.append("SSL variants must set MODEL.WEIGHTS to ''")
    else:
        backbone = (raw.get("MODEL") or {}).get("BACKBONE") or {}
        if backbone.get("SSL_METHOD") != "none":
            problems.append("no_ssl must set MODEL.BACKBONE.SSL_METHOD: 'none'")
        if backbone.get("USE_SSL_PRETRAINED") is not False:
            problems.append("no_ssl must set MODEL.BACKBONE.USE_SSL_PRETRAINED: false")
        weights = (raw.get("MODEL") or {}).get("WEIGHTS")
        if not weights:
            problems.append("no_ssl must point MODEL.WEIGHTS at a supervised checkpoint")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swin-depths",
        type=int,
        nargs=4,
        default=[2, 2, 18, 2],
        help="Depths injected as MODEL.SWIN.DEPTHS, matching the entry points.",
    )
    args = parser.parse_args()

    failures = []
    checked = 0

    for detector, task, num_classes in COMBINATIONS:
        for path in config_paths(detector, task):
            tag = f"{path.relative_to(REPO_ROOT)}"
            if not path.exists():
                failures.append(f"{tag}: file is missing")
                continue

            for problem in check_yaml_fragment(path, detector):
                failures.append(f"{tag}: {problem}")

            try:
                cfg = load_cfg(path, detector, task, num_classes, args.swin_depths)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append(f"{tag}: failed to load ({type(exc).__name__}: {exc})")
                continue

            checked += 1
            variant = path.stem

            # -- backbone registered? ------------------------------------- #
            try:
                BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)
            except KeyError:
                failures.append(f"{tag}: backbone {cfg.MODEL.BACKBONE.NAME!r} is not registered")
                continue

            if variant == "no_ssl":
                if cfg.MODEL.BACKBONE.SSL_METHOD != "none":
                    failures.append(f"{tag}: SSL_METHOD {cfg.MODEL.BACKBONE.SSL_METHOD!r} != 'none'")
                if cfg.MODEL.BACKBONE.USE_SSL_PRETRAINED:
                    failures.append(f"{tag}: USE_SSL_PRETRAINED must be false")
                continue

            spec = SSL_EXPECTATIONS[variant]
            if cfg.MODEL.BACKBONE.SSL_METHOD != spec["method"]:
                failures.append(
                    f"{tag}: SSL_METHOD {cfg.MODEL.BACKBONE.SSL_METHOD!r} != {spec['method']!r}"
                )
            if not cfg.MODEL.BACKBONE.USE_SSL_PRETRAINED:
                failures.append(f"{tag}: USE_SSL_PRETRAINED must be true")
            if not cfg.MODEL.BACKBONE.SSL_STRICT_LOAD:
                failures.append(f"{tag}: SSL_STRICT_LOAD must be true")

            # -- weights file exists -------------------------------------- #
            weights = Path(cfg.MODEL.BACKBONE.SSL_WEIGHTS)
            if not weights.is_absolute():
                weights = REPO_ROOT / weights
            if not weights.exists():
                failures.append(f"{tag}: SSL_WEIGHTS does not exist: {weights}")
            elif weights.name != spec["weights_basename"]:
                failures.append(
                    f"{tag}: SSL_WEIGHTS basename {weights.name!r} != {spec['weights_basename']!r}"
                )

            # -- expected builder name ------------------------------------ #
            expected_builder = spec["fpn_builder"] if "fpn" in cfg.MODEL.BACKBONE.NAME else spec["builder"]
            if cfg.MODEL.BACKBONE.NAME != expected_builder:
                failures.append(
                    f"{tag}: backbone {cfg.MODEL.BACKBONE.NAME!r} != {expected_builder!r}"
                )

            # -- size divisibility ---------------------------------------- #
            if detector == "maskdino":
                got = cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY
                want = spec["size_divisibility"]
                if got != want:
                    failures.append(f"{tag}: SIZE_DIVISIBILITY {got} != {want}")
            else:
                got = list(cfg.MODEL.FPN.IN_FEATURES)
                if got != ["res2", "res3", "res4", "res5"]:
                    failures.append(f"{tag}: FPN.IN_FEATURES {got} != res2-res5")

    print(f"checked {checked} configs")
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for problem in failures:
            print(f"  - {problem}")
        return 1

    print("All config invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
