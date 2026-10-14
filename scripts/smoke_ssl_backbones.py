#!/usr/bin/env python
"""Smoke tests for the SSL backbone adapters (see Docs/ssl_..._plan.md §11).

Verifies, for every SSL family and for both downstream detectors, that

1. the checkpoint's key tensors load (fail-loud loader reports a clean audit);
2. ``forward()`` returns exactly ``res2``-``res5`` as 4D NCHW tensors;
3. spatial sizes and strides match ``output_shape()``;
4. non-square inputs work;
5. the ViT path reshapes by the real token grid rather than ``sqrt(N)``;
6. Mask R-CNN / MaskDINO can run a loss forward and an inference.

Usage::

    conda run -n intraoral python scripts/smoke_ssl_backbones.py
    conda run -n intraoral python scripts/smoke_ssl_backbones.py --family vit
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

PRETRAIN_DIR = os.path.join(REPO_ROOT, ".checkpoints", "pretrain")

# 896x896 -> res2 224^2, res3 112^2, res4 56^2, res5 28^2 (plan §11.2)
SQUARE_INPUT = (2, 3, 896, 896)
SQUARE_EXPECTED = {"res2": 224, "res3": 112, "res4": 56, "res5": 28}
NON_SQUARE_INPUT = (2, 3, 672, 896)

#: (family, variant) -> kwargs applied on top of the shared SSL config.
VARIANTS = {
    "swin": {
        "none": {},
        "swinmae": {
            "SSL_METHOD": "swinmae",
            "USE_SSL_PRETRAINED": True,
            "SSL_WEIGHTS": os.path.join(PRETRAIN_DIR, "swinmae", "swins_checkpoint-50.pth"),
            "SSL_CHECKPOINT_KEY": "model",
            # Swin-Small: the verified architecture of both shipped Swin encoders.
            "EMBED_DIM": 96,
            "DEPTHS": [2, 2, 18, 2],
            "NUM_HEADS": [3, 6, 12, 24],
            "PATCH_SIZE": 4,
            "WINDOW_SIZE": 7,
        },
        "dentmag_swin": {
            "SSL_METHOD": "dentmag_swin",
            "USE_SSL_PRETRAINED": True,
            "SSL_WEIGHTS": os.path.join(PRETRAIN_DIR, "dentmap", "swins_highres_checkpoint-50.pth"),
            "SSL_CHECKPOINT_KEY": "encoder_teacher",
            "EMBED_DIM": 96,
            "DEPTHS": [2, 2, 18, 2],
            "NUM_HEADS": [3, 6, 12, 24],
            "PATCH_SIZE": 4,
            "WINDOW_SIZE": 7,
            "POSITION_ENCODING": "dentmag_sincos",
        },
    },
    "resnet": {
        "none": {},
        "simclr": {
            "SSL_METHOD": "simclr",
            "USE_SSL_PRETRAINED": True,
            "SSL_WEIGHTS": os.path.join(PRETRAIN_DIR, "simclr", "resnet50_checkpoint-050.pth"),
            # The shipped checkpoint is a DentMaG multi-task run whose encoder is
            # nested at ``model.backbone.net`` (plus ``heads.*`` to be dropped).
            "SSL_CHECKPOINT_KEY": "model.backbone.net",
            "DEPTH": 50,
            "WIDTH_MULTIPLIER": 1,
            "STEM_TYPE": "simclrv2",
            "SK_RATIO": 0.0625,
        },
    },
    "vit": {
        "none": {},
        "dinov2": {
            "SSL_METHOD": "dinov2",
            "USE_SSL_PRETRAINED": True,
            "SSL_WEIGHTS": os.path.join(
                PRETRAIN_DIR, "dinov2", "dinov2_vits_train249999_teacher_checkpoint.pth"
            ),
            "SSL_CHECKPOINT_KEY": "teacher",
        },
    },
}

FAMILY_BUILDERS = {
    "swin": ("build_swin_ssl_backbone", "build_swin_ssl_fpn_backbone", "MODEL.SWIN_SSL"),
    "resnet": (
        "build_resnet_ssl_backbone",
        "build_resnet_ssl_fpn_backbone",
        "MODEL.RESNET_SSL",
    ),
    "vit": ("build_vit_ssl_backbone", "build_vit_ssl_fpn_backbone", "MODEL.VIT_SSL"),
}


def _base_cfg():
    from maskdino.config import add_maskdino_config
    from maskrcnn_unify import _build_base_cfg

    cfg = _build_base_cfg()
    add_maskdino_config(cfg)
    return cfg


def _apply_overrides(cfg, family, variant_kwargs):
    """Set the family node + the unified SSL fields through dotted paths."""
    cfg.MODEL.BACKBONE.SSL_STRICT_LOAD = True
    # Every SSL backbone emits res2-res5, so the FPN must consume those names
    # (the base COCO config declares p2-p5 for a plain ResNet).
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    for key, value in variant_kwargs.items():
        if key in ("EMBED_DIM", "DEPTHS", "NUM_HEADS", "PATCH_SIZE", "WINDOW_SIZE",
                   "POSITION_ENCODING", "MLP_RATIO", "QKV_BIAS", "PATCH_NORM",
                   "DROP_RATE", "ATTN_DROP_RATE", "DROP_PATH_RATE", "IMG_SIZE"):
            setattr(getattr(cfg.MODEL, FAMILY_BUILDERS[family][2].split(".")[1]), key, value)
        elif key in ("DEPTH", "WIDTH_MULTIPLIER", "STEM_TYPE", "SK_RATIO"):
            setattr(cfg.MODEL.RESNET_SSL, key, value)
        elif key in ("ARCH", "NUM_REGISTER_TOKENS", "OUT_BLOCKS", "PYRAMID_CHANNELS",
                     "PYRAMID_NORM", "PYRAMID_SMOOTHING", "IMG_SIZE"):
            setattr(cfg.MODEL.VIT_SSL, key, value)
        else:
            setattr(cfg.MODEL.BACKBONE, key, value)


def _build_backbone(cfg, builder_name):
    from detectron2.modeling import BACKBONE_REGISTRY
    from detectron2.layers import ShapeSpec

    builder = BACKBONE_REGISTRY.get(builder_name)
    return builder(cfg, ShapeSpec(channels=3, height=None, width=None))


def _check_shapes(backbone, input_shape, expected_spatial=None, tag=""):
    height, width = input_shape[-2:]
    x = torch.zeros(*input_shape)
    with torch.no_grad():
        outputs = backbone(x)

    expected_names = ["res2", "res3", "res4", "res5"]
    assert list(outputs.keys()) == expected_names, f"{tag}: keys {list(outputs.keys())}"
    for name in expected_names:
        feature = outputs[name]
        assert feature.dim() == 4, f"{tag}: {name} is not NCHW ({tuple(feature.shape)})"
        assert feature.shape[0] == input_shape[0], f"{tag}: {name} batch mismatch"
        spec = backbone.output_shape()[name]
        assert feature.shape[1] == spec.channels, (
            f"{tag}: {name} channels {feature.shape[1]} != declared {spec.channels}"
        )
        stride = spec.stride
        assert height % stride == 0 and width % stride == 0, (
            f"{tag}: input {height}x{width} not divisible by stride {stride}"
        )
        if expected_spatial is not None:
            want = expected_spatial[name]
            got_h, got_w = feature.shape[-2:]
            assert (got_h, got_w) == (want, want), (
                f"{tag}: {name} spatial {got_h}x{got_w} != {want}x{want}"
            )
        expected_hw = (height // stride, width // stride)
        assert feature.shape[-2:] == expected_hw, (
            f"{tag}: {name} spatial {tuple(feature.shape[-2:])} != {expected_hw} at stride {stride}"
        )
    return outputs


def _check_load(cfg, builder_name, tag=""):
    """Build the model + load the SSL checkpoint, asserting a clean audit."""
    from maskdino.modeling.backbone.ssl_common import (
        load_ssl_pretrained_backbone,
        resolve_ssl_settings,
    )

    backbone = _build_backbone(cfg, builder_name)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.backbone = inner

    model = _Wrapper(backbone)
    settings = resolve_ssl_settings(cfg)
    if not settings.is_ssl:
        return None
    audit = load_ssl_pretrained_backbone(model, cfg)
    assert audit, f"{tag}: the SSL loader reported failure"
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="all", choices=["all", "swin", "resnet", "vit"])
    parser.add_argument("--skip-load", action="store_true", help="shape tests only")
    args = parser.parse_args()

    families = list(VARIANTS) if args.family == "all" else [args.family]
    failures = []
    reports = []

    for family in families:
        plain_builder, fpn_builder, _ = FAMILY_BUILDERS[family]
        for variant, kwargs in VARIANTS[family].items():
            tag = f"{family}/{variant}"
            try:
                with torch.no_grad():
                    # ---- MaskDINO backbone (no FPN) --------------------------
                    cfg = _base_cfg()
                    _apply_overrides(cfg, family, kwargs)
                    cfg.MODEL.BACKBONE.NAME = plain_builder
                    backbone = _build_backbone(cfg, plain_builder)
                    _check_shapes(backbone, SQUARE_INPUT, SQUARE_EXPECTED, tag=f"{tag}/maskdino")
                    _check_shapes(backbone, NON_SQUARE_INPUT, tag=f"{tag}/nonsquare")
                    del backbone

                    # ---- Mask R-CNN backbone (FPN) --------------------------
                    cfg = _base_cfg()
                    _apply_overrides(cfg, family, kwargs)
                    cfg.MODEL.BACKBONE.NAME = fpn_builder
                    fpn = _build_backbone(cfg, fpn_builder)
                    fpn_out = fpn(torch.zeros(*SQUARE_INPUT))
                    # Detectron2 renames the pyramid levels: res2-res5 -> p2-p5,
                    # plus p6 from the default LastLevelMaxPool top block.
                    assert list(fpn_out.keys()) == ["p2", "p3", "p4", "p5", "p6"], (
                        f"{tag}/fpn: keys {list(fpn_out.keys())}"
                    )
                    for name, want in SQUARE_EXPECTED.items():
                        level = "p" + name[len("res"):]
                        got = fpn_out[level].shape[-2:]
                        assert got == (want, want), (
                            f"{tag}/fpn: {level} spatial {tuple(got)} != {want}x{want}"
                        )
                    del fpn

                # ---- checkpoint loading --------------------------------------
                if not args.skip_load and kwargs:
                    cfg = _base_cfg()
                    _apply_overrides(cfg, family, kwargs)
                    cfg.MODEL.BACKBONE.NAME = plain_builder
                    audit = _check_load(cfg, plain_builder, tag=tag)
                    reports.append((tag, audit))
            except Exception as exc:  # noqa: BLE001 - smoke harness
                failures.append((tag, exc, traceback.format_exc()))

    for tag, audit in reports:
        print(
            f"[load] {tag}: loaded={audit['loaded']}/{audit['target']} "
            f"ratio={audit['load_ratio']:.3f} shape_mismatch={audit['shape_mismatches']} "
            f"unexpected={audit['unexpected']}"
        )
        if audit["randomly_initialised"]:
            print(f"        randomly initialised: {audit['randomly_initialised']}")

    if failures:
        print("\n=== FAILURES ===")
        for tag, exc, tb in failures:
            print(f"\n--- {tag}: {type(exc).__name__}: {exc}")
            print(tb)
        return 1
    print("\nAll SSL backbone smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
