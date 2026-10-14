#!/usr/bin/env python
"""End-to-end detector smoke tests for every config in the organised tree.

For each ``configs/<detector>/<task>/<variant>.yaml`` this script

1. builds the real model through ``detectron2.modeling.build_model``,
2. runs the unified SSL initialiser (``load_ssl_pretrained_backbone``) and
   checks the audit ratios,
3. runs one training-mode forward on a synthetic batch and asserts every loss
   is finite,
4. runs one inference-mode forward and asserts the predictions are finite and
   that masks are only produced when the task uses them.

Nothing is saved and no dataset is touched, so this is a cheap gate before
launching a real run.

Run from the repository root:

    conda run -n intraoral python scripts/smoke_detectors.py
    conda run -n intraoral python scripts/smoke_detectors.py --family swin --variant dentmag_swin
"""

import argparse
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectron2.modeling import build_model  # noqa: E402
from detectron2.structures import BitMasks, Boxes, Instances  # noqa: E402
from detectron2.utils.events import EventStorage  # noqa: E402

from maskdino.modeling.backbone.ssl_common import load_ssl_pretrained_backbone  # noqa: E402
from validate_ssl_configs import COMBINATIONS, SSL_EXPECTATIONS, VARIANTS, load_cfg  # noqa: E402

#: ``(height, width)`` per task. Deliberately *not* multiples of the Swin/ViT
#: super-patch so the backbone padding path is exercised too.
TASK_INPUT = {
    "caries": (400, 600),
    "orth": (800, 1332),
}

#: SSL methods -> family, used by ``--family``.
FAMILY_OF_METHOD = {
    "simclr": "resnet",
    "swinmae": "swin",
    "dentmag_swin": "swin",
    "dinov2": "vit",
}
FAMILY_OF_VARIANT = {"no_ssl": "none", **FAMILY_OF_METHOD}


def make_batch(detector, task, height, width, device):
    """Two boxes in Detectron2's post-mapper format.

    The ``gt_masks`` field cannot be shared between the two detectors: Detectron2's
    own instance mappers keep a :class:`BitMasks`, while MaskDINO's DETR mapper
    replaces it with the raw boolean tensor produced by
    ``convert_coco_poly_to_mask`` and ``maskdino.py`` then indexes ``.shape``.

    Masks are attached for *both* tasks: ``maskdino.py``'s ``prepare_targets_detr``
    reads ``gt_masks`` unconditionally, so the boxes-only orth configs still need
    the field even though ``MASK_WEIGHT``/``DICE_WEIGHT`` are zero.
    """
    image = torch.rand(3, height, width)
    boxes = torch.tensor(
        [[10.0, 20.0, 100.0, 120.0], [200.0, 40.0, 260.0, 200.0]],
        dtype=torch.float32,
    )
    classes = torch.tensor([1, 2], dtype=torch.int64)

    instances = Instances((height, width), gt_boxes=Boxes(boxes), gt_classes=classes)
    masks = torch.zeros((2, height, width), dtype=torch.bool)
    masks[0, 20:120, 10:100] = True
    masks[1, 40:200, 200:260] = True
    instances.gt_masks = BitMasks(masks) if detector == "maskrcnn" else masks
    instances = instances.to(device)

    return [{"image": image.to(device), "height": height, "width": width, "instances": instances}]


def run_one(detector, task, variant, device, skip_load, swin_depths):
    path = REPO_ROOT / "configs" / detector / task / f"{variant}.yaml"
    tag = f"{detector}/{task}/{variant}"
    print(f"--- {tag} ---")

    num_classes = 9 if task == "caries" else 5
    cfg = load_cfg(path, detector, task, num_classes, swin_depths)
    cfg.MODEL.DEVICE = device
    # Keep the smoke run light: no test-time augmentation, no inference crops.
    cfg.TEST.AUG.ENABLED = False

    # -- 2. SSL initialisation ------------------------------------------- #
    model = build_model(cfg)
    if skip_load:
        print("  ssl: skipped (--skip-load)")
    else:
        # ``no_ssl`` has no SSL checkpoint at all; every other variant must
        # report a healthy audit from the unified loader.
        audit = load_ssl_pretrained_backbone(model, cfg)
        if variant == "no_ssl":
            if audit is not False:
                raise AssertionError(f"{tag}: expected no SSL init, got audit={audit!r}")
            print("  ssl: not configured (expected)")
        else:
            if not isinstance(audit, dict):
                raise AssertionError(
                    f"{tag}: loader returned {type(audit).__name__}, expected a dict"
                )
            ratio = audit["load_ratio"]
            print(
                f"  ssl: loaded={audit['loaded']}/{audit['target']} "
                f"ratio={ratio:.3f} shape_mismatch={audit['shape_mismatches']} "
                f"unexpected={audit['unexpected']}"
            )
            if ratio < 0.90:
                raise AssertionError(f"{tag}: load_ratio {ratio:.3f} < 0.90")
            if audit["shape_mismatches"]:
                raise AssertionError(
                    f"{tag}: {audit['shape_mismatches']} shape mismatches"
                )
            # Only the freshly added pyramid adapters (and DINOv2's unused mask
            # token) may stay random; anything else means a silent skip.
            stray = [
                name
                for name in audit["randomly_initialised"]
                if "adapter" not in name and "mask_token" not in name
            ]
            if stray:
                raise AssertionError(f"{tag}: unexpected random encoder tensors {stray[:5]}")

    model.to(device)
    height, width = TASK_INPUT[task]
    batch = make_batch(detector, task, height, width, device)

    # -- 3. training forward --------------------------------------------- #
    model.train()
    # Detectron2's RPN/ROI losses log scalars through the global event storage.
    with EventStorage(0), torch.no_grad():
        losses = model(batch)
    if not losses:
        raise AssertionError(f"{tag}: training forward returned no losses")
    bad = {name: float(value) for name, value in losses.items() if not torch.isfinite(value)}
    if bad:
        raise AssertionError(f"{tag}: non-finite losses {bad}")
    print("  train losses: " + ", ".join(f"{k}={float(v):.3f}" for k, v in sorted(losses.items())))

    if task == "caries":
        mask_keys = [k for k in losses if "mask" in k or "dice" in k]
        if detector == "maskrcnn" and "loss_mask" not in losses:
            raise AssertionError(f"{tag}: Mask R-CNN caries produced no loss_mask")
        if detector == "maskdino" and not mask_keys:
            raise AssertionError(f"{tag}: MaskDINO caries produced no mask loss")

    # -- 4. inference forward -------------------------------------------- #
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    if len(outputs) != 1:
        raise AssertionError(f"{tag}: expected 1 prediction, got {len(outputs)}")
    result = outputs[0]
    instances = result["instances"] if "instances" in result else result
    if not hasattr(instances, "pred_boxes"):
        raise AssertionError(f"{tag}: prediction has no pred_boxes")
    boxes = instances.pred_boxes.tensor
    if not torch.isfinite(boxes).all():
        raise AssertionError(f"{tag}: non-finite pred_boxes")
    has_masks = hasattr(instances, "pred_masks") and instances.pred_masks is not None
    if has_masks and not torch.isfinite(instances.pred_masks.to(torch.float32)).all():
        raise AssertionError(f"{tag}: non-finite pred_masks")
    if task == "caries" and not has_masks:
        raise AssertionError(f"{tag}: caries is a mask task but no pred_masks were returned")
    if task == "orth":
        if detector == "maskrcnn" and has_masks:
            raise AssertionError(f"{tag}: orth is boxes-only but pred_masks were returned")
        # MaskDINO's instance_inference() always produces masks -- the mask
        # branch is part of the decoder, not of the dataset. The orth config
        # turns it off by *training* with zero mask weight instead, which is
        # what makes it boxes-only; assert that, not the absence of tensors.
        if detector == "maskdino":
            weights = {
                "MASK_WEIGHT": float(cfg.MODEL.MaskDINO.MASK_WEIGHT),
                "DICE_WEIGHT": float(cfg.MODEL.MaskDINO.DICE_WEIGHT),
            }
            nonzero = {k: v for k, v in weights.items() if v != 0.0}
            if nonzero:
                raise AssertionError(
                    f"{tag}: orth must not train the mask branch, got {nonzero}"
                )
    print(
        f"  inference: {len(boxes)} boxes, masks={has_masks}, input={height}x{width}"
    )
    del model, batch, outputs, losses
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["all", "maskrcnn", "maskdino"], default="all")
    parser.add_argument("--task", choices=["all", "caries", "orth"], default="all")
    parser.add_argument("--family", choices=["all", "none", "resnet", "swin", "vit"], default="all")
    parser.add_argument("--variant", default=None, help="Run a single variant by config stem.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-load", action="store_true", help="Do not restore SSL weights.")
    parser.add_argument(
        "--swin-depths",
        type=int,
        nargs=4,
        default=[2, 2, 18, 2],
        help="Depths injected as MODEL.SWIN.DEPTHS, matching the entry points.",
    )
    args = parser.parse_args()

    failures = []
    passed = 0

    for detector, task, num_classes in COMBINATIONS:
        if args.detector != "all" and detector != args.detector:
            continue
        if args.task != "all" and task != args.task:
            continue
        for variant in VARIANTS:
            if args.variant and variant != args.variant:
                continue
            if args.family != "all" and FAMILY_OF_VARIANT[variant] != args.family:
                continue
            try:
                run_one(
                    detector,
                    task,
                    variant,
                    args.device,
                    args.skip_load,
                    args.swin_depths,
                )
                passed += 1
            except Exception as exc:  # noqa: BLE001 - report every failure
                failures.append(f"{detector}/{task}/{variant}: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    print()
    if failures:
        print(f"{len(failures)} failure(s) ({passed} passed):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"All {passed} detector smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
