#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build tooth_mask and tooth_outline guidance images for the orth_test dataset.

Reorganizes each `<sample_id>/` folder from:
    D.jpg F.jpg L.jpg R.jpg U.jpg [R.json ...]
into:
    images/{D,F,L,R,U}.png        original photos, converted to png
    anno/*.json                   original LabelMe annotations
    tooth_mask/{D,F,L,R,U}.png    original photo, non-tooth pixels blacked out
    tooth_outline/{D,F,L,R,U}.png black background, white per-tooth contour lines

Tooth regions come from the same YOLO (per-view detector) + SAM (per-box
segmentation) pipeline used in tmp_prompt.py.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from tqdm import tqdm

from tools.sam import sam_load, sam_predict
from utils import suppress_stdout

LOGGER.setLevel("ERROR")

VIEW_MAPPING = {"F": "front", "U": "upper", "D": "lower", "L": "left", "R": "right"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build tooth_mask/tooth_outline guidance images")
    parser.add_argument("--base_dir", type=str, default=".datasets/intraoral_anno/orth_test/orth_test")
    parser.add_argument("--weight_dir", type=str, default=".checkpoints/segtooth_model")
    parser.add_argument("--sam_batch_size", type=int, default=10)
    parser.add_argument("--min_conf", type=float, default=0.35)
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--outline_thickness", type=int, default=2)
    parser.add_argument("--sample_ids", nargs="*", default=None, help="Only process these sample ids")
    return parser.parse_args()


def get_model_path(view, weight_dir):
    if view == "left":
        view = "right"
    name = "vit_tiny.pt" if view == "sam" else f"yolo11_{view}.pt"
    return str(Path(weight_dir) / f"segmentanytooth_{name}")


def load_models(weight_dir):
    with suppress_stdout():
        sam = sam_load(get_model_path("sam", weight_dir))
        yolos = {v: YOLO(get_model_path(v, weight_dir)) for v in ("front", "upper", "lower", "right")}
    return sam, yolos


def detect_tooth_mask(image_bgr, view, sam, yolo, args):
    """Run YOLO detection + SAM segmentation, return an FDI-labeled mask (H, W) uint8."""
    should_flip = view == "left"
    image = cv2.flip(image_bgr, 1) if should_flip else image_bgr

    with suppress_stdout():
        r = yolo.predict(image, save=False, save_txt=False, save_conf=False, save_crop=False, project=None)[0]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if r.boxes is None or len(r.boxes) == 0:
        return mask

    names = LEFT_CLASSES if should_flip else r.names
    boxes = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(np.int32)
    confs = r.boxes.conf.cpu().numpy()
    fdis = np.array([int("".join(filter(str.isdigit, names[c]))) for c in clss])

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (confs >= args.min_conf) & (areas >= args.min_area)
    boxes, confs, fdis = boxes[keep], confs[keep], fdis[keep]
    if len(boxes) == 0:
        return mask

    # keep only the highest-confidence detection per FDI id
    best = [idx[np.argmax(confs[idx])] for idx in (np.where(fdis == f)[0] for f in np.unique(fdis))]
    boxes, fdis = boxes[best], fdis[best]

    if should_flip:
        image = cv2.flip(image, 1)
        w = image.shape[1]
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_masks = sam_predict(sam=sam, boxes_xyxy=boxes, image=image_rgb, batch_size=args.sam_batch_size)
    for fdi, tooth_mask in zip(fdis, sam_masks):
        mask[tooth_mask == 1] = fdi
    return mask


def build_outline(mask, shape, thickness):
    outline = np.zeros(shape, dtype=np.uint8)
    for fdi in np.unique(mask):
        if fdi == 0:
            continue
        contours, _ = cv2.findContours((mask == fdi).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours, -1, (255, 255, 255), thickness)
    return outline


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    sam, yolos = load_models(args.weight_dir)

    sample_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    if args.sample_ids:
        sample_dirs = [d for d in sample_dirs if d.name in args.sample_ids]

    for sample_dir in tqdm(sample_dirs, desc="samples"):
        images_dir = sample_dir / "images"
        anno_dir = sample_dir / "anno"
        mask_dir = sample_dir / "tooth_mask"
        outline_dir = sample_dir / "tooth_outline"

        for stem, view in VIEW_MAPPING.items():
            out_name = f"{stem}.png"
            if (images_dir / out_name).exists():
                continue
            src_path = next((p for p in sample_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTS), None)
            if src_path is None:
                continue

            image = cv2.imread(str(src_path))
            yolo = yolos["right"] if view == "left" else yolos[view]
            mask = detect_tooth_mask(image, view, sam, yolo, args)

            for d in (images_dir, mask_dir, outline_dir):
                d.mkdir(parents=True, exist_ok=True)

            tooth_mask_img = cv2.bitwise_and(image, image, mask=(mask > 0).astype(np.uint8) * 255)
            cv2.imwrite(str(mask_dir / out_name), tooth_mask_img)
            cv2.imwrite(str(outline_dir / out_name), build_outline(mask, image.shape, args.outline_thickness))
            cv2.imwrite(str(images_dir / out_name), image)
            src_path.unlink()

        json_paths = list(sample_dir.glob("*.json"))
        if json_paths:
            anno_dir.mkdir(parents=True, exist_ok=True)
            for json_path in json_paths:
                shutil.move(str(json_path), str(anno_dir / json_path.name))


if __name__ == "__main__":
    main()
