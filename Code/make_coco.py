#!/usr/bin/env python3
"""
Build a COCO instance-segmentation JSON from rendered images + binary masks.

Expected layout (defaults, all configurable via CLI):

    images/Image0001.png          <- RGB render for frame 0001
    masks/mask_1_0001.png         <- mask of object 1 in frame 0001
    masks/mask_2_0001.png         <- mask of object 2 in frame 0001
    masks/mask_3_0001.png         <- mask of object 3 in frame 0001

Masks are black/white. An all-black mask means the object is not in that
frame, so no annotation is written for it.

All objects (1, 2, 3) belong to the SAME single category (category_id = 1).
Each (frame, object) pair with a non-empty mask becomes one annotation
(one instance). If an object is split into several disconnected blobs
(e.g. partially occluded), they are stored as multiple polygons in one
annotation.

Usage:
    python make_coco.py
    python make_coco.py --images_dir images --masks_dir masks \
        --output annotations.json --class_name widget

Requires: opencv-python, numpy
    pip install opencv-python numpy
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMAGE_RE = re.compile(r"^Image(\d+)\.png$", re.IGNORECASE)
CATEGORY_ID = 0  # every object maps to this single category


def parse_args():
    p = argparse.ArgumentParser(description="Create a COCO instance segmentation JSON.")
    p.add_argument("--images_dir", default="images", help="Folder with ImageXXXX.png files.")
    p.add_argument("--masks_dir", default="masks",
                   help="Folder with mask_N_XXXX.png files (default: 'masks').")
    p.add_argument("--output", default="coco.json", help="Output COCO JSON path.")
    p.add_argument("--num_objects", type=int, default=3,
                   help="Number of mask objects per frame (mask_1_..., mask_2_..., ...). "
                        "Default 3.")
    p.add_argument("--class_name", default="frame",
                   help="Name of the single category all objects belong to. Default 'object'.")
    p.add_argument("--threshold", type=int, default=127,
                   help="Pixel value above which a mask pixel counts as foreground.")
    p.add_argument("--min_area", type=int, default=1,
                   help="Skip masks whose foreground pixel count is below this.")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Polygon simplification (approxPolyDP epsilon in px). "
                        "0 keeps every contour point.")
    return p.parse_args()


def load_mask(path: Path, threshold: int):
    """Load a mask as a binary uint8 array (0/1), or None if unreadable."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > threshold).astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, epsilon: float):
    """Convert a binary mask to a list of COCO polygons ([x1,y1,x2,y2,...])."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        c = c.reshape(-1, 2)
        if len(c) < 3:          # need at least a triangle
            continue
        polygons.append([float(v) for v in c.flatten()])
    return polygons


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)

    if not images_dir.is_dir():
        sys.exit(f"Images folder not found: {images_dir}")
    if not masks_dir.is_dir():
        sys.exit(f"Masks folder not found: {masks_dir}")

    categories = [{"id": CATEGORY_ID, "name": "object", "supercategory": "frame"}]

    # Collect images, sorted by frame index.
    frames = []
    for f in images_dir.iterdir():
        m = IMAGE_RE.match(f.name)
        if m:
            frames.append((int(m.group(1)), m.group(1), f))
    frames.sort(key=lambda t: t[0])
    if not frames:
        sys.exit(f"No files matching 'ImageXXXX.png' found in {images_dir}")

    images, annotations = [], []
    ann_id = 1
    skipped_empty = 0
    missing_files = 0

    for img_id, (frame_idx, frame_str, img_path) in enumerate(frames, start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping frame.")
            continue
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "license": 1,
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        for obj in range(1, args.num_objects + 1):
            # Keep the same zero-padding as the image's frame string (e.g. 0001).
            mask_path = masks_dir / f"mask_{obj}_{frame_str}.png"
            if not mask_path.exists():
                missing_files += 1
                continue

            mask = load_mask(mask_path, args.threshold)
            if mask is None:
                print(f"WARNING: could not read {mask_path}, skipping.")
                continue
            if mask.shape != (h, w):
                print(f"WARNING: {mask_path.name} is {mask.shape[::-1]} but image is "
                      f"{(w, h)}; resizing mask (nearest).")
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            area = int(mask.sum())
            if area < args.min_area:       # black mask -> object not present
                skipped_empty += 1
                continue

            polygons = mask_to_polygons(mask, args.epsilon)
            if not polygons:
                skipped_empty += 1
                continue

            ys, xs = np.where(mask > 0)
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_ID,
                "segmentation": polygons,
                "area": float(area),
                "bbox": [float(v) for v in bbox],
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {
            "year": "2026",
            "version": "1",
            "description": "Instance segmentation dataset generated from Blender renders",
            "contributor": "Alec",
            "url": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    #print("Images: " + images)
    #print("Annoations: " + annotations)
    #print("Categories: " + categories)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco, f)

    print(f"Wrote {out}")
    print(f"  images:      {len(images)}")
    print(f"  annotations: {len(annotations)}")
    print(f"  empty masks skipped: {skipped_empty}")
    if missing_files:
        print(f"  mask files not found: {missing_files}")


if __name__ == "__main__":
    main()
