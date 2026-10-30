#!/usr/bin/env python3
"""
Train a YOLO26 instance segmentation model from a folder of images
with a COCO-style annotation file, then save the trained weights.

Install:
    pip install -U ultralytics pyyaml

Usage:
    python train_yolo26_seg.py --data-dir /path/to/folder
    python train_yolo26_seg.py --data-dir /path/to/folder --ann instances.json \
        --model yolo26s-seg.pt --epochs 150 --imgsz 320 --batch 32 --device 0

The folder should contain the images (optionally in subfolders) and one COCO
JSON file (e.g. annotations.json). If --ann is omitted, the script uses the
only .json file it finds in the folder.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO


# --------------------------------------------------------------------------- #
# COCO -> YOLO segmentation conversion
# --------------------------------------------------------------------------- #
def find_annotation_file(data_dir: Path, ann: str | None) -> Path:
    if ann:
        p = Path(ann)
        p = p if p.is_absolute() else data_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {p}")
        return p
    jsons = sorted(data_dir.rglob("*.json"))
    if len(jsons) != 1:
        raise ValueError(
            f"Expected exactly one .json in {data_dir}, found {len(jsons)}. "
            "Pass the right one with --ann."
        )
    return jsons[0]


def resolve_image(data_dir: Path, file_name: str, index: dict) -> Path | None:
    p = data_dir / file_name
    if p.exists():
        return p
    return index.get(Path(file_name).name)  # fall back to basename lookup


def rle_to_polygons(segmentation, h, w):
    """Convert RLE (iscrowd) masks to polygons if pycocotools + cv2 are available."""
    try:
        import cv2
        import numpy as np
        from pycocotools import mask as mask_utils

        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, h, w)
        m = mask_utils.decode(rle).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
    except Exception:
        return []


def convert_coco_to_yolo(data_dir: Path, ann_path: Path, out_dir: Path,
                         val_split: float, seed: int) -> Path:
    with open(ann_path) as f:
        coco = json.load(f)

    # Contiguous class ids (0..N-1) ordered by COCO category id
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    names = {i: c["name"] for i, c in enumerate(categories)}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    image_index = {}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            image_index.setdefault(p.name, p)

    # Keep only images that exist on disk
    images = []
    for im in coco["images"]:
        src = resolve_image(data_dir, im["file_name"], image_index)
        if src is None:
            print(f"[warn] image not found, skipping: {im['file_name']}")
            continue
        images.append((im, src))
    if not images:
        raise RuntimeError("No images from the annotation file were found on disk.")

    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_split)) if len(images) > 1 else 0
    splits = {"val": images[:n_val], "train": images[n_val:]}

    if out_dir.exists():
        shutil.rmtree(out_dir)

    n_instances = 0
    for split, items in splits.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for im, src in items:
            w, h = im["width"], im["height"]
            flat = Path(im["file_name"]).as_posix().replace("/", "_")
            dst = img_out / flat
            shutil.copy2(src, dst)

            lines = []
            for a in anns_by_img.get(im["id"], []):
                cls = cat_to_idx[a["category_id"]]
                seg = a.get("segmentation")
                polys = []
                if isinstance(seg, list) and seg:
                    polys = [p for p in seg if len(p) >= 6]
                elif isinstance(seg, dict):  # RLE
                    polys = rle_to_polygons(seg, h, w)
                if not polys and a.get("bbox"):  # fallback: box as polygon
                    x, y, bw, bh = a["bbox"]
                    polys = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]

                for poly in polys:
                    coords = []
                    for i in range(0, len(poly), 2):
                        coords.append(min(max(poly[i] / w, 0.0), 1.0))
                        coords.append(min(max(poly[i + 1] / h, 0.0), 1.0))
                    lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in coords))
                    n_instances += 1

            (lbl_out / f"{Path(flat).stem}.txt").write_text("\n".join(lines))

    data_yaml = out_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.safe_dump(
            {"path": str(out_dir.resolve()), "train": "images/train",
             "val": "images/val", "names": names},
            f, sort_keys=False,
        )

    print(f"Converted {len(images)} images ({len(splits['train'])} train / "
          f"{len(splits['val'])} val), {n_instances} instances, {len(names)} classes.")
    return data_yaml


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train YOLO26 instance segmentation on COCO-format data")
    ap.add_argument("--data-dir", required=True, help="Folder with images + COCO json")
    ap.add_argument("--ann", default=None, help="COCO json (relative to data-dir or absolute)")
    ap.add_argument("--model", default="yolo26n-seg.pt",
                    help="yolo26n/s/m/l/x-seg.pt (pretrained) or a .yaml for scratch training")
    ap.add_argument("--epochs", type=int, default=100)
    # Images are 320x213. YOLO takes one int (the long side, must be a multiple
    # of the stride 32): 320 keeps native resolution with no upscaling.
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--rect", action="store_true",
                    help="Rectangular batches (320x224 instead of 320x320 padding). "
                         "Faster, but disables shuffling and mosaic is less effective.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu, mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs/segment")
    ap.add_argument("--name", default="yolo26_seg")
    ap.add_argument("--dataset-dir", default="yolo_dataset", help="Where the converted YOLO dataset goes")
    ap.add_argument("--output", default="yolo26_seg_best.pt", help="Final weights file")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    ann_path = find_annotation_file(data_dir, args.ann)
    print(f"Using annotations: {ann_path}")

    data_yaml = convert_coco_to_yolo(
        data_dir, ann_path, Path(args.dataset_dir).resolve(), args.val_split, args.seed
    )

    
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
        rect=args.rect,
        plots=True,
    )

    # Ultralytics saves weights/best.pt and weights/last.pt in the run folder
    weights_dir = Path(model.trainer.save_dir) / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    src = best if best.exists() else last
    shutil.copy2(src, args.output)
    print(f"\nTraining complete. Weights saved to: {Path(args.output).resolve()}")
    print(f"(Also in {weights_dir})")

    # Optional: validate and show a quick mask metric summary
    metrics = YOLO(args.output).val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)
    print(f"Box mAP50-95: {metrics.box.map:.4f} | Mask mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
