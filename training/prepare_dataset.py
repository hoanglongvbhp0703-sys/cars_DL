"""
Merge two YOLO-format datasets → COCO JSON format required by MediaPipe Model Maker.
Split 80 / 10 / 10  (train / val / test).

Unified classes
───────────────
0: Ambulance   (from cars-detection only)
1: Bus
2: Car         (+ all top-view "Vehicle" remapped here)
3: Motorcycle
4: Truck

COCO output per split
─────────────────────
data/dataset_merged/
  train/  images/  labels.json
  val/    images/  labels.json
  test/   images/  labels.json
  dataset.yaml   (for reference)
"""

import json
import math
import random
import shutil
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "uploads" / "datasets"
OUT_DIR      = PROJECT_ROOT / "data" / "dataset_merged"

DS_TOPVIEW = DATASETS_DIR / "top-view-vehicle" / "Vehicle_Detection_Image_Dataset"
DS_CARS    = DATASETS_DIR / "cars-detection"   / "Cars Detection"

UNIFIED_CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
NC = len(UNIFIED_CLASSES)

# YOLO class_id → unified class_id
TOPVIEW_REMAP: dict[int, int] = {0: 2}          # Vehicle → Car
CARS_REMAP:    dict[int, int] = {i: i for i in range(5)}

SPLIT = (0.80, 0.10, 0.10)
SEED  = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── YOLO label reader ──────────────────────────────────────────────────────────

def read_yolo_boxes(lbl_path: Path, remap: dict[int, int], img_w: int, img_h: int):
    """Parse YOLO label file; return list of (unified_cls_id, x, y, w, h) in pixels."""
    boxes = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        new_id = remap.get(cls_id)
        if new_id is None:
            continue
        cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
        bw = w_n * img_w
        bh = h_n * img_h
        x1 = max(0.0, (cx_n - w_n / 2) * img_w)
        y1 = max(0.0, (cy_n - h_n / 2) * img_h)
        boxes.append((new_id, round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)))
    return boxes


# ── Sample collector ───────────────────────────────────────────────────────────

def collect(img_dirs: list[Path], lbl_dirs: list[Path], remap: dict[int, int]):
    samples = []
    for img_dir, lbl_dir in zip(img_dirs, lbl_dirs):
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                samples.append((img_path, lbl_path, remap))
    return samples


# ── COCO JSON builder ──────────────────────────────────────────────────────────

def build_coco(samples: list, split_dir: Path) -> dict:
    images_out = split_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    coco: dict = {
        "info": {"description": "Vehicle Detection Merged Dataset"},
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "vehicle"}
            for i, name in enumerate(UNIFIED_CLASSES)
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    for img_id, (img_src, lbl_src, remap) in enumerate(samples, start=1):
        # Copy image
        dst_name = f"{img_id:06d}{img_src.suffix}"
        dst_path = images_out / dst_name
        shutil.copy2(img_src, dst_path)

        # Image dimensions
        try:
            with Image.open(img_src) as im:
                w, h = im.size
        except Exception:
            w, h = 640, 480

        coco["images"].append({"id": img_id, "file_name": dst_name, "width": w, "height": h})

        boxes = read_yolo_boxes(lbl_src, remap, w, h)
        for (cls_id, x, y, bw, bh) in boxes:
            area = round(bw * bh, 2)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,   # COCO is 1-indexed
                "bbox": [x, y, bw, bh],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        if img_id % 200 == 0:
            print(f"    {img_id}/{len(samples)}", end="\r", flush=True)

    return coco


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Collecting samples...")
    topview = collect(
        [DS_TOPVIEW / "train" / "images", DS_TOPVIEW / "valid" / "images"],
        [DS_TOPVIEW / "train" / "labels", DS_TOPVIEW / "valid" / "labels"],
        TOPVIEW_REMAP,
    )
    cars = collect(
        [DS_CARS / "train" / "images", DS_CARS / "valid" / "images", DS_CARS / "test" / "images"],
        [DS_CARS / "train" / "labels", DS_CARS / "valid" / "labels", DS_CARS / "test" / "labels"],
        CARS_REMAP,
    )

    all_samples = topview + cars
    print(f"  top-view-vehicle : {len(topview):>5}")
    print(f"  cars-detection   : {len(cars):>5}")
    print(f"  total            : {len(all_samples):>5}")

    if not all_samples:
        print("ERROR: no samples found"); sys.exit(1)

    random.seed(SEED)
    random.shuffle(all_samples)

    n       = len(all_samples)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    splits  = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train : n_train + n_val],
        "test":  all_samples[n_train + n_val :],
    }
    print(f"\nSplit (seed={SEED}):")
    for k, v in splits.items():
        print(f"  {k:<6}: {len(v):>5}")

    # Rebuild output dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # Build COCO JSON per split
    for split_name, samples in splits.items():
        print(f"\nBuilding {split_name}...")
        split_dir = OUT_DIR / split_name
        coco = build_coco(samples, split_dir)
        lbl_path = split_dir / "labels.json"
        lbl_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        boxes_total = len(coco["annotations"])
        print(f"  {len(samples)} images, {boxes_total} annotations -> {lbl_path}")

    # Class distribution on train
    print("\nClass distribution (train):")
    ann_file = OUT_DIR / "train" / "labels.json"
    coco_train = json.loads(ann_file.read_text())
    counts = {c: 0 for c in UNIFIED_CLASSES}
    for ann in coco_train["annotations"]:
        cls_name = UNIFIED_CLASSES[ann["category_id"] - 1]
        counts[cls_name] += 1
    for cls, cnt in counts.items():
        bar = "#" * (cnt // 30)
        print(f"  {cls:<12}: {cnt:>6}  {bar}")

    # Write dataset.yaml for reference
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n\n"
        f"nc: {NC}\n"
        f"names: {UNIFIED_CLASSES}\n",
        encoding="utf-8",
    )

    print(f"\nDone. Dataset ready at: {OUT_DIR}".encode("ascii","replace").decode())


if __name__ == "__main__":
    main()
