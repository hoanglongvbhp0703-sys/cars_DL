"""
Fine-tune YOLOv8 on the merged vehicle dataset, then export TFLite
so the trained model can be loaded by MediaPipe's ObjectDetector.

Usage:
    python training/train.py [--epochs 50] [--batch 16] [--model yolov8n]

After training the best weights are exported as:
    data/models/vehicle_detector.tflite   ← used by MediaPipe pipeline
    data/models/vehicle_detector.pt       ← PyTorch weights for fine-tuning
"""

import argparse
import shutil
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "data" / "dataset_merged"
MODEL_OUT    = PROJECT_ROOT / "data" / "models"
RUN_DIR      = PROJECT_ROOT / "training" / "runs"

# Re-use the existing YOLO-format datasets (no conversion needed for training)
DATASET_YOLO_YAML = DATASET_DIR / "dataset.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 vehicle detector")
    p.add_argument("--epochs", type=int,   default=50,       help="Training epochs")
    p.add_argument("--batch",  type=int,   default=16,       help="Batch size (lower if OOM)")
    p.add_argument("--imgsz",  type=int,   default=640,      help="Input image size")
    p.add_argument("--model",  type=str,   default="yolov8n",
                   choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                   help="YOLO model variant (n=nano, s=small, m=medium, l=large)")
    p.add_argument("--device", type=str,   default="",       help="Device: '' auto, 'cpu', '0'")
    p.add_argument("--resume", action="store_true",          help="Resume from last checkpoint")
    return p.parse_args()


def check_dataset():
    """dataset_merged must exist with YOLO-format images + labels."""
    yaml = DATASET_YOLO_YAML
    if not yaml.exists():
        print(f"ERROR: dataset.yaml not found at {yaml}")
        print("Run: python training/prepare_dataset.py  first.")
        sys.exit(1)

    # dataset.yaml for YOLO training must point to YOLO label dirs (not COCO JSON)
    # prepare_dataset.py already writes a YOLO-compatible yaml
    for split in ("train", "val"):
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        if not img_dir.exists():
            print(f"ERROR: {img_dir} not found. Run prepare_dataset.py first.")
            sys.exit(1)
        if not lbl_dir.exists():
            print(f"WARNING: {lbl_dir} not found – will be generated from COCO JSON")
    print(f"Dataset: {DATASET_DIR}")
    for split in ("train", "val", "test"):
        d = DATASET_DIR / split / "images"
        if d.exists():
            n = len(list(d.glob("*.*")))
            print(f"  {split:<6}: {n} images")


def coco_to_yolo_labels(dataset_dir: Path) -> None:
    """Convert COCO JSON annotations to YOLO .txt label files."""
    import json as _json

    CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
    for split in ("train", "val", "test"):
        json_path = dataset_dir / split / "labels.json"
        lbl_dir   = dataset_dir / split / "labels"
        img_dir   = dataset_dir / split / "images"
        if not json_path.exists() or not img_dir.exists():
            continue
        lbl_dir.mkdir(exist_ok=True)
        coco = _json.loads(json_path.read_text())
        img_map = {img["id"]: img for img in coco["images"]}
        ann_by_img: dict = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)
        for img_id, img_info in img_map.items():
            stem = Path(img_info["file_name"]).stem
            lbl_path = lbl_dir / f"{stem}.txt"
            lines = []
            for ann in ann_by_img.get(img_id, []):
                cls_id = ann["category_id"] - 1   # COCO 1-indexed → 0-indexed
                x, y, bw, bh = ann["bbox"]
                iw, ih = img_info["width"], img_info["height"]
                cx_n = (x + bw / 2) / iw
                cy_n = (y + bh / 2) / ih
                w_n  = bw / iw
                h_n  = bh / ih
                lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"  YOLO labels written for {split}: {lbl_dir}")


def write_yolo_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    content = (
        f"path: {dataset_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n\n"
        f"nc: 5\n"
        f"names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def export_tflite(weights_path: Path, imgsz: int) -> Path | None:
    """Export YOLOv8 .pt → TFLite via ultralytics export."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        model.export(format="tflite", imgsz=imgsz, int8=False)
        # YOLO exports alongside the .pt file
        tflite_src = weights_path.parent / (weights_path.stem + "_saved_model") / (weights_path.stem + ".tflite")
        if not tflite_src.exists():
            # Try alternate path
            for candidate in weights_path.parent.rglob("*.tflite"):
                tflite_src = candidate
                break
        if tflite_src and tflite_src.exists():
            return tflite_src
    except Exception as e:
        print(f"TFLite export failed: {e}")
    return None


def main():
    args = parse_args()

    print("=" * 60)
    print("  YOLOv8 Vehicle Detector Training")
    print("  (exports TFLite for MediaPipe inference)")
    print("=" * 60)

    check_dataset()

    # Ensure YOLO-format label files exist
    lbl_dir_train = DATASET_DIR / "train" / "labels"
    if not lbl_dir_train.exists() or not any(lbl_dir_train.glob("*.txt")):
        print("\nConverting COCO JSON -> YOLO labels...")
        coco_to_yolo_labels(DATASET_DIR)

    yaml_path = write_yolo_yaml(DATASET_DIR)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_weight = f"{args.model}.pt"
    print(f"\nModel    : {model_weight}")
    print(f"Epochs   : {args.epochs}")
    print(f"Batch    : {args.batch}")
    print(f"Img size : {args.imgsz}")
    device   = args.device if args.device else "0" if _has_gpu() else "cpu"
    print(f"Device   : {device}")
    print()

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_weight)
    t0 = time.perf_counter()

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(RUN_DIR),
        name="vehicle_yolov8",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        resume=args.resume,
        verbose=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # ── Copy best weights ────────────────────────────────────────────────────
    best_pt = Path(str(RUN_DIR)) / "vehicle_yolov8" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(str(RUN_DIR)) / "vehicle_yolov8" / "weights" / "last.pt"

    dst_pt = MODEL_OUT / "vehicle_detector.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dst_pt)
        print(f"Best weights → {dst_pt}")

    # ── Validate on test set ─────────────────────────────────────────────────
    print("\nValidating on test set...")
    val_model = YOLO(str(dst_pt))
    metrics = val_model.val(data=str(yaml_path), split="test", verbose=False)
    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    print(f"  mAP@50     : {map50:.4f}")
    print(f"  mAP@50-95  : {map50_95:.4f}")

    # ── Export TFLite ────────────────────────────────────────────────────────
    print("\nExporting TFLite for MediaPipe...")
    tflite_src = export_tflite(dst_pt, args.imgsz)
    dst_tflite = MODEL_OUT / "vehicle_detector.tflite"
    if tflite_src and tflite_src.exists():
        shutil.copy2(tflite_src, dst_tflite)
        print(f"TFLite model → {dst_tflite}")
    else:
        print("TFLite export skipped (requires TensorFlow). Model will use .pt weights.")

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "training_time_min": round(elapsed / 60, 2),
        "test_metrics": {
            "mAP50":    round(map50, 4),
            "mAP50_95": round(map50_95, 4),
        },
        "weights_pt":     str(dst_pt),
        "weights_tflite": str(dst_tflite) if dst_tflite.exists() else None,
    }
    report_path = MODEL_OUT / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training report → {report_path}")
    print("\nDone!")
    if dst_tflite.exists():
        print("MediaPipe pipeline will automatically use the fine-tuned TFLite model.")
    else:
        print("To use fine-tuned model, update detection_pipeline.py to load vehicle_detector.pt")


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
