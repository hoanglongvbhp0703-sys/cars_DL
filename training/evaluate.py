"""
Evaluate a trained .tflite model on the test split.
Outputs per-class metrics + confusion summary + annotated sample images.

Usage:
    python training/evaluate.py [--model data/models/vehicle_detector.tflite]
                                [--conf 0.45] [--samples 20]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DATASET_DIR = PROJECT_ROOT / "data" / "dataset_merged"
EVAL_OUT    = PROJECT_ROOT / "training" / "runs" / "eval"

CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
COLORS  = [(255,50,50),(50,50,255),(50,200,80),(255,150,0),(200,200,50)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default=str(PROJECT_ROOT / "data" / "models" / "vehicle_detector.tflite"))
    p.add_argument("--conf",    type=float, default=0.45)
    p.add_argument("--samples", type=int,   default=20, help="Number of annotated sample images to save")
    return p.parse_args()


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_gt(labels_json: Path) -> dict[int, list]:
    coco = json.loads(labels_json.read_text())
    gt: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        gt.setdefault(img_id, []).append((ann["category_id"] - 1, *ann["bbox"]))
    return gt


def run_inference(model_path: str, img_path: Path, conf_thresh: float):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        print("mediapipe not installed"); sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        score_threshold=conf_thresh,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        detector.close()
        return [], None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    detector.close()
    dets = []
    for d in result.detections:
        cat = d.categories[0]
        bb  = d.bounding_box
        dets.append((cat.category_name.lower(), float(cat.score),
                     int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)))
    return dets, bgr


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python training/train.py")
        sys.exit(1)

    test_img_dir = DATASET_DIR / "test" / "images"
    test_lbl     = DATASET_DIR / "test" / "labels.json"
    if not test_img_dir.exists():
        print("Test set not found. Run: python training/prepare_dataset.py"); sys.exit(1)

    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    coco     = json.loads(test_lbl.read_text())
    img_map  = {img["id"]: img for img in coco["images"]}
    gt_boxes = load_gt(test_lbl)

    img_files = sorted(test_img_dir.glob("*.*"))
    print(f"Evaluating {len(img_files)} test images with model {model_path.name} ...")

    tp_total = fp_total = fn_total = 0
    per_class_tp   = {c: 0 for c in CLASSES}
    per_class_fp   = {c: 0 for c in CLASSES}
    per_class_fn   = {c: 0 for c in CLASSES}
    save_count     = 0
    sample_indices = set(random.sample(range(len(img_files)), min(args.samples, len(img_files))))

    for idx, img_path in enumerate(img_files):
        img_id = int(img_path.stem)
        gt = gt_boxes.get(img_id, [])
        dets, bgr = run_inference(str(model_path), img_path, args.conf)

        matched_gt = set()
        for (lbl, score, dx, dy, dw, dh) in dets:
            cls_name = lbl.capitalize()
            cls_id   = CLASSES.index(cls_name) if cls_name in CLASSES else -1
            best_iou, best_j = 0.0, -1
            for j, (gc, gx, gy, gw, gh) in enumerate(gt):
                if gc != cls_id or j in matched_gt:
                    continue
                v = iou((dx, dy, dw, dh), (gx, gy, gw, gh))
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.5 and best_j >= 0:
                tp_total += 1
                if cls_name in CLASSES:
                    per_class_tp[cls_name] += 1
                matched_gt.add(best_j)
            else:
                fp_total += 1
                if cls_name in CLASSES:
                    per_class_fp[cls_name] += 1

        fn = len(gt) - len(matched_gt)
        fn_total += fn
        for j, (gc, *_) in enumerate(gt):
            if j not in matched_gt:
                per_class_fn[CLASSES[gc]] += 1

        # Save annotated samples
        if idx in sample_indices and bgr is not None:
            out = bgr.copy()
            for (lbl, score, dx, dy, dw, dh) in dets:
                ci = CLASSES.index(lbl.capitalize()) if lbl.capitalize() in CLASSES else 0
                c  = COLORS[ci]
                cv2.rectangle(out, (dx, dy), (dx+dw, dy+dh), c, 2)
                cv2.putText(out, f"{lbl} {score:.0%}", (dx, dy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            save_path = EVAL_OUT / f"sample_{idx:04d}.jpg"
            cv2.imwrite(str(save_path), out)
            save_count += 1

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(img_files)}", end="\r", flush=True)

    # ── Metrics ──────────────────────────────────────────────────────────────
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Overall  |  P: {precision:.3f}  R: {recall:.3f}  F1: {f1:.3f}")
    print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"{'='*50}")
    print(f"  {'Class':<14}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'P':>6}  {'R':>6}")
    results_per_class = {}
    for cls in CLASSES:
        tp = per_class_tp[cls]; fp = per_class_fp[cls]; fn = per_class_fn[cls]
        p  = tp/(tp+fp) if (tp+fp) else 0
        r  = tp/(tp+fn) if (tp+fn) else 0
        results_per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p,4), "recall": round(r,4)}
        print(f"  {cls:<14}  {tp:>5}  {fp:>5}  {fn:>5}  {p:>6.3f}  {r:>6.3f}")
    print(f"{'='*50}")
    print(f"{save_count} annotated samples → {EVAL_OUT}")

    report = {
        "model": str(model_path),
        "conf_threshold": args.conf,
        "overall": {"precision": round(precision,4), "recall": round(recall,4), "f1": round(f1,4)},
        "per_class": results_per_class,
    }
    (EVAL_OUT / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"Report → {EVAL_OUT / 'eval_report.json'}")


if __name__ == "__main__":
    main()
