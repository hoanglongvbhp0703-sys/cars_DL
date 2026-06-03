"""
MediaPipe-based vehicle detection pipeline.

Model priority (auto-resolved from data/models/):
  1. vehicle_detector.tflite  ← fine-tuned via MediaPipe Model Maker / YOLOv8 export
  2. vehicle_detector.pt      ← YOLOv8 PyTorch weights (used via ultralytics inference)
  3. efficientdet_lite0.tflite← pre-trained COCO model (auto-downloaded if missing)

All paths run through the same annotate() / detect() API so video_processor
does not need to know which backend is active.
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path
from typing import List, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..core.logging_config import get_logger
from ..schemas.detection import VehicleDetection, BoundingBox

logger = get_logger(__name__)

# ── Class definitions ─────────────────────────────────────────────────────────
PRETRAINED_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
FINETUNED_VEHICLE_CLASSES  = {"ambulance", "bus", "car", "motorcycle", "truck"}

VEHICLE_COLORS: dict[str, Tuple[int, int, int]] = {
    "car":        (0,   200,  80),
    "motorcycle": (255, 140,   0),
    "bus":        (0,   120, 255),
    "truck":      (220,  50,  50),
    "bicycle":    (200, 200,   0),
    "ambulance":  (255,  50, 200),
}

# Model filenames
FINETUNED_TFLITE   = "vehicle_detector.tflite"
FINETUNED_PT       = "vehicle_detector.pt"
PRETRAINED_TFLITE  = "efficientdet_lite0.tflite"
PRETRAINED_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
)

# YOLO class names (must match training dataset)
YOLO_CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]


def download_model(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading pre-trained model → {dest}")
    urllib.request.urlretrieve(url, dest)
    logger.info("Download complete.")


def resolve_model_path(models_dir: Path) -> tuple[Path, str]:
    """
    Return (model_path, backend) where backend is one of:
      'tflite_finetuned' | 'yolo' | 'tflite_pretrained'
    """
    p_tflite  = models_dir / FINETUNED_TFLITE
    p_pt      = models_dir / FINETUNED_PT
    p_pretrained = models_dir / PRETRAINED_TFLITE

    if p_tflite.exists():
        logger.info(f"Fine-tuned TFLite model found: {p_tflite}")
        return p_tflite, "tflite_finetuned"

    if p_pt.exists():
        logger.info(f"Fine-tuned YOLOv8 model found: {p_pt}")
        return p_pt, "yolo"

    if not p_pretrained.exists():
        logger.info("Downloading pre-trained EfficientDet Lite 0...")
        download_model(PRETRAINED_URL, p_pretrained)

    return p_pretrained, "tflite_pretrained"


# ── MediaPipe TFLite backend ──────────────────────────────────────────────────

class _TFLiteBackend:
    def __init__(self, model_path: Path, confidence: float, finetuned: bool):
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=confidence,
            max_results=50,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)
        self._classes  = FINETUNED_VEHICLE_CLASSES if finetuned else PRETRAINED_VEHICLE_CLASSES

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for det in result.detections:
            cat = det.categories[0]
            label = cat.category_name.lower()
            if label not in self._classes:
                continue
            bb = det.bounding_box
            out.append(VehicleDetection(
                label=label,
                confidence=round(float(cat.score), 4),
                bbox=BoundingBox(x=int(bb.origin_x), y=int(bb.origin_y),
                                 width=int(bb.width), height=int(bb.height)),
                color=list(VEHICLE_COLORS.get(label, (0, 255, 0))),
            ))
        return out

    def close(self):
        self._detector.close()


# ── YOLOv8 backend ────────────────────────────────────────────────────────────

class _YOLOBackend:
    def __init__(self, model_path: Path, confidence: float):
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
            self._conf  = confidence
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        results = self._model.predict(frame_bgr, conf=self._conf, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = YOLO_CLASSES[cls_id].lower() if cls_id < len(YOLO_CLASSES) else "vehicle"
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                out.append(VehicleDetection(
                    label=label,
                    confidence=round(conf, 4),
                    bbox=BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1),
                    color=list(VEHICLE_COLORS.get(label, (0, 255, 0))),
                ))
        return out

    def close(self):
        pass


# ── Public API ────────────────────────────────────────────────────────────────

class DetectionPipeline:
    """
    Unified vehicle detection pipeline.
    Internally selects TFLite (MediaPipe) or YOLOv8 backend automatically.
    """

    def __init__(self, model_path: Path, confidence_threshold: float = 0.45):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "  Fine-tune  : python training/train.py\n"
                "  Pre-trained: python scripts/download_models.py"
            )

        self.model_name = model_path.name
        self.confidence_threshold = confidence_threshold

        if model_path.suffix == ".pt":
            self._backend = _YOLOBackend(model_path, confidence_threshold)
            self._backend_name = "YOLOv8 (fine-tuned)"
            self._is_finetuned = True
        else:
            is_finetuned = (model_path.name == FINETUNED_TFLITE)
            self._backend = _TFLiteBackend(model_path, confidence_threshold, is_finetuned)
            self._backend_name = "MediaPipe TFLite " + ("(fine-tuned)" if is_finetuned else "(pre-trained)")
            self._is_finetuned = is_finetuned

        logger.info(f"DetectionPipeline ready | {self._backend_name} | conf={confidence_threshold}")

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        return self._backend.detect(frame_bgr)

    def annotate(self, frame_bgr: np.ndarray, detections: List[VehicleDetection]) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1 = max(0, det.bbox.x)
            y1 = max(0, det.bbox.y)
            x2 = min(w, x1 + det.bbox.width)
            y2 = min(h, y1 + det.bbox.height)
            color_bgr = (det.color[2], det.color[1], det.color[0])

            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
            label_txt = f"{det.label}  {det.confidence:.0%}"
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            (tw, text_h), baseline = cv2.getTextSize(label_txt, font, fs, th)
            tag_y1 = max(0, y1 - text_h - baseline - 6)
            cv2.rectangle(out, (x1, tag_y1), (x1 + tw + 8, y1), color_bgr, -1)
            cv2.putText(out, label_txt, (x1 + 4, y1 - baseline - 2),
                        font, fs, (255, 255, 255), th, cv2.LINE_AA)

        # HUD
        count = len(detections)
        hud = f"Vehicles: {count}  [{self._backend_name}]"
        cv2.putText(out, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),    4, cv2.LINE_AA)
        cv2.putText(out, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

        return out

    def close(self):
        self._backend.close()
