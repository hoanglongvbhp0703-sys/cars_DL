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

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}

VEHICLE_COLORS: dict[str, Tuple[int, int, int]] = {
    "car":        (0,   200,  80),
    "motorcycle": (255, 140,   0),
    "bus":        (0,   120, 255),
    "truck":      (220,  50,  50),
    "bicycle":    (200, 200,   0),
}


def download_model(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model to {dest} ...")
    urllib.request.urlretrieve(url, dest)
    logger.info("Model download complete.")


class DetectionPipeline:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.45):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run scripts/download_models.py first."
            )
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=confidence_threshold,
            max_results=50,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)
        self.confidence_threshold = confidence_threshold
        logger.info(f"DetectionPipeline ready (threshold={confidence_threshold})")

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        detections: List[VehicleDetection] = []
        for det in result.detections:
            cat = det.categories[0]
            label = cat.category_name.lower()
            if label not in VEHICLE_CLASSES:
                continue
            bb = det.bounding_box
            color = list(VEHICLE_COLORS.get(label, (0, 255, 0)))
            detections.append(
                VehicleDetection(
                    label=label,
                    confidence=round(float(cat.score), 4),
                    bbox=BoundingBox(
                        x=int(bb.origin_x),
                        y=int(bb.origin_y),
                        width=int(bb.width),
                        height=int(bb.height),
                    ),
                    color=color,
                )
            )
        return detections

    def annotate(self, frame_bgr: np.ndarray, detections: List[VehicleDetection]) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1 = max(0, det.bbox.x)
            y1 = max(0, det.bbox.y)
            x2 = min(w, x1 + det.bbox.width)
            y2 = min(h, y1 + det.bbox.height)
            color_bgr = (det.color[2], det.color[1], det.color[0])

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)

            # Label background
            label_txt = f"{det.label}  {det.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_txt, font, font_scale, thickness)
            tag_y1 = max(0, y1 - th - baseline - 6)
            tag_y2 = y1
            cv2.rectangle(out, (x1, tag_y1), (x1 + tw + 8, tag_y2), color_bgr, -1)
            cv2.putText(
                out, label_txt,
                (x1 + 4, tag_y2 - baseline - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

        # HUD: total vehicle count
        count = len(detections)
        hud = f"Vehicles: {count}"
        cv2.putText(out, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)

        return out

    def close(self):
        self._detector.close()
