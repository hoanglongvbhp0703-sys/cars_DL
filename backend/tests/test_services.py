import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.schemas.detection import VehicleDetection, BoundingBox


def make_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_detection(label="car", x=50, y=60, w=200, h=120, conf=0.85):
    return VehicleDetection(
        label=label,
        confidence=conf,
        bbox=BoundingBox(x=x, y=y, width=w, height=h),
        color=[0, 200, 80],
    )


class TestDetectionPipeline:
    def test_annotate_draws_bbox(self):
        from app.services.detection_pipeline import DetectionPipeline

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("app.services.detection_pipeline.vision.ObjectDetector.create_from_options"):
            pipeline = DetectionPipeline.__new__(DetectionPipeline)
            pipeline.confidence_threshold = 0.45
            pipeline._detector = MagicMock()

        frame = make_frame()
        det = make_detection()
        result = pipeline.annotate(frame, [det])

        assert result.shape == frame.shape
        # Pixels in bbox area should differ from black background
        assert result[90, 150].sum() > 0 or result[60, 50].sum() > 0

    def test_annotate_empty_detections_returns_hud(self):
        from app.services.detection_pipeline import DetectionPipeline

        with patch("app.services.detection_pipeline.vision.ObjectDetector.create_from_options"):
            pipeline = DetectionPipeline.__new__(DetectionPipeline)
            pipeline._detector = MagicMock()

        frame = make_frame()
        result = pipeline.annotate(frame, [])
        assert result.shape == frame.shape

    def test_vehicle_classes_filter(self):
        from app.services.detection_pipeline import VEHICLE_CLASSES
        assert "car" in VEHICLE_CLASSES
        assert "truck" in VEHICLE_CLASSES
        assert "bus" in VEHICLE_CLASSES
        assert "motorcycle" in VEHICLE_CLASSES
        assert "person" not in VEHICLE_CLASSES

    def test_vehicle_colors_defined(self):
        from app.services.detection_pipeline import VEHICLE_COLORS
        for cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
            assert cls in VEHICLE_COLORS
            color = VEHICLE_COLORS[cls]
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


class TestVideoProcessor:
    def test_create_job_returns_id(self):
        from app.services.video_processor import create_job, get_job
        job_id = create_job()
        assert job_id
        job = get_job(job_id)
        assert job is not None
        assert job.status.value == "queued"

    def test_get_nonexistent_job(self):
        from app.services.video_processor import get_job
        assert get_job("does-not-exist") is None

    def test_get_progress_initial_state(self):
        from app.services.video_processor import create_job, get_progress
        job_id = create_job()
        progress = get_progress(job_id)
        assert progress is not None
        assert progress["progress"] == 0.0
        assert progress["status"].value == "queued"


class TestFileUtils:
    def test_validate_rejects_non_video(self):
        from fastapi import UploadFile
        from app.utils.file_utils import validate_video_file
        import io
        from fastapi import HTTPException

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "document.pdf"
        with pytest.raises(HTTPException) as exc:
            validate_video_file(mock_file)
        assert exc.value.status_code == 415

    def test_validate_accepts_mp4(self):
        from fastapi import UploadFile
        from app.utils.file_utils import validate_video_file

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "road_video.mp4"
        validate_video_file(mock_file)  # Should not raise

    def test_validate_accepts_all_supported(self):
        from fastapi import UploadFile
        from app.utils.file_utils import validate_video_file

        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = f"video{ext}"
            validate_video_file(mock_file)
