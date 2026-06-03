from pydantic import BaseModel, Field
from typing import List, Dict


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class VehicleDetection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    color: List[int] = Field(default_factory=list)


class FrameResult(BaseModel):
    frame_index: int
    timestamp_ms: float
    detections: List[VehicleDetection]
    vehicle_count: int


class DetectionSummary(BaseModel):
    total_frames: int
    processed_frames: int
    total_detections: int
    vehicle_counts: Dict[str, int]
    avg_detections_per_frame: float
    peak_count: int
    processing_fps: float
