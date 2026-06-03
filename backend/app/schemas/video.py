from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime
from .detection import DetectionSummary


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    filename: str
    size_bytes: int
    duration_sec: float
    fps: float
    width: int
    height: int
    total_frames: int


class JobCreate(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job created"


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    current_frame: int
    total_frames: int
    message: str
    elapsed_sec: float = 0.0
    eta_sec: Optional[float] = None


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    video_metadata: Optional[VideoMetadata] = None
    summary: Optional[DetectionSummary] = None
    output_filename: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class KaggleDataset(BaseModel):
    ref: str
    title: str
    size_mb: float
    download_count: int
    vote_count: int
    last_updated: str
    url: str
    tags: List[str] = []
