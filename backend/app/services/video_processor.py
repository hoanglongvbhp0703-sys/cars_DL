import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Awaitable
import cv2

from ..core.config import settings
from ..core.logging_config import get_logger
from ..schemas.detection import DetectionSummary
from ..schemas.video import JobResult, JobStatus, VideoMetadata
from .detection_pipeline import DetectionPipeline

logger = get_logger(__name__)

# In-memory job store (replace with Redis in production)
_jobs: Dict[str, JobResult] = {}
_progress: Dict[str, dict] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobResult(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
    )
    _progress[job_id] = {
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "message": "Queued",
        "elapsed_sec": 0.0,
        "eta_sec": None,
    }
    return job_id


def get_job(job_id: str) -> Optional[JobResult]:
    return _jobs.get(job_id)


def get_progress(job_id: str) -> Optional[dict]:
    return _progress.get(job_id)


def list_jobs() -> list[JobResult]:
    return list(_jobs.values())


def _probe_video(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total / fps if fps else 0.0
    return VideoMetadata(
        filename=path.name,
        size_bytes=path.stat().st_size,
        duration_sec=round(duration, 2),
        fps=round(fps, 2),
        width=w,
        height=h,
        total_frames=total,
    )


async def process_video_job(
    job_id: str,
    input_path: Path,
    progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> None:
    pipeline: Optional[DetectionPipeline] = None
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None

    try:
        _jobs[job_id].status = JobStatus.PROCESSING
        _progress[job_id].update({"status": JobStatus.PROCESSING, "message": "Initialising pipeline..."})

        meta = _probe_video(input_path)
        _jobs[job_id].video_metadata = meta

        pipeline = DetectionPipeline(settings.model_path, settings.DETECTION_CONFIDENCE)

        output_path = settings.PROCESSED_DIR / f"{job_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap = cv2.VideoCapture(str(input_path))
        writer = cv2.VideoWriter(str(output_path), fourcc, meta.fps, (meta.width, meta.height))

        vehicle_counts: Dict[str, int] = {}
        total_detections = 0
        peak_count = 0
        frame_idx = 0
        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = pipeline.detect(frame)
            annotated = pipeline.annotate(frame, detections)
            writer.write(annotated)

            for det in detections:
                vehicle_counts[det.label] = vehicle_counts.get(det.label, 0) + 1
                total_detections += 1
            peak_count = max(peak_count, len(detections))
            frame_idx += 1

            if frame_idx % 10 == 0 or frame_idx == meta.total_frames:
                elapsed = time.perf_counter() - start_time
                progress_pct = (frame_idx / meta.total_frames * 100) if meta.total_frames else 0
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                remaining = (meta.total_frames - frame_idx) / fps_proc if fps_proc > 0 else None
                progress_data = {
                    "status": JobStatus.PROCESSING,
                    "progress": round(progress_pct, 1),
                    "current_frame": frame_idx,
                    "total_frames": meta.total_frames,
                    "message": f"Processing frame {frame_idx}/{meta.total_frames}",
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": round(remaining, 1) if remaining else None,
                }
                _progress[job_id].update(progress_data)
                if progress_callback:
                    await progress_callback(progress_data)
                await asyncio.sleep(0)  # yield to event loop

        elapsed_total = time.perf_counter() - start_time
        summary = DetectionSummary(
            total_frames=meta.total_frames,
            processed_frames=frame_idx,
            total_detections=total_detections,
            vehicle_counts=vehicle_counts,
            avg_detections_per_frame=round(total_detections / frame_idx, 2) if frame_idx else 0,
            peak_count=peak_count,
            processing_fps=round(frame_idx / elapsed_total, 1) if elapsed_total else 0,
        )

        _jobs[job_id].status = JobStatus.COMPLETED
        _jobs[job_id].summary = summary
        _jobs[job_id].output_filename = output_path.name
        _jobs[job_id].completed_at = datetime.utcnow()
        _progress[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100.0,
            "message": "Processing complete",
        })
        if progress_callback:
            await progress_callback(_progress[job_id])

        logger.info(f"Job {job_id} complete: {frame_idx} frames, {total_detections} detections")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
        _jobs[job_id].status = JobStatus.FAILED
        _jobs[job_id].error = str(exc)
        _progress[job_id].update({"status": JobStatus.FAILED, "message": f"Error: {exc}"})
        if progress_callback:
            await progress_callback(_progress[job_id])

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if pipeline:
            pipeline.close()
