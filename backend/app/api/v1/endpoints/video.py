import asyncio
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
import json

from app.core.config import settings
from app.schemas.video import JobCreate, JobResult, JobStatus
from app.services import video_processor
from app.utils.file_utils import validate_video_file, save_upload, get_processed_path

router = APIRouter(prefix="/videos", tags=["videos"])

# Active WebSocket connections per job_id
_ws_clients: dict[str, list[WebSocket]] = {}


async def _broadcast(job_id: str, data: dict) -> None:
    clients = _ws_clients.get(job_id, [])
    dead = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


@router.post("/upload", response_model=JobCreate, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    validate_video_file(file)
    job_id = video_processor.create_job()
    input_path = await save_upload(file, job_id)

    async def _run():
        await video_processor.process_video_job(
            job_id, input_path, progress_callback=lambda d: _broadcast(job_id, d)
        )

    background_tasks.add_task(_run)
    return JobCreate(job_id=job_id, status=JobStatus.QUEUED, message="Video uploaded, processing started")


@router.websocket("/ws/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    _ws_clients.setdefault(job_id, []).append(websocket)

    # Send current state immediately on connect
    progress = video_processor.get_progress(job_id)
    if progress:
        await websocket.send_text(json.dumps(progress))

    try:
        while True:
            await asyncio.sleep(0.5)
            job = video_processor.get_job(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
    except WebSocketDisconnect:
        pass
    finally:
        clients = _ws_clients.get(job_id, [])
        if websocket in clients:
            clients.remove(websocket)


@router.get("/{job_id}/status", response_model=JobResult)
async def get_job_status(job_id: str):
    job = video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get("/{job_id}/download")
async def download_processed_video(job_id: str):
    job = video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job not complete (status={job.status})")
    path = get_processed_path(job_id)
    if not path:
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=path.name,
    )


@router.get("/", response_model=List[JobResult])
async def list_jobs():
    return video_processor.list_jobs()
