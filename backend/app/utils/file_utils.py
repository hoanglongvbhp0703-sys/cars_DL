import shutil
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException, status

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)


def validate_video_file(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(settings.ALLOWED_VIDEO_EXTENSIONS)}",
        )


async def save_upload(file: UploadFile, job_id: str) -> Path:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    dest = settings.UPLOAD_DIR / f"{job_id}{suffix}"
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(8 * 1024 * 1024):  # 8 MB chunks
                if dest.stat().st_size > settings.max_upload_bytes:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {settings.MAX_VIDEO_SIZE_MB} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.error(f"Failed to save upload: {exc}")
        raise HTTPException(status_code=500, detail="File save failed") from exc
    return dest


def cleanup_job_files(job_id: str) -> None:
    for ext in settings.ALLOWED_VIDEO_EXTENSIONS:
        p = settings.UPLOAD_DIR / f"{job_id}{ext}"
        p.unlink(missing_ok=True)
    (settings.PROCESSED_DIR / f"{job_id}_processed.mp4").unlink(missing_ok=True)


def get_processed_path(job_id: str) -> Optional[Path]:
    p = settings.PROCESSED_DIR / f"{job_id}_processed.mp4"
    return p if p.exists() else None
