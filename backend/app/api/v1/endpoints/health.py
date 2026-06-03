from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    model_ready = settings.model_path.exists()
    return {
        "status": "ok",
        "version": settings.VERSION,
        "model_ready": model_ready,
        "model_path": str(settings.model_path),
    }
