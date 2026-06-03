from fastapi import APIRouter
from .endpoints import health, video, dataset

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(video.router)
api_router.include_router(dataset.router)
