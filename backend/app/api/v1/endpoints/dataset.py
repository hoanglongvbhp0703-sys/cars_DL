from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.schemas.video import KaggleDataset
from app.services import kaggle_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/search", response_model=List[KaggleDataset])
async def search_kaggle_datasets(
    q: str = Query(..., description="Search keyword, e.g. 'vehicle detection'"),
    limit: int = Query(20, ge=1, le=50),
):
    try:
        return kaggle_service.search_datasets(q, max_results=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/download")
async def download_dataset(ref: str = Query(..., description="dataset ref, e.g. owner/name")):
    if "/" not in ref:
        raise HTTPException(status_code=400, detail="ref must be 'owner/dataset-name'")
    try:
        path = kaggle_service.download_dataset(ref)
        return {"message": "Download complete", "path": str(path)}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/local")
async def list_local_datasets():
    return kaggle_service.list_downloaded_datasets()
