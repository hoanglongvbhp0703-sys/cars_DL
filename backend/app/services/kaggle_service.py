import os
import json
import zipfile
from pathlib import Path
from typing import List, Optional

from ..core.config import settings
from ..core.logging_config import get_logger
from ..schemas.video import KaggleDataset

logger = get_logger(__name__)


def _configure_kaggle_env() -> None:
    os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY


def search_datasets(query: str, max_results: int = 20) -> List[KaggleDataset]:
    _configure_kaggle_env()
    try:
        import kaggle
        kaggle.api.authenticate()
        datasets = kaggle.api.dataset_list(search=query, max_size=None, page_size=max_results)
        results: List[KaggleDataset] = []
        for ds in datasets:
            try:
                size_mb = round(getattr(ds, "totalBytes", 0) / (1024 * 1024), 1)
                results.append(
                    KaggleDataset(
                        ref=str(ds.ref),
                        title=str(ds.title),
                        size_mb=size_mb,
                        download_count=int(getattr(ds, "downloadCount", 0)),
                        vote_count=int(getattr(ds, "voteCount", 0)),
                        last_updated=str(getattr(ds, "lastUpdated", "")),
                        url=f"https://www.kaggle.com/datasets/{ds.ref}",
                        tags=[str(t) for t in getattr(ds, "tags", [])],
                    )
                )
            except Exception as e:
                logger.warning(f"Skipping dataset due to parse error: {e}")
        return results
    except Exception as exc:
        logger.error(f"Kaggle search failed: {exc}")
        raise RuntimeError(f"Kaggle API error: {exc}") from exc


def download_dataset(dataset_ref: str, dest_dir: Optional[Path] = None) -> Path:
    _configure_kaggle_env()
    if dest_dir is None:
        owner, name = dataset_ref.split("/")
        dest_dir = settings.UPLOAD_DIR / "datasets" / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
        logger.info(f"Downloading dataset {dataset_ref} -> {dest_dir}")
        kaggle.api.dataset_download_files(
            dataset_ref,
            path=str(dest_dir),
            unzip=True,
            quiet=False,
        )
        logger.info(f"Dataset downloaded to {dest_dir}")
        return dest_dir
    except Exception as exc:
        logger.error(f"Kaggle download failed: {exc}")
        raise RuntimeError(f"Kaggle download error: {exc}") from exc


def list_downloaded_datasets() -> List[dict]:
    datasets_dir = settings.UPLOAD_DIR / "datasets"
    if not datasets_dir.exists():
        return []
    result = []
    for d in datasets_dir.iterdir():
        if d.is_dir():
            files = list(d.rglob("*"))
            total_bytes = sum(f.stat().st_size for f in files if f.is_file())
            result.append({
                "name": d.name,
                "path": str(d),
                "file_count": len([f for f in files if f.is_file()]),
                "size_mb": round(total_bytes / (1024 * 1024), 1),
            })
    return result
