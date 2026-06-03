import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import io
import numpy as np
import cv2
import tempfile


@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_ready" in data


@pytest.mark.asyncio
async def test_upload_unsupported_type(client):
    content = b"fake data"
    resp = await client.post(
        "/api/v1/videos/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_upload_valid_video(client):
    # Create a minimal valid MP4 in memory using OpenCV
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, 25.0, (320, 240))
    for _ in range(10):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    video_bytes = tmp_path.read_bytes()
    tmp_path.unlink()

    with patch("app.api.v1.endpoints.video.video_processor.process_video_job", new_callable=AsyncMock):
        resp = await client.post(
            "/api/v1/videos/upload",
            files={"file": ("road.mp4", io.BytesIO(video_bytes), "video/mp4")},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_job_not_found(client):
    resp = await client.get("/api/v1/videos/nonexistent-job-id/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_jobs(client):
    resp = await client.get("/api/v1/videos/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_download_not_found(client):
    resp = await client.get("/api/v1/videos/bad-job/download")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_kaggle_search_invalid_credentials(client):
    with patch("app.services.kaggle_service.search_datasets", side_effect=RuntimeError("Auth failed")):
        resp = await client.get("/api/v1/datasets/search?q=vehicle")
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_kaggle_search_returns_results(client):
    mock_results = [
        {
            "ref": "user/vehicle-dataset",
            "title": "Vehicle Detection Dataset",
            "size_mb": 150.0,
            "download_count": 5000,
            "vote_count": 200,
            "last_updated": "2024-01-01",
            "url": "https://www.kaggle.com/datasets/user/vehicle-dataset",
            "tags": ["vehicles", "detection"],
        }
    ]
    with patch("app.services.kaggle_service.search_datasets", return_value=mock_results):
        resp = await client.get("/api/v1/datasets/search?q=vehicle")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_local_datasets_empty(client):
    resp = await client.get("/api/v1/datasets/local")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
