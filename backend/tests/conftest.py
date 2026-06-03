import pytest
import numpy as np
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_frame() -> np.ndarray:
    """640x480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection():
    return {
        "label": "car",
        "confidence": 0.92,
        "bbox": {"x": 100, "y": 150, "width": 200, "height": 120},
        "color": [0, 200, 80],
    }


@pytest.fixture
async def client():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
