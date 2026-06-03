from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # App
    PROJECT_NAME: str = "Vehicle Detection System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    # Kaggle
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""

    # Detection
    DETECTION_CONFIDENCE: float = 0.45
    MAX_VIDEO_SIZE_MB: int = 500
    ALLOWED_VIDEO_EXTENSIONS: set = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # Paths
    UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "data" / "models"

    # MediaPipe model
    MODEL_FILENAME: str = "efficientdet_lite0.tflite"
    MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
    )

    @property
    def model_path(self) -> Path:
        return self.MODELS_DIR / self.MODEL_FILENAME

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024

    def ensure_dirs(self):
        for d in (self.UPLOAD_DIR, self.PROCESSED_DIR, self.MODELS_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
