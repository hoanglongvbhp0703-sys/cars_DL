"""Download MediaPipe EfficientDet Lite model for vehicle detection."""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"\r  [{bar:<50}] {pct:.1f}%", end="", flush=True)


def main():
    dest = settings.model_path
    if dest.exists():
        print(f"Model already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {settings.MODEL_FILENAME} ...")
    print(f"  Source : {settings.MODEL_URL}")
    print(f"  Dest   : {dest}")

    urllib.request.urlretrieve(settings.MODEL_URL, dest, reporthook=show_progress)
    print(f"\nDone. Model saved to {dest}")


if __name__ == "__main__":
    main()
