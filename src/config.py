from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_IMAGE_DIR = DATA_DIR / "sample_images"
SAMPLE_VIDEO_DIR = DATA_DIR / "sample_videos"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEBUG_DIR = OUTPUT_DIR / "debug"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_DIR = OUTPUT_DIR / "results"
ALERT_DIR = OUTPUT_DIR / "captured_alerts"

DEFAULT_MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.35
TARGET_CLASSES = ["person", "car"]


def ensure_directories() -> None:
    """Create output directories required by the current project step."""
    for directory in (DEBUG_DIR, LOG_DIR, RESULT_DIR, ALERT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
