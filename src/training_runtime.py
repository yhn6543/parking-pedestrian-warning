import os
from pathlib import Path


def configure_ultralytics_runtime(project_root: str | Path | None = None) -> Path:
    """Keep Ultralytics runtime files inside this workspace."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config_dir = root / "outputs" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir
