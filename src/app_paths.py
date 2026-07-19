import re
import shutil
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def sanitize_filename_component(name: str) -> str:
    value = str(name or "").strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"_+", "_", value).strip(" _")
    return value or "unknown_source"


def get_video_prefix(video_path_or_name: str) -> str:
    stem = Path(str(video_path_or_name)).stem
    return sanitize_filename_component(stem)


def get_realtime_prefix(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def list_available_roi_json_files(roi_dir: str = "data/roi_zones") -> list[str]:
    path = Path(roi_dir)
    if not path.exists():
        return []
    return sorted(item.as_posix() for item in path.glob("*_danger_zone.json") if item.is_file())


def ensure_roi_json_exists_or_fallback(
    target_roi_path: str,
    fallback_roi_path: str = "data/danger_zone.json",
) -> str:
    target_path = Path(target_roi_path)
    fallback_path = Path(fallback_roi_path)

    if target_path.exists():
        return target_path.as_posix()
    if fallback_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fallback_path, target_path)
    return target_path.as_posix()
