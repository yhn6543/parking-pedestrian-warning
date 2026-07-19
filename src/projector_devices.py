from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import DEBUG_DIR, PROJECT_ROOT


PROJECTOR_DEVICE_DIR = PROJECT_ROOT / "data" / "projector_devices"
PROJECTOR_PREVIEW_DIR = DEBUG_DIR / "projector_previews"
PROJECTOR_SCHEMA_VERSION = 1
PROJECTOR_COORDINATE_SYSTEM = "original_image_pixel"


@dataclass(frozen=True)
class ProjectorDevice:
    id: str
    name: str
    x: int
    y: int
    enabled: bool = True
    endpoint: str = ""


def sanitize_projector_prefix(prefix: str) -> str:
    value = str(prefix or "").strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"_+", "_", value).strip(" _")
    return value or "unknown_source"


def build_projector_artifact_path(prefix: str) -> dict:
    safe_prefix = sanitize_projector_prefix(prefix)
    return {
        "projector_json_path": PROJECTOR_DEVICE_DIR / f"{safe_prefix}_projectors.json",
        "projector_preview_output_path": PROJECTOR_PREVIEW_DIR / f"{safe_prefix}_projectors_preview.jpg",
    }


def create_empty_projector_config(image_size: dict | None = None) -> dict:
    width = int((image_size or {}).get("width", 0) or 0)
    height = int((image_size or {}).get("height", 0) or 0)
    return {
        "schema_version": PROJECTOR_SCHEMA_VERSION,
        "image_size": {"width": width, "height": height},
        "coordinate_system": PROJECTOR_COORDINATE_SYSTEM,
        "devices": [],
    }


def _coerce_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, not bool.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return int(round(number))


def _normalize_image_size(image_size: Any) -> dict:
    if image_size is None:
        return {"width": 0, "height": 0}
    if not isinstance(image_size, dict):
        raise ValueError("image_size must be a dict.")
    width = _coerce_int(image_size.get("width", 0), "image_size.width")
    height = _coerce_int(image_size.get("height", 0), "image_size.height")
    if width < 0 or height < 0:
        raise ValueError("image_size width and height must be non-negative.")
    return {"width": width, "height": height}


def _normalize_device(device: ProjectorDevice | dict, index: int) -> dict:
    if isinstance(device, ProjectorDevice):
        data = asdict(device)
    elif isinstance(device, dict):
        data = dict(device)
    else:
        raise ValueError(f"projector device #{index} must be a dict.")

    device_id = str(data.get("id") or f"projector_{index + 1}").strip()
    name = str(data.get("name") or f"Projector {index + 1}").strip()
    if not device_id:
        device_id = f"projector_{index + 1}"
    if not name:
        name = f"Projector {index + 1}"

    return {
        "id": device_id,
        "name": name,
        "x": _coerce_int(data.get("x"), f"devices[{index}].x"),
        "y": _coerce_int(data.get("y"), f"devices[{index}].y"),
        "enabled": bool(data.get("enabled", True)),
        "endpoint": str(data.get("endpoint", "") or ""),
    }


def normalize_projector_devices(devices: Any) -> list[dict]:
    if devices is None:
        return []
    if not isinstance(devices, list):
        raise ValueError("devices must be a list.")
    return [_normalize_device(device, index) for index, device in enumerate(devices)]


def validate_projector_config(config: dict) -> bool:
    if not isinstance(config, dict):
        raise ValueError("projector config must be a dict.")

    schema_version = int(config.get("schema_version", PROJECTOR_SCHEMA_VERSION))
    if schema_version != PROJECTOR_SCHEMA_VERSION:
        raise ValueError(f"Unsupported projector schema_version: {schema_version}")

    coordinate_system = str(config.get("coordinate_system", PROJECTOR_COORDINATE_SYSTEM))
    if coordinate_system != PROJECTOR_COORDINATE_SYSTEM:
        raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")

    _normalize_image_size(config.get("image_size"))
    normalize_projector_devices(config.get("devices", []))
    return True


def normalize_projector_config(config: dict) -> dict:
    validate_projector_config(config)
    return {
        "schema_version": PROJECTOR_SCHEMA_VERSION,
        "image_size": _normalize_image_size(config.get("image_size")),
        "coordinate_system": PROJECTOR_COORDINATE_SYSTEM,
        "devices": normalize_projector_devices(config.get("devices", [])),
    }


def load_projector_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return create_empty_projector_config()
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid projector JSON: {config_path}") from exc
    return normalize_projector_config(raw_config)


def save_projector_config(path: str | Path, devices: list[dict], image_size: dict | None = None) -> str:
    config = create_empty_projector_config(image_size)
    config["devices"] = normalize_projector_devices(devices)
    validate_projector_config(config)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def enabled_projector_devices(devices: list[dict]) -> list[dict]:
    return [device for device in normalize_projector_devices(devices) if bool(device.get("enabled", True))]


def _normalize_point(point_xy: Any) -> tuple[int, int]:
    if not isinstance(point_xy, (list, tuple)) or len(point_xy) != 2:
        raise ValueError("point_xy must be [x, y].")
    return _coerce_int(point_xy[0], "point_xy.x"), _coerce_int(point_xy[1], "point_xy.y")


def find_nearest_projector(point_xy: list[int] | tuple[int, int], devices: list[dict]) -> dict | None:
    point_x, point_y = _normalize_point(point_xy)
    nearest = None
    nearest_distance = None

    for device in enabled_projector_devices(devices):
        dx = int(device["x"]) - point_x
        dy = int(device["y"]) - point_y
        distance = math.hypot(dx, dy)
        if nearest_distance is None or distance < nearest_distance:
            nearest = dict(device)
            nearest_distance = distance

    if nearest is None:
        return None
    nearest["distance"] = round(float(nearest_distance), 4)
    return nearest


def find_nearest_projectors_for_risk_persons(
    risk_person_points: list[list[int] | tuple[int, int]],
    devices: list[dict],
) -> list[dict]:
    results = []
    for person_index, point in enumerate(risk_person_points):
        nearest = find_nearest_projector(point, devices)
        if nearest is None:
            continue
        results.append(
            {
                "person_index": int(person_index),
                "person_point": [int(point[0]), int(point[1])],
                "projector_id": nearest["id"],
                "projector_name": nearest["name"],
                "projector_x": int(nearest["x"]),
                "projector_y": int(nearest["y"]),
                "distance": float(nearest["distance"]),
            }
        )
    return results


def selected_projector_from_assignments(assignments: list[dict], devices: list[dict]) -> dict | None:
    if not assignments:
        return None

    best_assignment = min(assignments, key=lambda item: float(item["distance"]))
    by_id = {device["id"]: device for device in normalize_projector_devices(devices)}
    selected = dict(by_id.get(best_assignment["projector_id"], {}))
    if not selected:
        return None
    selected["distance"] = float(best_assignment["distance"])
    return selected


def scale_projector_devices_to_frame(
    devices: list[dict],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[dict]:
    source_width = int(source_width)
    source_height = int(source_height)
    target_width = int(target_width)
    target_height = int(target_height)
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        return normalize_projector_devices(devices)

    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    scaled = []
    for device in normalize_projector_devices(devices):
        item = dict(device)
        item["original_x"] = int(device["x"])
        item["original_y"] = int(device["y"])
        item["x"] = int(round(int(device["x"]) * scale_x))
        item["y"] = int(round(int(device["y"]) * scale_y))
        scaled.append(item)
    return scaled
