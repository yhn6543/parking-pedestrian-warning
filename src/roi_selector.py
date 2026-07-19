import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.danger_zone import load_danger_zone


def _coerce_coordinate(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, not bool.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return int(number)


def _normalize_point(point: Any, index: int) -> list[int]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"ROI point #{index} must be [x, y].")
    return [
        _coerce_coordinate(point[0], f"ROI point #{index} x"),
        _coerce_coordinate(point[1], f"ROI point #{index} y"),
    ]


def _normalize_points(points: list[list[int]], min_points: int = 0) -> list[list[int]]:
    if not isinstance(points, list):
        raise ValueError("ROI points must be a list.")
    if len(points) < min_points:
        raise ValueError(f"ROI polygon must contain at least {min_points} points.")
    return [_normalize_point(point, index) for index, point in enumerate(points)]


def validate_roi_points(points: list[list[int]]) -> bool:
    _normalize_points(points, min_points=3)
    return True


def save_roi_json(points: list[list[int]], output_path: str) -> str:
    validate_roi_points(points)
    normalized_points = _normalize_points(points, min_points=3)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"danger_zone": normalized_points}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def load_roi_json(json_path: str) -> list[list[int]]:
    return load_danger_zone(json_path)


def calculate_display_size(
    width: int,
    height: int,
    max_display_width: int = 960,
    max_display_height: int = 720,
) -> dict:
    width = int(width)
    height = int(height)
    max_display_width = int(max_display_width)
    max_display_height = int(max_display_height)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than 0.")
    if max_display_width <= 0 or max_display_height <= 0:
        raise ValueError("max_display_width and max_display_height must be greater than 0.")

    display_scale = min(
        max_display_width / width,
        max_display_height / height,
        1.0,
    )
    display_width = max(1, int(round(width * display_scale)))
    display_height = max(1, int(round(height * display_scale)))

    return {
        "original_width": width,
        "original_height": height,
        "display_width": display_width,
        "display_height": display_height,
        "scale_x": width / display_width,
        "scale_y": height / display_height,
    }


def resize_image_for_display(
    image,
    max_display_width: int = 960,
    max_display_height: int = 720,
) -> tuple:
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    height, width = image.shape[:2]
    display_info = calculate_display_size(
        width=width,
        height=height,
        max_display_width=max_display_width,
        max_display_height=max_display_height,
    )
    display_width = display_info["display_width"]
    display_height = display_info["display_height"]

    if display_width == width and display_height == height:
        return image.copy(), display_info

    display_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
    return display_image, display_info


def _get_positive_scale(display_info: dict, key: str) -> float:
    try:
        scale = float(display_info[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"display_info must contain numeric {key}.") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"display_info {key} must be greater than 0.")
    return scale


def _clamp_coordinate(value: int, max_value: Any | None = None) -> int:
    value = max(0, int(value))
    if max_value is None:
        return value
    try:
        upper_bound = int(max_value) - 1
    except (TypeError, ValueError) as exc:
        raise ValueError("Coordinate clamp size must be numeric.") from exc
    if upper_bound < 0:
        raise ValueError("Coordinate clamp size must be greater than 0.")
    return min(value, upper_bound)


def display_point_to_original_point(point: tuple[int, int], display_info: dict) -> list[int]:
    display_x, display_y = _normalize_point(point, 0)
    scale_x = _get_positive_scale(display_info, "scale_x")
    scale_y = _get_positive_scale(display_info, "scale_y")

    original_x = int(round(display_x * scale_x))
    original_y = int(round(display_y * scale_y))

    return [
        _clamp_coordinate(original_x, display_info.get("original_width")),
        _clamp_coordinate(original_y, display_info.get("original_height")),
    ]


def original_points_to_display_points(points: list[list[int]], display_info: dict) -> list[list[int]]:
    normalized_points = _normalize_points(points, min_points=0)
    scale_x = _get_positive_scale(display_info, "scale_x")
    scale_y = _get_positive_scale(display_info, "scale_y")

    display_points = []
    for point in normalized_points:
        display_x = int(round(point[0] / scale_x))
        display_y = int(round(point[1] / scale_y))
        display_points.append(
            [
                _clamp_coordinate(display_x, display_info.get("display_width")),
                _clamp_coordinate(display_y, display_info.get("display_height")),
            ]
        )
    return display_points


def draw_roi_preview(
    image,
    points: list[list[int]],
    closed: bool = False,
    display_info: dict | None = None,
):
    """Draw ROI points whose coordinate system matches the provided image."""
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    normalized_points = _normalize_points(points, min_points=0)
    output = image.copy()
    height, width = output.shape[:2]

    if len(normalized_points) >= 3:
        overlay = output.copy()
        contour = np.array(normalized_points, dtype=np.int32)
        cv2.fillPoly(overlay, [contour], (0, 180, 255))
        cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)

    if len(normalized_points) >= 2:
        contour = np.array(normalized_points, dtype=np.int32)
        cv2.polylines(
            output,
            [contour],
            isClosed=bool(closed and len(normalized_points) >= 3),
            color=(0, 90, 255),
            thickness=3,
        )

    for index, point in enumerate(normalized_points, start=1):
        x, y = point
        cv2.circle(output, (x, y), 7, (0, 60, 255), thickness=-1)
        cv2.circle(output, (x, y), 11, (255, 255, 255), thickness=2)

        label = str(index)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_x = min(max(0, x + 12), max(0, width - label_size[0] - 8))
        label_y = min(max(label_size[1] + baseline + 2, y - 8), height - 4)
        cv2.rectangle(
            output,
            (label_x - 4, label_y - label_size[1] - baseline - 4),
            (label_x + label_size[0] + 4, label_y + baseline),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (label_x, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    panel_lines = [
        "Left click: add point",
        "U: undo   R: reset   S: save",
        "Q/ESC: quit",
    ]
    if display_info is not None:
        panel_lines.extend(
            [
                f"Display: {display_info['display_width']}x{display_info['display_height']}",
                f"Original: {display_info['original_width']}x{display_info['original_height']}",
                f"Scale: x{display_info['scale_x']:.2f}, y{display_info['scale_y']:.2f}",
            ]
        )

    panel_width = min(max(width - 16, 1), 520)
    panel_height = 34 + len(panel_lines) * 18
    cv2.rectangle(output, (8, 8), (8 + panel_width, 8 + panel_height), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"Step 11 ROI Selector | points: {len(normalized_points)}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, line in enumerate(panel_lines):
        cv2.putText(
            output,
            line,
            (18, 52 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def create_default_roi_for_image(width: int, height: int) -> list[list[int]]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than 0.")

    return [
        [int(round(width * 0.15)), int(round(height * 0.55))],
        [int(round(width * 0.85)), int(round(height * 0.55))],
        [int(round(width * 0.90)), int(round(height * 0.95))],
        [int(round(width * 0.10)), int(round(height * 0.95))],
    ]
