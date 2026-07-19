import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def validate_polygon(polygon: list[list[int]]) -> bool:
    """Validate polygon coordinate format."""
    if not isinstance(polygon, list):
        raise ValueError("Polygon must be a list of [x, y] points.")

    if len(polygon) < 3:
        raise ValueError("Polygon must contain at least 3 points.")

    for index, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Polygon point #{index} must be [x, y].")

        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Polygon point #{index} must contain numeric x, y values.")

    return True


def _normalize_polygon(raw_polygon: Any) -> list[list[int]]:
    validate_polygon(raw_polygon)
    return [[int(point[0]), int(point[1])] for point in raw_polygon]


def load_danger_zone(json_path: str) -> list[list[int]]:
    """Load and validate danger zone polygon coordinates from JSON."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc

    if "danger_zone" not in data:
        raise ValueError("Missing required key: danger_zone")

    polygon = _normalize_polygon(data["danger_zone"])
    validate_polygon(polygon)
    return polygon


def is_point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    """Return True when the point is inside or on the polygon boundary."""
    validate_polygon(polygon)

    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError("Point must be in (x, y) format.")

    x, y = int(point[0]), int(point[1])
    contour = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
    return result >= 0


def draw_danger_zone(image: np.ndarray, polygon: list[list[int]], alpha: float = 0.35) -> np.ndarray:
    """Draw a semi-transparent danger zone polygon on a copy of the image."""
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    validate_polygon(polygon)
    alpha = max(0.0, min(float(alpha), 1.0))

    output = image.copy()
    overlay = output.copy()
    contour = np.array(polygon, dtype=np.int32)

    fill_color = (0, 180, 255)
    line_color = (0, 80, 255)
    cv2.fillPoly(overlay, [contour], fill_color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.polylines(output, [contour], isClosed=True, color=line_color, thickness=3)

    cv2.rectangle(output, (8, 8), (250, 42), (20, 20, 20), thickness=-1)
    cv2.putText(
        output,
        "Danger Zone ROI",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def draw_test_points(image: np.ndarray, points_with_status: list[dict]) -> np.ndarray:
    """Draw ROI point test results on a copy of the image."""
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    output = image.copy()

    for item in points_with_status:
        point = item.get("point")
        inside = bool(item.get("inside"))
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each test point must have point: [x, y].")

        x, y = int(point[0]), int(point[1])
        label = "IN" if inside else "OUT"
        color = (60, 200, 80) if inside else (60, 60, 230)

        cv2.circle(output, (x, y), 7, color, thickness=-1)
        cv2.circle(output, (x, y), 10, (255, 255, 255), thickness=2)
        cv2.putText(
            output,
            label,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return output
