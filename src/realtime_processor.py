import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@dataclass
class RealtimeStats:
    processed_frames: int = 0
    raw_risk_frames: int = 0
    stable_risk_frames: int = 0
    logged_alert_count: int = 0
    max_risk_person_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    last_fps: float = 0.0

    def update(self, risk_result: dict, temporal_result: dict, logged: bool) -> None:
        self.processed_frames += 1

        if bool(temporal_result.get("raw_risk", risk_result.get("risk_detected", False))):
            self.raw_risk_frames += 1
        if bool(temporal_result.get("stable_risk", False)):
            self.stable_risk_frames += 1
        if logged:
            self.logged_alert_count += 1

        self.max_risk_person_count = max(
            self.max_risk_person_count,
            int(risk_result.get("risk_person_count", 0)),
        )

    def calculate_fps(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        if self.processed_frames <= 0 or elapsed <= 0:
            self.last_fps = 0.0
        else:
            self.last_fps = float(self.processed_frames / elapsed)
        return self.last_fps

    def to_dict(self) -> dict:
        return {
            "processed_frames": int(self.processed_frames),
            "raw_risk_frames": int(self.raw_risk_frames),
            "stable_risk_frames": int(self.stable_risk_frames),
            "logged_alert_count": int(self.logged_alert_count),
            "max_risk_person_count": int(self.max_risk_person_count),
            "last_fps": float(self.last_fps),
        }


def open_realtime_source(
    source: str,
    camera_index: int = 0,
    video_path: str | None = None,
):
    source_name = str(source).lower()
    if source_name == "webcam":
        capture = cv2.VideoCapture(int(camera_index))
        description = f"webcam index {camera_index}"
    elif source_name == "video":
        if not video_path:
            raise ValueError("video_path is required when source is 'video'.")
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        description = str(path)
    else:
        raise ValueError("source must be either 'webcam' or 'video'.")

    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Failed to open realtime source: {description}")

    return capture


def resize_frame_keep_aspect(frame, target_width: int | None):
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("Invalid frame: expected an OpenCV BGR numpy array.")
    if target_width is None:
        return frame

    target_width = int(target_width)
    if target_width <= 0:
        raise ValueError("target_width must be greater than 0.")

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions.")

    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def scale_polygon_to_frame(
    polygon: list[list[int]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[list[int]]:
    source_width = int(source_width)
    source_height = int(source_height)
    target_width = int(target_width)
    target_height = int(target_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_width and source_height must be greater than 0.")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be greater than 0.")

    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    scaled_polygon = []
    for point in polygon:
        if len(point) != 2:
            raise ValueError("Each polygon point must contain exactly two values.")
        x = int(round(float(point[0]) * scale_x))
        y = int(round(float(point[1]) * scale_y))
        x = min(max(x, 0), target_width - 1)
        y = min(max(y, 0), target_height - 1)
        scaled_polygon.append([x, y])
    return scaled_polygon


def get_image_size(image_path: str | Path) -> tuple[int, int]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image file: {path}")

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {path}")
    return int(width), int(height)


def draw_realtime_overlay(frame, frame_index: int, fps: float, source_name: str):
    if frame is None or not hasattr(frame, "copy"):
        raise ValueError("Invalid frame: expected an OpenCV BGR numpy array.")

    output = frame.copy()
    height, _width = output.shape[:2]
    panel_x = 8
    panel_height = 134
    panel_y = max(8, height - panel_height - 8)
    panel_width = 330

    cv2.rectangle(
        output,
        (panel_x, panel_y),
        (panel_x + panel_width, min(panel_y + panel_height, height - 1)),
        (20, 20, 20),
        thickness=-1,
    )

    lines = [
        "Step 10 Realtime Mode",
        f"Source: {source_name}",
        f"Frame: {int(frame_index)}",
        f"FPS: {float(fps):.2f}",
        "Press Q to quit",
    ]
    for index, line in enumerate(lines):
        color = (0, 255, 255) if index == 0 else (255, 255, 255)
        cv2.putText(
            output,
            line,
            (panel_x + 10, panel_y + 24 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def save_realtime_summary(summary: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)
