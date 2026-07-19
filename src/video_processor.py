import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_FPS = 20.0


def _safe_fps(fps: float) -> float:
    if fps is None:
        return DEFAULT_FPS
    fps = float(fps)
    if fps <= 0 or not math.isfinite(fps):
        return DEFAULT_FPS
    return fps


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


def get_video_info(video_path: str) -> dict:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _safe_fps(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions: {path}")

    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frame_count": int(frame_count),
    }


def create_video_writer(output_path: str, width: int, height: int, fps: float):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, _safe_fps(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise ValueError(f"Failed to create video writer: {path}")

    return writer


def summarize_frame_result(frame_index: int, risk_result: dict) -> dict:
    return {
        "frame_index": int(frame_index),
        "risk_detected": bool(risk_result.get("risk_detected", False)),
        "risk_level": str(risk_result.get("risk_level", "none")),
        "person_count": int(risk_result.get("person_count", 0)),
        "car_count": int(risk_result.get("car_count", 0)),
        "risk_person_count": int(risk_result.get("risk_person_count", 0)),
    }


def save_video_summary(summary: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def create_dummy_video_from_image(
    image_path: str,
    output_video_path: str,
    frame_count: int = 30,
    fps: float = 10.0,
) -> str:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"File not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise ValueError(f"Failed to read image file: {image_file}")

    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than 0.")

    height, width = image.shape[:2]
    writer = create_video_writer(output_video_path, width, height, fps)
    try:
        for frame_index in range(frame_count):
            frame = image.copy()
            cv2.putText(
                frame,
                f"Frame {frame_index}",
                (width - 160, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    return str(Path(output_video_path))
