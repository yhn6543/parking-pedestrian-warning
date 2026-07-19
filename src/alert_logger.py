import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CSV_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "raw_risk",
    "stable_risk",
    "risk_level",
    "person_count",
    "car_count",
    "risk_person_count",
    "risk_count",
    "window_size",
    "min_risk_count",
    "image_path",
    "selected_projector_id",
    "selected_projector_name",
    "selected_projector_x",
    "selected_projector_y",
    "selected_projector_distance",
    "projector_dispatch_status",
]


def _to_basic_value(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if value is None:
        return ""
    return str(value)


class AlertLogger:
    def __init__(
        self,
        log_csv_path: str,
        alert_image_dir: str,
        cooldown_frames: int = 10,
        image_prefix: str = "alert_frame",
    ):
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be greater than or equal to 0.")

        self.log_csv_path = Path(log_csv_path)
        self.alert_image_dir = Path(alert_image_dir)
        self.cooldown_frames = int(cooldown_frames)
        self.image_prefix = str(image_prefix)
        self.last_logged_frame_index: int | None = None

        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_image_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_csv_path.exists():
            with self.log_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def should_log(self, frame_index: int, stable_risk: bool) -> bool:
        if not stable_risk:
            return False

        frame_index = int(frame_index)
        if self.last_logged_frame_index is None:
            return True

        return (frame_index - self.last_logged_frame_index) >= self.cooldown_frames

    def save_alert_image(self, frame, frame_index: int) -> str:
        output_path = self.alert_image_dir / f"{self.image_prefix}_{int(frame_index):06d}.jpg"
        success = cv2.imwrite(str(output_path), frame)
        if not success:
            raise ValueError(f"Failed to save alert image: {output_path}")
        return str(output_path)

    def append_log(self, event: dict) -> None:
        row = {column: _to_basic_value(event.get(column)) for column in CSV_COLUMNS}

        with self.log_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    def log_if_needed(self, frame, event: dict) -> dict:
        frame_index = int(event.get("frame_index", 0))
        stable_risk = bool(event.get("stable_risk", False))

        if not self.should_log(frame_index, stable_risk):
            return {"logged": False, "image_path": None}

        image_path = self.save_alert_image(frame, frame_index)
        log_event = dict(event)
        log_event["image_path"] = image_path
        self.append_log(log_event)
        self.last_logged_frame_index = frame_index

        return {"logged": True, "image_path": image_path}
