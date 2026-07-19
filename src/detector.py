import os
from typing import Iterable

import numpy as np

from src.config import (
    CONF_THRESHOLD,
    DEFAULT_MODEL_NAME,
    LOG_DIR,
    TARGET_CLASSES,
)


REQUIRED_RISK_CLASSES = {"person"}
OPTIONAL_DISPLAY_CLASSES = {"car", "truck", "bus", "motorcycle"}
DEFAULT_DETECTION_CLASSES = tuple(
    dict.fromkeys(["person", *TARGET_CLASSES, "truck", "bus", "motorcycle"])
)


def _format_bbox(bbox: Iterable[float]) -> list[int]:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"bbox must contain 4 values, got {len(values)}")
    return [int(round(float(value))) for value in values]


def validate_detection_format(detection: dict) -> bool:
    """Return True when a detection dict matches the public output contract."""
    required_keys = {"class_name", "confidence", "bbox"}
    if not required_keys.issubset(detection.keys()):
        return False

    if not isinstance(detection["class_name"], str):
        return False

    if not isinstance(detection["confidence"], float):
        return False

    bbox = detection["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    return all(isinstance(value, int) for value in bbox)


def filter_detections(
    raw_detections: Iterable[dict],
    target_classes: Iterable[str] = DEFAULT_DETECTION_CLASSES,
    conf_threshold: float = CONF_THRESHOLD,
) -> list[dict]:
    """Filter raw detections by class and confidence, then normalize output."""
    target_class_set = set(target_classes)
    filtered = []

    for raw_detection in raw_detections:
        class_name = str(raw_detection.get("class_name", ""))
        confidence = float(raw_detection.get("confidence", 0.0))

        if class_name not in target_class_set:
            continue
        if confidence < conf_threshold:
            continue

        detection = {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": _format_bbox(raw_detection.get("bbox", [])),
        }

        if not validate_detection_format(detection):
            raise ValueError(f"Invalid detection format: {detection}")

        filtered.append(detection)

    return filtered


class ObjectDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf_threshold: float = CONF_THRESHOLD,
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold

        ultralytics_settings_dir = LOG_DIR / "ultralytics"
        ultralytics_settings_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_settings_dir))

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load YOLO model. Check ultralytics installation, "
                "network access, or model file download availability."
            ) from exc

    def detect(self, image: np.ndarray) -> list[dict]:
        if image is None or not hasattr(image, "shape"):
            raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

        try:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
        except Exception as exc:
            raise RuntimeError("Failed to run YOLO inference on the input image.") from exc

        raw_detections = []
        for result in results:
            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                bbox = box.xyxy[0].tolist()

                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                else:
                    class_name = names[class_id]

                raw_detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                )

        return filter_detections(raw_detections, DEFAULT_DETECTION_CLASSES, self.conf_threshold)
