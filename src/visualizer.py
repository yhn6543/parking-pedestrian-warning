import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


BOX_COLORS = {
    "person": (40, 40, 220),
    "car": (220, 120, 40),
}
DEFAULT_BOX_COLOR = (80, 220, 80)
PROJECTOR_COLOR = (255, 180, 0)
SELECTED_PROJECTOR_COLOR = (0, 255, 255)
COMPACT_RISK_PANEL_WIDTH = 360
COMPACT_RISK_PANEL_HEIGHT = 78


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw detection boxes and a compact count summary on a copy of the image."""
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    output = image.copy()
    counts = Counter(detection["class_name"] for detection in detections)

    summary = f"person: {counts.get('person', 0)}  car: {counts.get('car', 0)}  total: {len(detections)}"
    cv2.rectangle(output, (8, 8), (420, 42), (20, 20, 20), thickness=-1)
    cv2.putText(
        output,
        summary,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        color = BOX_COLORS.get(class_name, DEFAULT_BOX_COLOR)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)

        label = f"{class_name} {confidence:.4f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + baseline + 4)
        cv2.rectangle(
            output,
            (x1, label_y - label_size[1] - baseline - 4),
            (x1 + label_size[0] + 8, label_y + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def save_detections_json(detections: list[dict], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(
            json.dumps(detections, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(f"Failed to save detections JSON: {path}") from exc

    return str(path)


def _selected_projector_ids(selected_projectors) -> set[str]:
    selected_ids = set()
    for item in selected_projectors or []:
        projector_id = item.get("projector_id") or item.get("id")
        if projector_id:
            selected_ids.add(str(projector_id))
    return selected_ids


def draw_projector_devices(
    image: np.ndarray,
    projector_devices: list[dict] | None = None,
    selected_projectors: list[dict] | None = None,
) -> np.ndarray:
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    output = image.copy()
    if not projector_devices:
        return output

    selected_ids = _selected_projector_ids(selected_projectors)
    height, width = output.shape[:2]

    for index, device in enumerate(projector_devices, start=1):
        if not bool(device.get("enabled", True)):
            continue
        x = int(device.get("x", 0))
        y = int(device.get("y", 0))
        projector_id = str(device.get("id") or f"projector_{index}")
        color = SELECTED_PROJECTOR_COLOR if projector_id in selected_ids else PROJECTOR_COLOR

        cv2.circle(output, (x, y), 7, color, thickness=-1)
        cv2.circle(output, (x, y), 11, (255, 255, 255), thickness=2)

        label = projector_id
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
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
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def _get_value(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def get_near_projector_label(risk_result: dict) -> str:
    selected_projector = risk_result.get("selected_projector") or {}
    projector_name = _get_value(selected_projector, "name")
    projector_id = _get_value(selected_projector, "id")
    if projector_name:
        return str(projector_name)
    if projector_id:
        return str(projector_id)

    selected_projectors = risk_result.get("selected_projectors") or []
    if selected_projectors:
        first_projector = selected_projectors[0] or {}
        projector_name = _get_value(first_projector, "projector_name")
        projector_id = _get_value(first_projector, "projector_id")
        if projector_name:
            return str(projector_name)
        if projector_id:
            return str(projector_id)

    return "-"


def build_compact_risk_overlay_lines(risk_result: dict) -> list[str]:
    risk_detected = bool(risk_result.get("risk_detected", False))
    near_projector = get_near_projector_label(risk_result) if risk_detected else "-"
    return [
        f"Risk Detected: {'YES' if risk_detected else 'NO'}",
        f"Near Projector: {near_projector}",
    ]


def draw_risk_assessment(
    image: np.ndarray,
    risk_result: dict,
    polygon: list[list[int]],
    projector_devices: list[dict] | None = None,
    selected_projectors: list[dict] | None = None,
    compact_summary: bool = False,
) -> np.ndarray:
    """Draw ROI, detections, anchor points, and risk summary on a copy."""
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    output = image.copy()
    overlay = output.copy()
    contour = np.array(polygon, dtype=np.int32)

    cv2.fillPoly(overlay, [contour], (0, 180, 255))
    cv2.addWeighted(overlay, 0.28, output, 0.72, 0, output)
    cv2.polylines(output, [contour], isClosed=True, color=(0, 80, 255), thickness=3)

    for detection in risk_result.get("enhanced_detections", []):
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(value) for value in bbox]
        class_name = detection.get("class_name", "unknown")
        confidence = float(detection.get("confidence", 0.0))
        is_risk = bool(detection.get("is_risk", False))

        if class_name == "person" and is_risk:
            color = (30, 30, 230)
            label = f"DANGER {confidence:.4f}"
        elif class_name == "person":
            color = (60, 200, 80)
            label = f"person {confidence:.4f}"
        elif class_name == "car":
            color = (220, 120, 40)
            label = f"car {confidence:.4f}"
        else:
            color = DEFAULT_BOX_COLOR
            label = f"{class_name} {confidence:.4f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + baseline + 4)
        cv2.rectangle(
            output,
            (x1, label_y - label_size[1] - baseline - 4),
            (x1 + label_size[0] + 8, label_y + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        anchor_point = detection.get("anchor_point")
        if class_name == "person" and isinstance(anchor_point, list) and len(anchor_point) == 2:
            ax, ay = int(anchor_point[0]), int(anchor_point[1])
            cv2.circle(output, (ax, ay), 6, color, thickness=-1)
            cv2.circle(output, (ax, ay), 9, (255, 255, 255), thickness=2)

    risk_detected = bool(risk_result.get("risk_detected", False))
    if compact_summary:
        summary_risk_result = (
            risk_result
            if selected_projectors is None
            else {**risk_result, "selected_projectors": selected_projectors}
        )
        summary_lines = build_compact_risk_overlay_lines(summary_risk_result)
        panel_height = COMPACT_RISK_PANEL_HEIGHT
    else:
        summary_lines = [
            f"Risk Detected: {risk_detected}",
            f"Risk Level: {risk_result.get('risk_level', 'none')}",
            f"Person: {risk_result.get('person_count', 0)}",
            f"Car: {risk_result.get('car_count', 0)}",
            f"Risk Person: {risk_result.get('risk_person_count', 0)}",
        ]
        panel_height = 168 if risk_detected else 132

    cv2.rectangle(output, (8, 8), (COMPACT_RISK_PANEL_WIDTH, panel_height), (20, 20, 20), thickness=-1)
    for index, line in enumerate(summary_lines):
        cv2.putText(
            output,
            line,
            (16, 32 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if risk_detected and not compact_summary:
        cv2.putText(
            output,
            "WARNING: Pedestrian in Danger Zone",
            (16, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (40, 40, 255),
            2,
            cv2.LINE_AA,
        )

    return draw_projector_devices(
        output,
        projector_devices=projector_devices,
        selected_projectors=selected_projectors or risk_result.get("selected_projectors"),
    )


def draw_temporal_status(image: np.ndarray, temporal_result: dict) -> np.ndarray:
    """Draw temporal raw/stable risk state on a copy of the image."""
    if image is None or not hasattr(image, "copy"):
        raise ValueError("Invalid image: expected an OpenCV BGR numpy array.")

    output = image.copy()
    height, width = output.shape[:2]
    panel_width = min(320, max(240, width - 16))
    panel_x = max(8, width - panel_width - 8)
    panel_y = 8
    panel_height = 152

    raw_risk = bool(temporal_result.get("raw_risk", False))
    stable_risk = bool(temporal_result.get("stable_risk", False))
    risk_count = int(temporal_result.get("risk_count", 0))
    window_size = int(temporal_result.get("window_size", 0))
    min_risk_count = int(temporal_result.get("min_risk_count", 0))

    cv2.rectangle(
        output,
        (panel_x, panel_y),
        (panel_x + panel_width, min(panel_y + panel_height, height - 1)),
        (20, 20, 20),
        thickness=-1,
    )

    lines = [
        f"Raw Risk: {raw_risk}",
        f"Stable Risk: {stable_risk}",
        f"Risk Count: {risk_count} / {window_size}",
        f"Rule: {min_risk_count} / {window_size}",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (panel_x + 10, panel_y + 26 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if stable_risk:
        message = "STABLE WARNING: Pedestrian Risk"
        color = (30, 30, 255)
    elif raw_risk:
        message = "Raw Risk Detected - Waiting"
        color = (0, 200, 255)
    else:
        message = "No Stable Warning"
        color = (120, 220, 120)

    cv2.putText(
        output,
        message,
        (panel_x + 10, panel_y + 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )

    return output
