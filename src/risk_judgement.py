from src.danger_zone import is_point_in_polygon


def _validate_bbox(bbox: list[int]) -> list[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2].")
    return [int(value) for value in bbox]


def get_bbox_bottom_center(bbox: list[int]) -> tuple[int, int]:
    """Return the bottom-center anchor point of [x1, y1, x2, y2]."""
    x1, _y1, x2, y2 = _validate_bbox(bbox)
    return int((x1 + x2) / 2), int(y2)


def get_bbox_center(bbox: list[int]) -> tuple[int, int]:
    """Return the center point of [x1, y1, x2, y2] for debug use."""
    x1, y1, x2, y2 = _validate_bbox(bbox)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def judge_person_risk(detection: dict, polygon: list[list[int]]) -> dict:
    """Add ROI risk fields to one detection dict."""
    enhanced_detection = dict(detection)
    class_name = enhanced_detection.get("class_name")

    if class_name != "person":
        enhanced_detection["anchor_point"] = None
        enhanced_detection["is_risk"] = False
        enhanced_detection["risk_reason"] = "not_person"
        return enhanced_detection

    anchor_x, anchor_y = get_bbox_bottom_center(enhanced_detection.get("bbox", []))
    is_risk = is_point_in_polygon((anchor_x, anchor_y), polygon)

    enhanced_detection["anchor_point"] = [int(anchor_x), int(anchor_y)]
    enhanced_detection["is_risk"] = bool(is_risk)
    enhanced_detection["risk_reason"] = (
        "person_inside_danger_zone" if is_risk else "person_outside_danger_zone"
    )
    return enhanced_detection


def evaluate_risk(detections: list[dict], polygon: list[list[int]]) -> dict:
    """Evaluate single-image pedestrian risk from detections and a danger zone."""
    enhanced_detections = [judge_person_risk(detection, polygon) for detection in detections]
    person_count = sum(1 for detection in detections if detection.get("class_name") == "person")
    car_count = sum(1 for detection in detections if detection.get("class_name") == "car")
    risk_person_count = sum(
        1
        for detection in enhanced_detections
        if detection.get("class_name") == "person" and detection.get("is_risk")
    )

    risk_detected = risk_person_count >= 1

    return {
        "risk_detected": bool(risk_detected),
        "risk_level": "warning" if risk_detected else "none",
        "person_count": int(person_count),
        "car_count": int(car_count),
        "risk_person_count": int(risk_person_count),
        "enhanced_detections": enhanced_detections,
    }


def create_mock_person_detections() -> list[dict]:
    """Create mock person detections for visual testing without YOLO inference."""
    return [
        {
            "class_name": "person",
            "confidence": 0.99,
            "bbox": [290, 250, 350, 360],
        },
        {
            "class_name": "person",
            "confidence": 0.98,
            "bbox": [35, 60, 95, 180],
        },
    ]
