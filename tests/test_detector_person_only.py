import numpy as np

from src.detector import DEFAULT_DETECTION_CLASSES, filter_detections
from src.risk_judgement import evaluate_risk
from src.visualizer import draw_risk_assessment


POLYGON = [[100, 250], [540, 250], [600, 470], [80, 470]]


def test_default_detection_classes_require_person_but_allow_optional_vehicle_display() -> None:
    assert "person" in DEFAULT_DETECTION_CLASSES
    assert "car" in DEFAULT_DETECTION_CLASSES
    assert "truck" in DEFAULT_DETECTION_CLASSES


def test_person_only_filtering_does_not_require_car_class() -> None:
    raw_detections = [
        {"class_name": "person", "confidence": 0.82, "bbox": [290.2, 250.1, 350.0, 360.4]},
    ]

    detections = filter_detections(raw_detections, target_classes=["person"], conf_threshold=0.25)

    assert detections == [
        {"class_name": "person", "confidence": 0.82, "bbox": [290, 250, 350, 360]},
    ]


def test_person_only_detection_can_drive_risk_and_visualization_without_car() -> None:
    detections = [
        {"class_name": "person", "confidence": 0.82, "bbox": [290, 250, 350, 360]},
    ]
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    risk_result = evaluate_risk(detections, POLYGON)
    output = draw_risk_assessment(image, risk_result, POLYGON)

    assert risk_result["risk_detected"] is True
    assert risk_result["person_count"] == 1
    assert risk_result["car_count"] == 0
    assert risk_result["risk_person_count"] == 1
    assert output.shape == image.shape
