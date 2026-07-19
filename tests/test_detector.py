from src.config import TARGET_CLASSES
from src.detector import filter_detections, validate_detection_format


def test_validate_detection_format_accepts_expected_detection_dict() -> None:
    detection = {
        "class_name": "person",
        "confidence": 0.8754,
        "bbox": [10, 20, 30, 40],
    }

    assert validate_detection_format(detection)


def test_filter_detections_returns_int_bbox_values() -> None:
    raw_detections = [
        {
            "class_name": "car",
            "confidence": 0.91,
            "bbox": [10.2, 20.7, 80.4, 120.9],
        }
    ]

    detections = filter_detections(raw_detections, TARGET_CLASSES, 0.35)

    assert detections[0]["bbox"] == [10, 21, 80, 121]
    assert all(isinstance(value, int) for value in detections[0]["bbox"])


def test_filter_detections_applies_confidence_threshold() -> None:
    raw_detections = [
        {"class_name": "person", "confidence": 0.34, "bbox": [1, 2, 3, 4]},
        {"class_name": "person", "confidence": 0.35, "bbox": [5, 6, 7, 8]},
    ]

    detections = filter_detections(raw_detections, TARGET_CLASSES, 0.35)

    assert len(detections) == 1
    assert detections[0]["confidence"] == 0.35


def test_filter_detections_excludes_non_target_classes() -> None:
    raw_detections = [
        {"class_name": "dog", "confidence": 0.99, "bbox": [1, 2, 3, 4]},
        {"class_name": "car", "confidence": 0.99, "bbox": [5, 6, 7, 8]},
    ]

    detections = filter_detections(raw_detections, TARGET_CLASSES, 0.35)

    assert len(detections) == 1
    assert detections[0]["class_name"] == "car"
