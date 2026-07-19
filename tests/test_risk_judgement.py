import pytest

from src.risk_judgement import (
    create_mock_person_detections,
    evaluate_risk,
    get_bbox_bottom_center,
    judge_person_risk,
)


POLYGON = [[100, 250], [540, 250], [600, 470], [80, 470]]


def test_get_bbox_bottom_center_returns_expected_point() -> None:
    assert get_bbox_bottom_center([100, 100, 200, 300]) == (150, 300)


def test_get_bbox_bottom_center_raises_value_error_for_invalid_bbox() -> None:
    with pytest.raises(ValueError):
        get_bbox_bottom_center([100, 100, 200])


def test_judge_person_risk_returns_true_when_bottom_center_is_inside_polygon() -> None:
    detection = {"class_name": "person", "confidence": 0.9, "bbox": [290, 250, 350, 360]}

    result = judge_person_risk(detection, POLYGON)

    assert result["anchor_point"] == [320, 360]
    assert result["is_risk"] is True
    assert result["risk_reason"] == "person_inside_danger_zone"


def test_judge_person_risk_returns_false_when_bottom_center_is_outside_polygon() -> None:
    detection = {"class_name": "person", "confidence": 0.9, "bbox": [30, 60, 90, 180]}

    result = judge_person_risk(detection, POLYGON)

    assert result["anchor_point"] == [60, 180]
    assert result["is_risk"] is False
    assert result["risk_reason"] == "person_outside_danger_zone"


def test_judge_person_risk_marks_car_as_not_person() -> None:
    detection = {"class_name": "car", "confidence": 0.9, "bbox": [290, 250, 350, 360]}

    result = judge_person_risk(detection, POLYGON)

    assert result["is_risk"] is False
    assert result["risk_reason"] == "not_person"


def test_evaluate_risk_counts_people_cars_and_risk_people() -> None:
    detections = [
        {"class_name": "person", "confidence": 0.9, "bbox": [290, 250, 350, 360]},
        {"class_name": "person", "confidence": 0.8, "bbox": [30, 60, 90, 180]},
        {"class_name": "car", "confidence": 0.7, "bbox": [10, 10, 50, 50]},
    ]

    result = evaluate_risk(detections, POLYGON)

    assert result["person_count"] == 2
    assert result["car_count"] == 1
    assert result["risk_person_count"] == 1


def test_evaluate_risk_returns_warning_when_risk_person_exists() -> None:
    detections = [
        {"class_name": "person", "confidence": 0.9, "bbox": [290, 250, 350, 360]},
    ]

    result = evaluate_risk(detections, POLYGON)

    assert result["risk_detected"] is True
    assert result["risk_level"] == "warning"


def test_evaluate_risk_returns_none_when_no_risk_person_exists() -> None:
    detections = [
        {"class_name": "person", "confidence": 0.9, "bbox": [30, 60, 90, 180]},
        {"class_name": "car", "confidence": 0.9, "bbox": [290, 250, 350, 360]},
    ]

    result = evaluate_risk(detections, POLYGON)

    assert result["risk_detected"] is False
    assert result["risk_level"] == "none"
    assert result["risk_person_count"] == 0


def test_create_mock_person_detections_contains_at_least_two_people() -> None:
    detections = create_mock_person_detections()

    person_count = sum(1 for detection in detections if detection["class_name"] == "person")

    assert person_count >= 2
