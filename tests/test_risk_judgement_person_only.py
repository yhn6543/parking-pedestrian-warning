from src.risk_judgement import evaluate_risk


POLYGON = [[100, 250], [540, 250], [600, 470], [80, 470]]


def test_person_only_inside_roi_sets_risk_true_with_zero_car_count() -> None:
    result = evaluate_risk(
        [{"class_name": "person", "confidence": 0.9, "bbox": [290, 250, 350, 360]}],
        POLYGON,
    )

    assert result["risk_detected"] is True
    assert result["risk_level"] == "warning"
    assert result["person_count"] == 1
    assert result["car_count"] == 0
    assert result["risk_person_count"] == 1


def test_person_only_outside_roi_sets_risk_false_even_without_cars() -> None:
    result = evaluate_risk(
        [{"class_name": "person", "confidence": 0.9, "bbox": [30, 60, 90, 180]}],
        POLYGON,
    )

    assert result["risk_detected"] is False
    assert result["risk_level"] == "none"
    assert result["person_count"] == 1
    assert result["car_count"] == 0
    assert result["risk_person_count"] == 0
