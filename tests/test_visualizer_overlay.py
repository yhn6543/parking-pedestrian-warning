from types import SimpleNamespace

import numpy as np

from src.visualizer import (
    build_compact_risk_overlay_lines,
    draw_risk_assessment,
    get_near_projector_label,
)


POLYGON = [[4, 4], [60, 4], [60, 44], [4, 44]]


def test_compact_risk_overlay_lines_only_include_risk_and_projector_name() -> None:
    risk_result = {
        "risk_detected": True,
        "risk_level": "high",
        "person_count": 3,
        "car_count": 2,
        "risk_person_count": 1,
        "selected_projector": {"id": "projector_1", "name": "Projector 1"},
    }

    lines = build_compact_risk_overlay_lines(risk_result)

    assert lines == ["Risk Detected: YES", "Near Projector: Projector 1"]


def test_near_projector_label_uses_selected_projector_id_when_name_missing() -> None:
    risk_result = {"selected_projector": {"id": "projector_2"}}

    assert get_near_projector_label(risk_result) == "projector_2"


def test_near_projector_label_uses_first_assignment_name() -> None:
    risk_result = {
        "selected_projectors": [
            {"projector_id": "projector_3", "projector_name": "Projector 3"},
        ]
    }

    assert get_near_projector_label(risk_result) == "Projector 3"


def test_near_projector_label_uses_first_assignment_id_when_name_missing() -> None:
    risk_result = {"selected_projectors": [{"projector_id": "projector_3"}]}

    assert get_near_projector_label(risk_result) == "projector_3"


def test_near_projector_label_supports_object_attributes() -> None:
    risk_result = {"selected_projector": SimpleNamespace(id="projector_4", name="Projector 4")}

    assert get_near_projector_label(risk_result) == "Projector 4"


def test_compact_risk_overlay_uses_dash_when_risk_is_not_detected() -> None:
    risk_result = {
        "risk_detected": False,
        "selected_projector": {"id": "projector_1", "name": "Projector 1"},
    }

    lines = build_compact_risk_overlay_lines(risk_result)

    assert lines == ["Risk Detected: NO", "Near Projector: -"]


def test_compact_risk_overlay_uses_dash_without_projector() -> None:
    risk_result = {"risk_detected": True}

    lines = build_compact_risk_overlay_lines(risk_result)

    assert lines == ["Risk Detected: YES", "Near Projector: -"]


def test_draw_risk_assessment_compact_summary_without_projector_does_not_fail() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    risk_result = {"risk_detected": True, "enhanced_detections": []}

    output = draw_risk_assessment(image, risk_result, POLYGON, compact_summary=True)

    assert output.shape == image.shape
