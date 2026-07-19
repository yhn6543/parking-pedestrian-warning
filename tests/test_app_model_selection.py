from pathlib import Path

from app import (
    DEFAULT_VIDEO_ALERT_SOUND_TRIGGER_LABEL,
    VIDEO_ALERT_SOUND_TRIGGER_LABELS,
    get_default_video_alert_sound_trigger_index,
    get_model_requirement_summary,
    get_recommended_confidence,
    get_video_alert_sound_trigger_value,
    is_person_only_model_option,
)
from src.config import CONF_THRESHOLD
from src.model_registry import (
    FINETUNED_DEFAULT_PATH,
    PERSON_ONLY_DEFAULT_PATH,
    YOLO26_PERSON_ONLY_MODEL_KEY,
    YOLO26_PRETRAINED_MODEL_KEY,
    build_model_options,
    get_model_option_from_selection,
    get_model_path_from_selection,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_app_model_selection_helpers_detect_person_only_model(tmp_path: Path) -> None:
    touch(tmp_path / PERSON_ONLY_DEFAULT_PATH)
    person_only = build_model_options(tmp_path)[3]

    summary = get_model_requirement_summary(person_only)

    assert is_person_only_model_option(person_only) is True
    assert summary["model_type"] == "project_person_only"
    assert summary["expected_classes"] == ["person"]
    assert summary["person_detection_required"] is True
    assert summary["car_detection_required"] is False
    assert get_recommended_confidence(person_only) == 0.25


def test_app_model_selection_helpers_keep_2class_default_confidence(tmp_path: Path) -> None:
    touch(tmp_path / FINETUNED_DEFAULT_PATH)
    fine_tuned_2class = build_model_options(tmp_path)[2]

    summary = get_model_requirement_summary(fine_tuned_2class)

    assert is_person_only_model_option(fine_tuned_2class) is False
    assert summary["model_type"] == "project_2class"
    assert summary["expected_classes"] == ["person", "car"]
    assert summary["car_detection_required"] is False
    assert get_recommended_confidence(fine_tuned_2class) == CONF_THRESHOLD


def test_app_model_selection_helpers_use_yolo26_recommended_confidence(tmp_path: Path) -> None:
    options = build_model_options(tmp_path)
    yolo26_pretrained = get_model_option_from_selection(YOLO26_PRETRAINED_MODEL_KEY, options)
    yolo26_person_only = get_model_option_from_selection(YOLO26_PERSON_ONLY_MODEL_KEY, options)

    assert get_recommended_confidence(yolo26_pretrained) == 0.30
    assert get_recommended_confidence(yolo26_person_only) == 0.25


def test_internal_model_path_selection_still_returns_person_only_path(tmp_path: Path) -> None:
    options = build_model_options(tmp_path)

    assert get_model_path_from_selection("yolo11n_person_only", options) == (
        "models/fine_tuned/parking_yolo11n_person_only_best.pt"
    )


def test_default_video_alert_sound_trigger_is_every_stable_risk_frame() -> None:
    assert DEFAULT_VIDEO_ALERT_SOUND_TRIGGER_LABEL == "Every stable risk frame"
    assert VIDEO_ALERT_SOUND_TRIGGER_LABELS[get_default_video_alert_sound_trigger_index()] == (
        "Every stable risk frame"
    )
    assert get_video_alert_sound_trigger_value(DEFAULT_VIDEO_ALERT_SOUND_TRIGGER_LABEL) == "stable_every"
