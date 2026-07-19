from pathlib import Path

from src.model_registry import (
    FINETUNED_2CLASS_MODEL_KEY,
    FINETUNED_DEFAULT_PATH,
    PERSON_ONLY_DEFAULT_PATH,
    PERSON_ONLY_MODEL_KEY,
    YOLO26_PERSON_ONLY_DEFAULT_PATH,
    YOLO26_PERSON_ONLY_MODEL_KEY,
    YOLO26_PRETRAINED_MODEL_KEY,
    build_model_options,
    get_default_model_key,
    get_model_option_from_selection,
    get_model_path_from_selection,
    should_block_model_execution,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_model_options_keep_existing_keys_and_add_yolo26_options(tmp_path: Path) -> None:
    options = build_model_options(tmp_path)

    assert [option.key for option in options] == [
        "yolov8n_pretrained",
        "yolo11n_pretrained",
        FINETUNED_2CLASS_MODEL_KEY,
        PERSON_ONLY_MODEL_KEY,
        YOLO26_PRETRAINED_MODEL_KEY,
        YOLO26_PERSON_ONLY_MODEL_KEY,
    ]
    assert [option.display_name for option in options] == [
        "YOLOv8n pretrained",
        "YOLO11n pretrained",
        "Fine-tuned YOLO11n person/car",
        "Fine-tuned YOLO11n person-only",
        "YOLO26n pretrained",
        "Fine-tuned YOLO26n person-only",
    ]


def test_person_only_model_option_uses_expected_path_and_classes(tmp_path: Path) -> None:
    person_only = build_model_options(tmp_path)[3]

    assert person_only.key == PERSON_ONLY_MODEL_KEY
    assert person_only.path == "models/fine_tuned/parking_yolo11n_person_only_best.pt"
    assert person_only.model_type == "project_person_only"
    assert person_only.expected_classes == ("person",)
    assert person_only.exists is False
    assert "parking_yolo11n_person_only_best.pt" in person_only.warning


def test_2class_finetuned_model_option_uses_expected_path_and_classes(tmp_path: Path) -> None:
    fine_tuned = build_model_options(tmp_path)[2]

    assert fine_tuned.key == FINETUNED_2CLASS_MODEL_KEY
    assert fine_tuned.path == "models/fine_tuned/parking_yolo11n_best.pt"
    assert fine_tuned.model_type == "project_2class"
    assert fine_tuned.expected_classes == ("person", "car")


def test_yolo26_pretrained_option_uses_expected_path_and_confidence(tmp_path: Path) -> None:
    option = get_model_option_from_selection(YOLO26_PRETRAINED_MODEL_KEY, build_model_options(tmp_path))

    assert option is not None
    assert option.path == "yolo26n.pt"
    assert option.model_type == "coco_pretrained"
    assert option.expected_classes == ("person", "car")
    assert option.recommended_confidence == 0.30


def test_yolo26_person_only_option_uses_expected_path_and_missing_file_warning(tmp_path: Path) -> None:
    option = get_model_option_from_selection(YOLO26_PERSON_ONLY_MODEL_KEY, build_model_options(tmp_path))

    assert option is not None
    assert option.path == "models/fine_tuned/parking_yolo26n_person_only_best.pt"
    assert option.model_type == "project_person_only"
    assert option.expected_classes == ("person",)
    assert option.exists is False
    assert option.recommended_confidence == 0.25
    assert "parking_yolo26n_person_only_best.pt" in option.warning


def test_default_model_prefers_existing_person_only(tmp_path: Path) -> None:
    touch(tmp_path / "yolo11n.pt")
    touch(tmp_path / FINETUNED_DEFAULT_PATH)
    touch(tmp_path / PERSON_ONLY_DEFAULT_PATH)
    touch(tmp_path / "yolo26n.pt")
    touch(tmp_path / YOLO26_PERSON_ONLY_DEFAULT_PATH)

    assert get_default_model_key(build_model_options(tmp_path)) == PERSON_ONLY_MODEL_KEY


def test_default_model_falls_back_to_2class_when_person_only_is_missing(tmp_path: Path) -> None:
    touch(tmp_path / "yolo11n.pt")
    touch(tmp_path / FINETUNED_DEFAULT_PATH)

    assert get_default_model_key(build_model_options(tmp_path)) == FINETUNED_2CLASS_MODEL_KEY


def test_default_model_falls_back_to_yolo11n_then_yolov8n(tmp_path: Path) -> None:
    touch(tmp_path / "yolo11n.pt")
    touch(tmp_path / "yolov8n.pt")
    touch(tmp_path / "yolo26n.pt")
    touch(tmp_path / YOLO26_PERSON_ONLY_DEFAULT_PATH)

    assert get_default_model_key(build_model_options(tmp_path)) == "yolo11n_pretrained"

    (tmp_path / "yolo11n.pt").unlink()
    (tmp_path / YOLO26_PERSON_ONLY_DEFAULT_PATH).unlink()

    assert get_default_model_key(build_model_options(tmp_path)) == "yolov8n_pretrained"


def test_yolo26_options_do_not_change_default_selection_priority(tmp_path: Path) -> None:
    touch(tmp_path / "yolo26n.pt")
    touch(tmp_path / YOLO26_PERSON_ONLY_DEFAULT_PATH)

    assert get_default_model_key(build_model_options(tmp_path)) == "yolo11n_pretrained"


def test_default_model_returns_yolo11n_when_no_files_exist(tmp_path: Path) -> None:
    assert get_default_model_key(build_model_options(tmp_path)) == "yolo11n_pretrained"


def test_get_model_path_from_selection_returns_selected_path(tmp_path: Path) -> None:
    options = build_model_options(tmp_path)

    assert get_model_path_from_selection(PERSON_ONLY_MODEL_KEY, options) == (
        "models/fine_tuned/parking_yolo11n_person_only_best.pt"
    )
    assert get_model_path_from_selection("missing", options) == ""


def test_should_block_missing_project_models_without_manual_override(tmp_path: Path) -> None:
    options = build_model_options(tmp_path)
    fine_tuned_2class = options[2]
    person_only = options[3]
    pretrained = options[1]

    blocked_2class, message_2class = should_block_model_execution(
        fine_tuned_2class,
        use_manual_model_path=False,
    )
    blocked_person_only, message_person_only = should_block_model_execution(
        person_only,
        use_manual_model_path=False,
    )
    yolo26_person_only = get_model_option_from_selection(YOLO26_PERSON_ONLY_MODEL_KEY, options)
    blocked_yolo26_person_only, message_yolo26_person_only = should_block_model_execution(
        yolo26_person_only,
        use_manual_model_path=False,
    )
    manual_blocked, manual_message = should_block_model_execution(
        person_only,
        use_manual_model_path=True,
    )
    pretrained_blocked, pretrained_message = should_block_model_execution(
        pretrained,
        use_manual_model_path=False,
    )

    assert blocked_2class is True
    assert "parking_yolo11n_best.pt" in message_2class
    assert blocked_person_only is True
    assert "parking_yolo11n_person_only_best.pt" in message_person_only
    assert blocked_yolo26_person_only is True
    assert "parking_yolo26n_person_only_best.pt" in message_yolo26_person_only
    assert manual_blocked is False
    assert manual_message == ""
    assert pretrained_blocked is False
    assert pretrained_message == ""
