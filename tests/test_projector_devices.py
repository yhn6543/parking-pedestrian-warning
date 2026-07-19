import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import app
from src.projector_devices import (
    build_projector_artifact_path,
    find_nearest_projector,
    find_nearest_projectors_for_risk_persons,
    load_projector_config,
    save_projector_config,
    validate_projector_config,
)
from src.projector_dispatcher import dispatch_projector_alert


def sample_devices() -> list[dict]:
    return [
        {
            "id": "projector_1",
            "name": "Projector 1",
            "x": 100,
            "y": 500,
            "enabled": True,
            "endpoint": "",
        },
        {
            "id": "projector_2",
            "name": "Projector 2",
            "x": 800,
            "y": 500,
            "enabled": True,
            "endpoint": "",
        },
    ]


def test_projector_config_save_and_load_roundtrip(tmp_path: Path) -> None:
    output_path = tmp_path / "projectors.json"

    saved_path = save_projector_config(
        output_path,
        sample_devices(),
        image_size={"width": 960, "height": 720},
    )
    config = load_projector_config(saved_path)

    assert config["schema_version"] == 1
    assert config["coordinate_system"] == "original_image_pixel"
    assert config["image_size"] == {"width": 960, "height": 720}
    assert config["devices"][0]["id"] == "projector_1"


def test_disabled_projector_is_excluded_from_nearest() -> None:
    devices = sample_devices()
    devices[0]["enabled"] = False

    nearest = find_nearest_projector([110, 510], devices)

    assert nearest is not None
    assert nearest["id"] == "projector_2"


def test_find_nearest_projector_returns_closest_enabled_device() -> None:
    nearest = find_nearest_projector([130, 520], sample_devices())

    assert nearest is not None
    assert nearest["id"] == "projector_1"
    assert nearest["distance"] > 0


def test_find_nearest_projector_returns_none_when_no_devices() -> None:
    assert find_nearest_projector([100, 100], []) is None


def test_find_nearest_projectors_for_multiple_risk_persons() -> None:
    results = find_nearest_projectors_for_risk_persons(
        [[120, 500], [760, 510]],
        sample_devices(),
    )

    assert [item["projector_id"] for item in results] == ["projector_1", "projector_2"]
    assert [item["person_index"] for item in results] == [0, 1]


def test_validate_projector_config_rejects_bad_json_structure() -> None:
    bad_config = {
        "schema_version": 1,
        "image_size": {"width": 960, "height": 720},
        "coordinate_system": "original_image_pixel",
        "devices": [{"id": "bad", "name": "Bad"}],
    }

    try:
        validate_projector_config(bad_config)
    except ValueError as exc:
        assert "devices[0].x" in str(exc)
    else:
        raise AssertionError("validate_projector_config should reject missing coordinates")


def test_projector_artifact_path_does_not_collide_with_roi_path() -> None:
    roi_paths = app.build_roi_artifact_paths("sample_video")
    projector_paths = build_projector_artifact_path("sample_video")

    assert Path(roi_paths["danger_zone_json_path"]).name == "sample_video_danger_zone.json"
    assert Path(projector_paths["projector_json_path"]).name == "sample_video_projectors.json"
    assert Path(roi_paths["danger_zone_json_path"]).parent.name == "roi_zones"
    assert Path(projector_paths["projector_json_path"]).parent.name == "projector_devices"


def test_saving_projectors_does_not_modify_existing_danger_zone_json(tmp_path: Path) -> None:
    danger_zone_path = tmp_path / "danger_zone.json"
    original = {"danger_zone": [[0, 0], [10, 0], [10, 10]]}
    danger_zone_path.write_text(json.dumps(original, indent=2), encoding="utf-8")

    save_projector_config(tmp_path / "projectors.json", sample_devices(), {"width": 960, "height": 720})

    assert json.loads(danger_zone_path.read_text(encoding="utf-8")) == original


def test_attach_projector_selection_to_risk_result_handles_missing_projectors() -> None:
    risk_result = {
        "risk_detected": True,
        "enhanced_detections": [
            {
                "class_name": "person",
                "is_risk": True,
                "anchor_point": [120, 500],
            }
        ],
    }

    enriched = app.attach_projector_selection_to_risk_result(risk_result, [])

    assert enriched["selected_projector"] is None
    assert enriched["selected_projectors"] == []
    assert enriched["projector_warning"] == "No enabled projector devices configured."


def test_image_pipeline_records_selected_projector_with_mock_person(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image = np.zeros((720, 960, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    zone_path = tmp_path / "danger_zone.json"
    zone_path.write_text(
        json.dumps({"danger_zone": [[100, 250], [540, 250], [600, 470], [80, 470]]}),
        encoding="utf-8",
    )
    projector_path = tmp_path / "projectors.json"
    save_projector_config(
        projector_path,
        [
            {"id": "projector_1", "name": "Projector 1", "x": 330, "y": 365, "enabled": True, "endpoint": ""},
            {"id": "projector_2", "name": "Projector 2", "x": 900, "y": 700, "enabled": True, "endpoint": ""},
        ],
        {"width": 960, "height": 720},
    )

    result = app.run_image_pipeline(
        image_path=str(image_path),
        zone_path=str(zone_path),
        model_name="yolov8n.pt",
        conf=0.35,
        use_mock_person=True,
        projector_config_path=str(projector_path),
    )

    risk_result = result["risk_result"]

    assert risk_result["risk_detected"] is True
    assert risk_result["selected_projector"]["id"] == "projector_1"
    assert risk_result["selected_projectors"][0]["projector_id"] == "projector_1"
    assert risk_result["projector_dispatch_status"] == "mock_dispatched"


def test_projector_dispatcher_mock_mode_returns_status() -> None:
    result = dispatch_projector_alert(sample_devices()[0], {"risk_detected": True}, mode="mock")

    assert result["status"] == "mock_dispatched"
    assert result["projector_id"] == "projector_1"


def test_build_projector_selector_command_contains_expected_options() -> None:
    command = app.build_projector_selector_command(
        python_executable=sys.executable,
        image_path="data/sample_images/test.jpg",
        output_path="data/projector_devices/test_projectors.json",
        preview_output_path="outputs/debug/projector_previews/test_projectors_preview.jpg",
        zone_path="data/danger_zone.json",
        load_existing=True,
    )

    assert command[0] == sys.executable
    assert "run_projector_selector.py" in command[1]
    assert command[command.index("--image") + 1] == "data/sample_images/test.jpg"
    assert command[command.index("--output") + 1] == "data/projector_devices/test_projectors.json"
    assert command[command.index("--zone") + 1] == "data/danger_zone.json"
    assert "--load-existing" in command


def test_run_projector_selector_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_projector_selector.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--image" in result.stdout
    assert "--output" in result.stdout
    assert "--preview-output" in result.stdout
    assert "--zone" in result.stdout
