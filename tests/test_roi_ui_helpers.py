import json

import app


def test_realtime_sound_trigger_default_label_maps_to_stable() -> None:
    assert app.REALTIME_SOUND_TRIGGER_LABELS[0] == "While current frame is risky"
    assert app.get_realtime_sound_trigger_value(app.REALTIME_SOUND_TRIGGER_LABELS[0]) == "stable"


def test_realtime_sound_trigger_aliases_map_to_expected_values() -> None:
    assert app.get_realtime_sound_trigger_value("Stable risk start") == "stable_start"
    assert app.get_realtime_sound_trigger_value("Logged alert only") == "logged"
    assert app.get_realtime_sound_trigger_value("Stable risk with cooldown") == "stable"


def test_load_roi_display_data_reads_roi_json_and_preview(tmp_path) -> None:
    roi_path = tmp_path / "danger_zone.json"
    preview_path = tmp_path / "preview.jpg"
    roi_payload = {"danger_zone": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    roi_path.write_text(json.dumps(roi_payload), encoding="utf-8")
    preview_path.write_bytes(b"fake preview")

    result = app.load_roi_display_data(str(roi_path), str(preview_path))

    assert result["roi_exists"] is True
    assert result["roi_preview_exists"] is True
    assert result["roi_point_count"] == 4
    assert result["roi_points"] == roi_payload["danger_zone"]
    assert result["roi_json_data"] == roi_payload


def test_load_roi_display_data_reports_missing_json(tmp_path) -> None:
    result = app.load_roi_display_data(str(tmp_path / "missing.json"), str(tmp_path / "missing_preview.jpg"))

    assert result["roi_exists"] is False
    assert result["roi_preview_exists"] is False
    assert result["roi_point_count"] == 0
    assert "does not exist" in result["error"]
