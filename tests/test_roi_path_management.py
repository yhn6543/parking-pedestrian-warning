from datetime import datetime
import json

import app
import src.app_paths as app_paths


def test_sanitize_filename_component_replaces_spaces_and_windows_invalid_chars() -> None:
    assert app_paths.sanitize_filename_component("cctv parking.mp4") == "cctv_parking.mp4"
    assert app_paths.sanitize_filename_component("2026-06-08 14:30:22") == "2026-06-08_14_30_22"
    assert app_paths.sanitize_filename_component('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"
    assert app_paths.sanitize_filename_component("  ///  ") == "unknown_source"


def test_get_video_prefix_removes_extension_and_sanitizes_name() -> None:
    assert app.get_video_prefix("C:/temp/cctv parking.mp4") == "cctv_parking"


def test_get_realtime_prefix_accepts_fixed_datetime() -> None:
    value = app_paths.get_realtime_prefix(datetime(2026, 6, 8, 14, 30, 22))

    assert value == "2026-06-08_14-30-22"


def test_build_roi_artifact_paths_uses_shared_prefix() -> None:
    paths = app.build_roi_artifact_paths("parking_test")

    assert paths["danger_zone_json_path"] == "data/roi_zones/parking_test_danger_zone.json"
    assert paths["roi_base_image_path"] == "outputs/debug/roi_base_images/parking_test_roi_base_image.jpg"
    assert paths["roi_preview_output_path"] == "outputs/debug/roi_previews/parking_test_roi_selector_preview.jpg"


def test_list_available_roi_json_files_returns_sorted_paths(tmp_path) -> None:
    (tmp_path / "b_danger_zone.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a_danger_zone.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    result = app_paths.list_available_roi_json_files(str(tmp_path))

    assert result == [
        (tmp_path / "a_danger_zone.json").as_posix(),
        (tmp_path / "b_danger_zone.json").as_posix(),
    ]


def test_ensure_roi_json_exists_or_fallback_copies_fallback_json(tmp_path) -> None:
    fallback_path = tmp_path / "danger_zone.json"
    target_path = tmp_path / "nested" / "video_danger_zone.json"
    fallback_data = {"danger_zone": [[1, 2], [3, 4], [5, 6]]}
    fallback_path.write_text(json.dumps(fallback_data), encoding="utf-8")

    result = app_paths.ensure_roi_json_exists_or_fallback(
        str(target_path),
        fallback_roi_path=str(fallback_path),
    )

    assert result == target_path.as_posix()
    assert json.loads(target_path.read_text(encoding="utf-8")) == fallback_data


def test_ensure_roi_json_exists_or_fallback_returns_target_when_no_files_exist(tmp_path) -> None:
    target_path = tmp_path / "missing" / "video_danger_zone.json"

    result = app_paths.ensure_roi_json_exists_or_fallback(
        str(target_path),
        fallback_roi_path=str(tmp_path / "missing_fallback.json"),
    )

    assert result == target_path.as_posix()
    assert not target_path.exists()
