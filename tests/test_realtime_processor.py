import json

import numpy as np
import pytest

from src.realtime_processor import (
    RealtimeStats,
    draw_realtime_overlay,
    open_realtime_source,
    resize_frame_keep_aspect,
    scale_polygon_to_frame,
    save_realtime_summary,
)


def test_realtime_stats_initial_values_are_zero() -> None:
    stats = RealtimeStats()

    assert stats.processed_frames == 0
    assert stats.raw_risk_frames == 0
    assert stats.stable_risk_frames == 0
    assert stats.logged_alert_count == 0
    assert stats.max_risk_person_count == 0


def test_realtime_stats_update_counts_risk_and_logged_alert() -> None:
    stats = RealtimeStats()
    risk_result = {"risk_detected": True, "risk_person_count": 2}
    temporal_result = {"raw_risk": True, "stable_risk": True}

    stats.update(risk_result, temporal_result, logged=True)

    assert stats.processed_frames == 1
    assert stats.raw_risk_frames == 1
    assert stats.stable_risk_frames == 1
    assert stats.logged_alert_count == 1
    assert stats.max_risk_person_count == 2


def test_resize_frame_keep_aspect_uses_target_width() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_frame_keep_aspect(frame, target_width=100)

    assert resized.shape == (50, 100, 3)


def test_resize_frame_keep_aspect_rejects_zero_width() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_frame_keep_aspect(frame, target_width=0)


def test_scale_polygon_to_frame_scales_half_size_frame() -> None:
    polygon = [[0, 0], [960, 540], [1920, 1080]]

    scaled = scale_polygon_to_frame(
        polygon,
        source_width=1920,
        source_height=1080,
        target_width=960,
        target_height=540,
    )

    assert scaled == [[0, 0], [480, 270], [959, 539]]


def test_scale_polygon_to_frame_handles_640_to_720_width_resize() -> None:
    polygon = [[320, 240], [640, 480]]

    scaled = scale_polygon_to_frame(
        polygon,
        source_width=640,
        source_height=480,
        target_width=720,
        target_height=540,
    )

    assert scaled == [[360, 270], [719, 539]]


def test_scale_polygon_to_frame_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        scale_polygon_to_frame([[1, 2]], 0, 480, 720, 540)


def test_draw_realtime_overlay_returns_same_shape_image() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    output = draw_realtime_overlay(frame, frame_index=7, fps=12.34, source_name="video")

    assert output.shape == frame.shape
    assert output.sum() > 0


def test_save_realtime_summary_writes_json_file(tmp_path) -> None:
    output_path = tmp_path / "summary.json"
    summary = {
        "mode": "MOCK_PERSON",
        "processed_frames": np.int64(3),
        "stable_risk": np.bool_(True),
    }

    saved_path = save_realtime_summary(summary, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_path == str(output_path)
    assert data["processed_frames"] == 3
    assert data["stable_risk"] is True


def test_open_realtime_source_rejects_invalid_source() -> None:
    with pytest.raises(ValueError):
        open_realtime_source("invalid")
