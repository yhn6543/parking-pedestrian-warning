import json
import sys

import cv2
import numpy as np
import pytest

import app


def create_dummy_video(path, frame_count=3, width=64, height=48):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (width, height))
    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (index * 40, index * 30, index * 20)
            cv2.putText(
                frame,
                str(index),
                (5, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_capture_video_frame_for_roi_saves_requested_frame(tmp_path) -> None:
    video_path = create_dummy_video(tmp_path / "input.mp4")
    output_path = tmp_path / "roi_frame.jpg"

    saved_path = app.capture_video_frame_for_roi(
        video_path=str(video_path),
        frame_index=1,
        output_path=str(output_path),
    )

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_capture_video_frame_for_roi_accepts_frame_zero(tmp_path) -> None:
    video_path = create_dummy_video(tmp_path / "input.mp4")
    output_path = tmp_path / "frame_zero.jpg"

    app.capture_video_frame_for_roi(
        video_path=str(video_path),
        frame_index=0,
        output_path=str(output_path),
    )

    assert output_path.exists()


def test_capture_video_frame_for_roi_rejects_negative_frame_index(tmp_path) -> None:
    video_path = create_dummy_video(tmp_path / "input.mp4")

    with pytest.raises(ValueError):
        app.capture_video_frame_for_roi(
            video_path=str(video_path),
            frame_index=-1,
            output_path=str(tmp_path / "bad.jpg"),
        )


def test_capture_video_frame_for_roi_raises_for_missing_video(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        app.capture_video_frame_for_roi(
            video_path=str(tmp_path / "missing.mp4"),
            frame_index=0,
            output_path=str(tmp_path / "missing.jpg"),
        )


def test_capture_video_frame_for_roi_clamps_to_last_frame(tmp_path) -> None:
    video_path = create_dummy_video(tmp_path / "input.mp4", frame_count=2)
    output_path = tmp_path / "clamped.jpg"

    app.capture_video_frame_for_roi(
        video_path=str(video_path),
        frame_index=99,
        output_path=str(output_path),
    )

    assert output_path.exists()


def test_roi_selector_command_accepts_video_snapshot_path() -> None:
    snapshot_path = "outputs/debug/streamlit_video_roi_snapshot.jpg"
    command = app.build_roi_selector_command(
        python_executable=sys.executable,
        image_path=snapshot_path,
        output_path="data/danger_zone.json",
        preview_output_path="outputs/debug/streamlit_roi_selector_preview.jpg",
        load_existing=True,
    )

    assert command[command.index("--image") + 1] == snapshot_path
    assert "--load-existing" in command


def test_run_video_pipeline_summary_marks_entire_video_mode(tmp_path) -> None:
    video_path = create_dummy_video(tmp_path / "input.mp4", frame_count=3)
    zone_path = tmp_path / "zone.json"
    zone_path.write_text(
        json.dumps({"danger_zone": [[4, 4], [60, 4], [60, 44], [4, 44]]}),
        encoding="utf-8",
    )

    result = app.run_video_pipeline(
        video_path=str(video_path),
        zone_path=str(zone_path),
        model_name="yolov8n.pt",
        conf=0.35,
        max_frames=1,
        frame_step=1,
        window_size=2,
        min_risk_count=1,
        cooldown_frames=0,
        use_mock_person=True,
        process_entire_video=True,
    )

    summary = result["summary"]

    assert summary["process_entire_video"] is True
    assert summary["max_frames_limit"] is None
    assert summary["processed_frames"] == 3


def test_run_video_pipeline_does_not_draw_right_top_temporal_box(tmp_path, monkeypatch) -> None:
    video_path = create_dummy_video(tmp_path / "input.mp4", frame_count=1)
    zone_path = tmp_path / "zone.json"
    zone_path.write_text(
        json.dumps({"danger_zone": [[4, 4], [60, 4], [60, 44], [4, 44]]}),
        encoding="utf-8",
    )

    def fail_if_temporal_overlay_is_drawn(*args, **kwargs):
        raise AssertionError("Processed video should not draw the right-top temporal overlay.")

    monkeypatch.setattr(app, "draw_temporal_status", fail_if_temporal_overlay_is_drawn)

    result = app.run_video_pipeline(
        video_path=str(video_path),
        zone_path=str(zone_path),
        model_name="yolov8n.pt",
        conf=0.35,
        max_frames=1,
        frame_step=1,
        window_size=2,
        min_risk_count=1,
        cooldown_frames=0,
        use_mock_person=True,
        process_entire_video=False,
    )

    assert result["summary"]["processed_frames"] == 1
