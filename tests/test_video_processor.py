import json

import cv2
import numpy as np
import pytest

from src.video_processor import (
    create_dummy_video_from_image,
    get_video_info,
    save_video_summary,
    summarize_frame_result,
)


def test_summarize_frame_result_returns_expected_fields() -> None:
    risk_result = {
        "risk_detected": True,
        "risk_level": "warning",
        "person_count": 2,
        "car_count": 3,
        "risk_person_count": 1,
    }

    summary = summarize_frame_result(7, risk_result)

    assert summary == {
        "frame_index": 7,
        "risk_detected": True,
        "risk_level": "warning",
        "person_count": 2,
        "car_count": 3,
        "risk_person_count": 1,
    }


def test_save_video_summary_writes_json_file(tmp_path) -> None:
    output_path = tmp_path / "summary.json"
    summary = {
        "mode": "YOLO",
        "processed_frames": np.int64(1),
        "risk_frames": np.int64(0),
        "frame_results": [{"risk_detected": np.bool_(False)}],
    }

    saved_path = save_video_summary(summary, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_path == str(output_path)
    assert data["processed_frames"] == 1
    assert data["frame_results"][0]["risk_detected"] is False


def test_create_dummy_video_from_image_creates_video(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    video_path = tmp_path / "dummy.mp4"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    saved_path = create_dummy_video_from_image(str(image_path), str(video_path), frame_count=3, fps=5.0)

    assert saved_path == str(video_path)
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_get_video_info_returns_dummy_video_metadata(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    video_path = tmp_path / "dummy.mp4"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    create_dummy_video_from_image(str(image_path), str(video_path), frame_count=4, fps=5.0)

    info = get_video_info(str(video_path))

    assert info["width"] == 64
    assert info["height"] == 48
    assert info["fps"] > 0
    assert info["frame_count"] == 4


def test_get_video_info_raises_file_not_found_for_missing_video() -> None:
    with pytest.raises(FileNotFoundError):
        get_video_info("data/sample_videos/missing.mp4")
