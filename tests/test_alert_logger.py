import csv

import numpy as np
import pytest

from src.alert_logger import AlertLogger


def create_logger(tmp_path, cooldown_frames=10) -> AlertLogger:
    return AlertLogger(
        log_csv_path=str(tmp_path / "logs" / "risk.csv"),
        alert_image_dir=str(tmp_path / "alerts"),
        cooldown_frames=cooldown_frames,
    )


def test_alert_logger_creates_parent_directories(tmp_path) -> None:
    logger = create_logger(tmp_path)

    assert logger.log_csv_path.parent.exists()
    assert logger.alert_image_dir.exists()


def test_alert_logger_rejects_negative_cooldown_frames(tmp_path) -> None:
    with pytest.raises(ValueError):
        create_logger(tmp_path, cooldown_frames=-1)


def test_should_log_returns_false_when_stable_risk_is_false(tmp_path) -> None:
    logger = create_logger(tmp_path)

    assert logger.should_log(frame_index=0, stable_risk=False) is False


def test_should_log_returns_true_for_first_stable_risk(tmp_path) -> None:
    logger = create_logger(tmp_path)

    assert logger.should_log(frame_index=2, stable_risk=True) is True


def test_should_log_respects_cooldown_before_next_allowed_frame(tmp_path) -> None:
    logger = create_logger(tmp_path, cooldown_frames=10)
    logger.last_logged_frame_index = 2

    assert logger.should_log(frame_index=3, stable_risk=True) is False


def test_should_log_allows_frame_after_cooldown(tmp_path) -> None:
    logger = create_logger(tmp_path, cooldown_frames=10)
    logger.last_logged_frame_index = 2

    assert logger.should_log(frame_index=12, stable_risk=True) is True


def test_save_alert_image_writes_jpg_file(tmp_path) -> None:
    logger = create_logger(tmp_path)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    image_path = logger.save_alert_image(frame, frame_index=2)

    assert image_path.endswith("alert_frame_000002.jpg")
    assert logger.alert_image_dir.joinpath("alert_frame_000002.jpg").exists()


def test_append_log_writes_header_and_row(tmp_path) -> None:
    logger = create_logger(tmp_path)
    event = {
        "frame_index": 2,
        "timestamp_sec": 0.2,
        "raw_risk": True,
        "stable_risk": True,
        "risk_level": "warning",
        "person_count": 2,
        "car_count": 0,
        "risk_person_count": 1,
        "risk_count": 3,
        "window_size": 5,
        "min_risk_count": 3,
        "image_path": "alert.jpg",
        "selected_projector_id": "projector_1",
        "selected_projector_name": "Projector 1",
        "selected_projector_x": 210,
        "selected_projector_y": 580,
        "selected_projector_distance": 134.2,
        "projector_dispatch_status": "mock_dispatched",
    }

    logger.append_log(event)

    with logger.log_csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["frame_index"] == "2"
    assert rows[0]["image_path"] == "alert.jpg"
    assert rows[0]["selected_projector_id"] == "projector_1"
    assert rows[0]["projector_dispatch_status"] == "mock_dispatched"


def test_log_if_needed_logs_and_returns_image_path(tmp_path) -> None:
    logger = create_logger(tmp_path)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    event = {"frame_index": 2, "stable_risk": True}

    result = logger.log_if_needed(frame, event)

    assert result["logged"] is True
    assert result["image_path"] is not None


def test_log_if_needed_returns_false_during_cooldown(tmp_path) -> None:
    logger = create_logger(tmp_path, cooldown_frames=10)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    logger.log_if_needed(frame, {"frame_index": 2, "stable_risk": True})

    result = logger.log_if_needed(frame, {"frame_index": 3, "stable_risk": True})

    assert result == {"logged": False, "image_path": None}
