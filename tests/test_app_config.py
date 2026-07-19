from io import BytesIO

import pandas as pd

from app import (
    collect_alert_sound_times,
    create_alarm_wav_bytes,
    create_video_with_alert_audio,
    load_csv_if_exists,
    model_path_exists,
    resolve_video_frame_limit,
    summary_has_audio_alert,
    write_alert_audio_track,
    save_uploaded_file,
)


class DummyUpload(BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.name = "dummy.txt"

    def getbuffer(self):
        return super().getbuffer()


def test_save_uploaded_file_writes_bytes(tmp_path) -> None:
    uploaded_file = DummyUpload(b"hello streamlit")
    output_path = tmp_path / "nested" / "saved.txt"

    saved_path = save_uploaded_file(uploaded_file, output_path)

    assert saved_path == str(output_path)
    assert output_path.read_bytes() == b"hello streamlit"


def test_load_csv_if_exists_returns_none_for_missing_file(tmp_path) -> None:
    assert load_csv_if_exists(tmp_path / "missing.csv") is None


def test_load_csv_if_exists_returns_dataframe(tmp_path) -> None:
    csv_path = tmp_path / "log.csv"
    pd.DataFrame([{"frame_index": 1, "stable_risk": True}]).to_csv(csv_path, index=False)

    result = load_csv_if_exists(csv_path)

    assert result is not None
    assert list(result.columns) == ["frame_index", "stable_risk"]


def test_model_path_exists_supports_existing_and_missing_paths(tmp_path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")

    assert model_path_exists(str(model_path)) is True
    assert model_path_exists(str(tmp_path / "missing.pt")) is False


def test_resolve_video_frame_limit_supports_entire_video_mode() -> None:
    assert resolve_video_frame_limit(30, process_entire_video=False) == 30
    assert resolve_video_frame_limit(30, process_entire_video=True) is None
    assert resolve_video_frame_limit(0, process_entire_video=False) is None
    assert resolve_video_frame_limit(None, process_entire_video=False) is None


def test_create_alarm_wav_bytes_returns_wav_data() -> None:
    data = create_alarm_wav_bytes(duration_sec=0.1, sample_rate=8000)

    assert data.startswith(b"RIFF")
    assert b"WAVE" in data[:16]
    assert len(data) > 1000


def test_summary_has_audio_alert_uses_stable_risk_or_logged_count() -> None:
    assert summary_has_audio_alert({"stable_risk_frames": 0, "logged_alert_count": 0}) is False
    assert summary_has_audio_alert({"stable_risk_frames": 1, "logged_alert_count": 0}) is True
    assert summary_has_audio_alert({"stable_risk_frames": 0, "logged_alert_count": 1}) is True


def test_collect_alert_sound_times_prefers_logged_frames() -> None:
    summary = {
        "output_fps": 10.0,
        "frame_results": [
            {"stable_risk": False, "logged": False},
            {"stable_risk": True, "logged": False},
            {"stable_risk": True, "logged": True},
            {"stable_risk": True, "logged": False},
        ],
    }

    assert collect_alert_sound_times(summary) == [0.2]


def test_collect_alert_sound_times_uses_stable_risk_transitions_without_logged_frames() -> None:
    summary = {
        "output_fps": 5.0,
        "frame_results": [
            {"stable_risk": False, "logged": False},
            {"stable_risk": True, "logged": False},
            {"stable_risk": True, "logged": False},
            {"stable_risk": False, "logged": False},
            {"stable_risk": True, "logged": False},
        ],
    }

    assert collect_alert_sound_times(summary) == [0.2, 0.8]


def test_write_alert_audio_track_creates_wav_file(tmp_path) -> None:
    output_path = tmp_path / "alert_track.wav"

    saved_path = write_alert_audio_track([0.0, 0.4], duration_sec=1.0, output_path=output_path, sample_rate=8000)

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"RIFF")


def test_create_video_with_alert_audio_reports_missing_ffmpeg(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake mp4")
    monkeypatch.setattr("app.shutil.which", lambda _name: None)

    result = create_video_with_alert_audio(
        str(video_path),
        {
            "processed_frames": 10,
            "output_fps": 10.0,
            "stable_risk_frames": 2,
            "logged_alert_count": 1,
            "frame_results": [
                {"stable_risk": False, "logged": False},
                {"stable_risk": True, "logged": True},
            ],
        },
    )

    assert result["created"] is False
    assert "ffmpeg" in result["reason"]
    assert result["alert_times_sec"] == [0.1]
