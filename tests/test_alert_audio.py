import wave
from pathlib import Path

import pytest

import src.alert_audio as alert_audio
from src.alert_audio import (
    add_alert_sound_to_video,
    generate_beep_wave,
    get_alert_timestamps_from_summary,
)


def test_get_alert_timestamps_uses_logged_alerts_by_default() -> None:
    summary = {
        "output_fps": 10.0,
        "frame_results": [
            {"timestamp_sec": 0.0, "stable_risk": False, "logged": False},
            {"timestamp_sec": 0.1, "stable_risk": True, "logged": False},
            {"timestamp_sec": 0.23, "stable_risk": True, "logged": True},
        ],
    }

    assert get_alert_timestamps_from_summary(summary) == [0.23]
    assert get_alert_timestamps_from_summary(summary, "Logged alert only") == [0.23]


def test_get_alert_timestamps_supports_stable_risk_start_mode() -> None:
    summary = {
        "output_fps": 10.0,
        "frame_results": [
            {"timestamp_sec": 0.0, "stable_risk": False, "logged": False},
            {"timestamp_sec": 0.1, "stable_risk": True, "logged": False},
            {"timestamp_sec": 0.2, "stable_risk": True, "logged": False},
            {"timestamp_sec": 0.3, "stable_risk": False, "logged": False},
            {"timestamp_sec": 0.4, "stable_risk": True, "logged": False},
        ],
    }

    assert get_alert_timestamps_from_summary(summary, "stable_start") == [0.1, 0.4]


def test_get_alert_timestamps_supports_every_stable_risk_frame_mode() -> None:
    summary = {
        "output_fps": 10.0,
        "frame_results": [
            {"timestamp_sec": 0.0, "stable_risk": False, "logged": False},
            {"timestamp_sec": 0.1, "stable_risk": True, "logged": False},
            {"timestamp_sec": 0.2, "stable_risk": True, "logged": False},
        ],
    }

    assert get_alert_timestamps_from_summary(summary, "Every stable risk frame") == [0.1, 0.2]


def test_get_alert_timestamps_falls_back_to_frame_index_and_fps() -> None:
    summary = {
        "output_fps": 20.0,
        "frame_results": [
            {"frame_index": 5, "logged": True},
            {"frame_index": 6, "logged": False},
        ],
    }

    assert get_alert_timestamps_from_summary(summary) == [0.25]


def test_get_alert_timestamps_rejects_unknown_trigger_mode() -> None:
    with pytest.raises(ValueError, match="trigger_mode"):
        get_alert_timestamps_from_summary({"frame_results": []}, "unknown")


def test_generate_beep_wave_creates_valid_wav_file(tmp_path) -> None:
    output_path = tmp_path / "alert_beep.wav"

    saved_path = generate_beep_wave(
        output_wav_path=str(output_path),
        duration_sec=1.0,
        beep_timestamps=[0.1, 0.5],
        sample_rate=8000,
        beep_frequency=1000.0,
        beep_duration_sec=0.1,
        volume=0.5,
    )

    assert saved_path == str(output_path)
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 8000
        assert wav_file.getnframes() > 0


def test_add_alert_sound_to_video_skips_when_no_alert_timestamps(tmp_path) -> None:
    result = add_alert_sound_to_video(
        video_path=str(tmp_path / "missing.mp4"),
        summary={"output_fps": 10.0, "frame_results": [{"logged": False, "stable_risk": False}]},
        output_video_path=str(tmp_path / "with_sound.mp4"),
        output_wav_path=str(tmp_path / "beep.wav"),
    )

    assert result["with_sound"] is False
    assert result["reason"] == "no_alert_timestamps"
    assert result["alert_timestamps"] == []


def test_add_alert_sound_to_video_writes_audio_then_muxes_video(monkeypatch, tmp_path) -> None:
    calls = {}
    video_path = tmp_path / "input.mp4"
    output_video_path = tmp_path / "with_sound.mp4"
    output_wav_path = tmp_path / "beep.wav"

    def fake_generate_beep_wave(**kwargs):
        calls["generate"] = kwargs
        Path(kwargs["output_wav_path"]).write_bytes(b"RIFF fake WAVE")
        return kwargs["output_wav_path"]

    def fake_mux_video_with_audio(video_path, audio_wav_path, output_video_path):
        calls["mux"] = {
            "video_path": video_path,
            "audio_wav_path": audio_wav_path,
            "output_video_path": output_video_path,
        }
        Path(output_video_path).write_bytes(b"fake mp4")
        return output_video_path

    monkeypatch.setattr(alert_audio, "generate_beep_wave", fake_generate_beep_wave)
    monkeypatch.setattr(alert_audio, "mux_video_with_audio", fake_mux_video_with_audio)

    result = add_alert_sound_to_video(
        video_path=str(video_path),
        summary={
            "output_fps": 10.0,
            "processed_frames": 10,
            "frame_results": [{"frame_index": 2, "logged": True, "stable_risk": True}],
        },
        output_video_path=str(output_video_path),
        output_wav_path=str(output_wav_path),
        trigger_mode="logged",
        beep_frequency=1000.0,
        beep_duration_sec=0.25,
        volume=0.5,
    )

    assert result["with_sound"] is True
    assert result["alert_timestamps"] == [0.2]
    assert result["audio_wav_path"] == str(output_wav_path)
    assert result["video_with_sound_path"] == str(output_video_path)
    assert calls["generate"]["beep_timestamps"] == [0.2]
    assert calls["mux"]["video_path"] == str(video_path)
