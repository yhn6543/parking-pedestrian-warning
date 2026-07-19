from types import SimpleNamespace

import pytest

import app


def test_read_video_bytes_returns_file_bytes(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake video bytes")

    assert app.read_video_bytes(video_path) == b"fake video bytes"


def test_read_video_bytes_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        app.read_video_bytes(tmp_path / "missing.mp4")


def test_read_video_bytes_raises_for_empty_file(tmp_path) -> None:
    video_path = tmp_path / "empty.mp4"
    video_path.write_bytes(b"")

    with pytest.raises(ValueError):
        app.read_video_bytes(video_path)


def test_get_file_size_mb_returns_positive_value(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"x" * 2048)

    assert app.get_file_size_mb(video_path) > 0


def test_convert_video_for_streamlit_reports_missing_ffmpeg(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output_web.mp4"
    input_path.write_bytes(b"fake video bytes")
    monkeypatch.setattr(app, "get_ffmpeg_executable", lambda: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        app.convert_video_for_streamlit(input_path, output_path)


def test_convert_video_for_streamlit_preserves_audio_track(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output_web.mp4"
    input_path.write_bytes(b"fake video bytes")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output_path.write_bytes(b"converted video bytes")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(app, "get_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(app.subprocess, "run", fake_run)

    saved_path = app.convert_video_for_streamlit(input_path, output_path)

    assert saved_path == str(output_path)
    assert "-an" not in captured["command"]
    assert "-c:a" in captured["command"]
    assert "aac" in captured["command"]
