import math
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np


def _get_summary_fps(summary: dict) -> float:
    fps = float(summary.get("output_fps") or summary.get("fps") or 0.0)
    if fps <= 0 or not math.isfinite(fps):
        return 10.0
    return fps


def _get_frame_timestamp(frame_result: dict, index: int, fps: float) -> float:
    timestamp = frame_result.get("timestamp_sec")
    if timestamp is not None:
        return float(timestamp)
    frame_index = frame_result.get("frame_index", index)
    return float(frame_index) / fps


def get_alert_timestamps_from_summary(summary: dict, trigger_mode: str = "logged") -> list[float]:
    frame_results = list(summary.get("frame_results", []) or [])
    fps = _get_summary_fps(summary)
    normalized_mode = str(trigger_mode).strip().lower()
    if normalized_mode in {"logged alert only", "logged_only"}:
        normalized_mode = "logged"
    if normalized_mode in {"stable risk start", "stable_start"}:
        normalized_mode = "stable_start"
    if normalized_mode in {"every stable risk frame", "stable_every", "every_stable"}:
        normalized_mode = "stable_every"
    if normalized_mode not in {"logged", "stable_start", "stable_every"}:
        raise ValueError("trigger_mode must be 'logged', 'stable_start', or 'stable_every'.")

    timestamps = []
    previous_stable_risk = False
    for index, frame_result in enumerate(frame_results):
        if normalized_mode == "logged" and frame_result.get("logged"):
            timestamps.append(_get_frame_timestamp(frame_result, index, fps))
        elif normalized_mode == "stable_every" and frame_result.get("stable_risk"):
            timestamps.append(_get_frame_timestamp(frame_result, index, fps))
        elif normalized_mode == "stable_start":
            stable_risk = bool(frame_result.get("stable_risk"))
            if stable_risk and not previous_stable_risk:
                timestamps.append(_get_frame_timestamp(frame_result, index, fps))
            previous_stable_risk = stable_risk

    unique_timestamps = sorted({round(max(0.0, float(timestamp)), 4) for timestamp in timestamps})
    return unique_timestamps


def generate_beep_wave(
    output_wav_path: str,
    duration_sec: float,
    beep_timestamps: list[float],
    sample_rate: int = 44100,
    beep_frequency: float = 1000.0,
    beep_duration_sec: float = 0.25,
    volume: float = 0.5,
) -> str:
    duration_sec = float(duration_sec)
    sample_rate = int(sample_rate)
    beep_frequency = float(beep_frequency)
    beep_duration_sec = float(beep_duration_sec)
    volume = float(volume)

    if duration_sec <= 0 or not math.isfinite(duration_sec):
        raise ValueError("duration_sec must be greater than 0.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")
    if beep_frequency <= 0 or not math.isfinite(beep_frequency):
        raise ValueError("beep_frequency must be greater than 0.")
    if beep_duration_sec <= 0 or not math.isfinite(beep_duration_sec):
        raise ValueError("beep_duration_sec must be greater than 0.")
    volume = min(1.0, max(0.0, volume))

    total_samples = max(1, int(math.ceil(duration_sec * sample_rate)))
    track = np.zeros(total_samples, dtype=np.float32)
    beep_samples = max(1, int(round(beep_duration_sec * sample_rate)))
    beep_t = np.arange(beep_samples, dtype=np.float32) / float(sample_rate)
    beep = np.sin(2.0 * math.pi * beep_frequency * beep_t).astype(np.float32)

    fade_samples = min(max(1, int(sample_rate * 0.01)), max(1, beep_samples // 2))
    envelope = np.ones(beep_samples, dtype=np.float32)
    envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples)
    envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
    beep = beep * envelope * volume

    for timestamp in beep_timestamps:
        start = int(round(max(0.0, float(timestamp)) * sample_rate))
        end = min(total_samples, start + beep_samples)
        if end <= start:
            continue
        track[start:end] += beep[: end - start]

    track = np.clip(track, -1.0, 1.0)
    pcm = (track * 32767).astype(np.int16)

    output_path = Path(output_wav_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return str(output_path)


def _get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("imageio-ffmpeg is not available.") from exc


def mux_video_with_audio(video_path: str, audio_wav_path: str, output_video_path: str) -> str:
    video = Path(video_path)
    audio = Path(audio_wav_path)
    output = Path(output_video_path)
    if not video.exists() or video.stat().st_size <= 0:
        raise FileNotFoundError(f"Video file is missing or empty: {video}")
    if not audio.exists() or audio.stat().st_size <= 0:
        raise FileNotFoundError(f"Audio wav file is missing or empty: {audio}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _get_ffmpeg_executable(),
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        details = completed.stderr[-700:] or completed.stdout[-700:] or "ffmpeg mux failed."
        raise RuntimeError(details)
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("ffmpeg finished but did not create a valid output video.")
    return str(output)


def _get_video_duration_sec(video_path: str) -> float | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count > 0 and fps > 0 and math.isfinite(fps):
            return frame_count / fps
    finally:
        capture.release()
    return None


def _estimate_summary_duration_sec(summary: dict, video_path: str, beep_duration_sec: float) -> float:
    video_duration = _get_video_duration_sec(video_path)
    if video_duration and video_duration > 0:
        return float(video_duration)

    fps = _get_summary_fps(summary)
    processed_frames = int(summary.get("processed_frames", 0) or 0)
    frame_results = list(summary.get("frame_results", []) or [])
    timestamps = [
        _get_frame_timestamp(frame_result, index, fps)
        for index, frame_result in enumerate(frame_results)
    ]
    if timestamps:
        return max(timestamps) + max(beep_duration_sec, 1.0 / fps)
    if processed_frames > 0:
        return processed_frames / fps
    return max(beep_duration_sec, 0.25)


def add_alert_sound_to_video(
    video_path: str,
    summary: dict,
    output_video_path: str,
    output_wav_path: str,
    trigger_mode: str = "logged",
    beep_frequency: float = 1000.0,
    beep_duration_sec: float = 0.25,
    volume: float = 0.5,
) -> dict:
    try:
        alert_timestamps = get_alert_timestamps_from_summary(summary, trigger_mode=trigger_mode)
        if not alert_timestamps:
            return {
                "with_sound": False,
                "reason": "no_alert_timestamps",
                "video_path": str(video_path),
                "alert_timestamps": [],
            }

        duration_sec = max(
            _estimate_summary_duration_sec(summary, video_path, beep_duration_sec),
            max(alert_timestamps) + float(beep_duration_sec),
        )
        audio_wav_path = generate_beep_wave(
            output_wav_path=output_wav_path,
            duration_sec=duration_sec,
            beep_timestamps=alert_timestamps,
            beep_frequency=beep_frequency,
            beep_duration_sec=beep_duration_sec,
            volume=volume,
        )
        video_with_sound_path = mux_video_with_audio(
            video_path=video_path,
            audio_wav_path=audio_wav_path,
            output_video_path=output_video_path,
        )
        return {
            "with_sound": True,
            "alert_timestamps": alert_timestamps,
            "audio_wav_path": audio_wav_path,
            "video_with_sound_path": video_with_sound_path,
        }
    except Exception as exc:
        return {
            "with_sound": False,
            "reason": str(exc),
            "video_path": str(video_path),
        }
