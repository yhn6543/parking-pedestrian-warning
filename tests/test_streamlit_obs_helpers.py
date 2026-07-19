import sys

import numpy as np

import app


def test_build_roi_selector_command_contains_expected_options() -> None:
    command = app.build_roi_selector_command(
        python_executable=sys.executable,
        image_path="data/sample_images/test.jpg",
        output_path="data/danger_zone.json",
        preview_output_path="outputs/debug/streamlit_roi_selector_preview.jpg",
        load_existing=True,
    )

    assert command[0] == sys.executable
    assert "run_step11_roi_selector.py" in command[1]
    assert "--image" in command
    assert "data/sample_images/test.jpg" in command
    assert "--output" in command
    assert "data/danger_zone.json" in command
    assert "--preview-output" in command
    assert "outputs/debug/streamlit_roi_selector_preview.jpg" in command
    assert command[command.index("--max-display-width") + 1] == "960"
    assert command[command.index("--max-display-height") + 1] == "720"
    assert "--load-existing" in command


def test_build_roi_selector_command_can_skip_load_existing() -> None:
    command = app.build_roi_selector_command(
        python_executable="python",
        image_path="image.jpg",
        output_path="zone.json",
        preview_output_path="preview.jpg",
        load_existing=False,
    )

    assert "--load-existing" not in command


def test_build_roi_selector_command_uses_selected_roi_image_path() -> None:
    selected_roi_image_path = "outputs/debug/streamlit_obs_roi_snapshot.jpg"

    command = app.build_roi_selector_command(
        python_executable=sys.executable,
        image_path=selected_roi_image_path,
        output_path="data/danger_zone.json",
        preview_output_path="outputs/debug/streamlit_roi_selector_preview.jpg",
        load_existing=True,
    )

    image_option_index = command.index("--image")

    assert command[image_option_index + 1] == selected_roi_image_path


def test_build_continuous_realtime_command_uses_unlimited_webcam_run() -> None:
    command = app.build_continuous_realtime_command(
        python_executable=sys.executable,
        camera_index=2,
        zone_path="data/danger_zone.json",
        model_name="yolov8n.pt",
        conf=0.35,
        frame_step=2,
        window_size=7,
        min_risk_count=4,
        cooldown_frames=30,
        resize_width=960,
        use_mock_person=True,
        save_output_video=True,
    )

    assert command[0] == sys.executable
    assert "run_step10_realtime_test.py" in command[1]
    assert command[command.index("--source") + 1] == "webcam"
    assert command[command.index("--camera-index") + 1] == "2"
    assert command[command.index("--zone") + 1] == "data/danger_zone.json"
    assert command[command.index("--model") + 1] == "yolov8n.pt"
    assert command[command.index("--conf") + 1] == "0.35"
    assert command[command.index("--frame-step") + 1] == "2"
    assert command[command.index("--window-size") + 1] == "7"
    assert command[command.index("--min-risk-count") + 1] == "4"
    assert command[command.index("--cooldown-frames") + 1] == "30"
    assert command[command.index("--resize-width") + 1] == "960"
    assert command[command.index("--max-frames") + 1] == "0"
    assert "--enable-alert-sound" in command
    assert command[command.index("--sound-trigger") + 1] == "stable"
    assert command[command.index("--sound-cooldown-frames") + 1] == "10"
    assert command[command.index("--beep-frequency") + 1] == "1000"
    assert command[command.index("--beep-duration-ms") + 1] == "250"
    assert "--use-mock-person" in command
    assert "--save-output-video" in command
    assert "--no-display" not in command


def test_build_continuous_realtime_command_can_disable_alert_sound() -> None:
    command = app.build_continuous_realtime_command(
        python_executable=sys.executable,
        camera_index=0,
        zone_path="data/danger_zone.json",
        model_name="yolov8n.pt",
        conf=0.35,
        frame_step=1,
        window_size=5,
        min_risk_count=3,
        cooldown_frames=30,
        resize_width=720,
        use_mock_person=False,
        save_output_video=False,
        roi_source_image="outputs/debug/streamlit_obs_roi_snapshot.jpg",
        enable_alert_sound=False,
        sound_trigger="logged",
        sound_cooldown_frames=20,
        beep_frequency=1200,
        beep_duration_ms=300,
    )

    assert "--disable-alert-sound" in command
    assert "--enable-alert-sound" not in command
    assert command[command.index("--roi-source-image") + 1] == "outputs/debug/streamlit_obs_roi_snapshot.jpg"
    assert command[command.index("--sound-trigger") + 1] == "logged"
    assert command[command.index("--sound-cooldown-frames") + 1] == "20"
    assert command[command.index("--beep-frequency") + 1] == "1200"
    assert command[command.index("--beep-duration-ms") + 1] == "300"


def test_streamlit_obs_output_paths_use_expected_prefixes() -> None:
    assert app.STREAMLIT_OBS_LAST_FRAME_PATH.name == "streamlit_obs_last_frame.jpg"
    assert app.STREAMLIT_OBS_SUMMARY_PATH.name == "streamlit_obs_summary.json"
    assert app.STREAMLIT_OBS_LOG_CSV_PATH.name == "streamlit_obs_risk_log.csv"
    assert app.STREAMLIT_OBS_VIDEO_PATH.name == "streamlit_obs_processed_video.mp4"
    assert app.STREAMLIT_OBS_ROI_SNAPSHOT_PATH.name == "streamlit_obs_roi_snapshot.jpg"
    assert app.STREAMLIT_ROI_SELECTOR_PREVIEW_PATH.name == "streamlit_roi_selector_preview.jpg"


def test_capture_camera_snapshot_with_mocked_capture(monkeypatch, tmp_path) -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)

    class FakeCapture:
        def __init__(self, *_args):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            self.released = True

    monkeypatch.setattr(app.cv2, "VideoCapture", FakeCapture)

    output_path = tmp_path / "snapshot.jpg"
    saved_path = app.capture_camera_snapshot(
        camera_index=0,
        output_path=str(output_path),
        resize_width=100,
    )

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_launch_roi_selector_waits_for_process_by_default(monkeypatch) -> None:
    calls = []

    class FakeCompleted:
        returncode = 0

    def fake_run(command, cwd=None, check=None):
        calls.append({"command": command, "cwd": cwd, "check": check})
        return FakeCompleted()

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    result = app.launch_roi_selector(["python", "script.py"])

    assert result["started"] is True
    assert result["finished"] is True
    assert result["returncode"] == 0
    assert calls == [{"command": ["python", "script.py"], "cwd": str(app.PROJECT_ROOT), "check": False}]


def test_launch_roi_selector_can_start_without_waiting(monkeypatch) -> None:
    calls = []

    class FakePopen:
        def __init__(self, command, cwd=None):
            calls.append({"command": command, "cwd": cwd})

    monkeypatch.setattr(app.subprocess, "Popen", FakePopen)

    result = app.launch_roi_selector(["python", "script.py"], wait=False)

    assert result["started"] is True
    assert result["finished"] is False
    assert calls == [{"command": ["python", "script.py"], "cwd": str(app.PROJECT_ROOT)}]


def test_launch_roi_selector_reports_failure(monkeypatch) -> None:
    def fake_run(_command, cwd=None, check=None):
        raise OSError("cannot start")

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    result = app.launch_roi_selector(["python", "script.py"])

    assert result["started"] is False
    assert result["finished"] is False
    assert "cannot start" in result["error"]


def test_start_continuous_realtime_process_uses_popen_without_real_camera(monkeypatch) -> None:
    calls = []

    class FakePopen:
        def __init__(self, command, cwd=None, stdout=None, stderr=None):
            calls.append({"command": command, "cwd": cwd, "stdout": stdout, "stderr": stderr})

    monkeypatch.setattr(app.subprocess, "Popen", FakePopen)

    process = app.start_continuous_realtime_process(["python", "script.py"])

    assert isinstance(process, FakePopen)
    assert calls == [
        {
            "command": ["python", "script.py"],
            "cwd": str(app.PROJECT_ROOT),
            "stdout": app.subprocess.DEVNULL,
            "stderr": app.subprocess.DEVNULL,
        }
    ]


def test_stop_continuous_realtime_process_terminates_running_process() -> None:
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.killed = True

    process = FakeProcess()

    result = app.stop_continuous_realtime_process(process)

    assert result["stopped"] is True
    assert result["killed"] is False
    assert process.terminated is True
    assert process.killed is False


def test_stop_continuous_realtime_process_handles_missing_process() -> None:
    result = app.stop_continuous_realtime_process(None)

    assert result["stopped"] is False
    assert result["already_stopped"] is True
