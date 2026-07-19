import base64
import io
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_logger import AlertLogger
from src.alert_audio import add_alert_sound_to_video
from src.app_paths import (
    display_path,
    ensure_roi_json_exists_or_fallback,
    get_realtime_prefix,
    get_video_prefix as get_video_prefix,
    list_available_roi_json_files,
    sanitize_filename_component,
)
from src.config import (
    ALERT_DIR,
    CONF_THRESHOLD,
    DEBUG_DIR,
    DEFAULT_MODEL_NAME,
    LOG_DIR,
    RESULT_DIR,
    SAMPLE_IMAGE_DIR,
    SAMPLE_VIDEO_DIR,
    ensure_directories,
)
from src.danger_zone import draw_danger_zone, load_danger_zone
from src.detector import ObjectDetector
from src.input_loader import load_image, save_debug_image
from src.model_registry import (
    ModelOption,
    build_model_options,
    get_default_model_key,
    get_model_path_from_selection,
    get_model_option_from_selection,
    resolve_selected_model_path,
    should_block_model_execution,
)
from src.projector_devices import (
    build_projector_artifact_path,
    enabled_projector_devices,
    find_nearest_projectors_for_risk_persons,
    load_projector_config,
    scale_projector_devices_to_frame,
    selected_projector_from_assignments,
)
from src.projector_dispatcher import dispatch_projector_alert
from src.realtime_processor import (
    RealtimeStats,
    draw_realtime_overlay,
    get_image_size,
    resize_frame_keep_aspect,
    scale_polygon_to_frame,
    save_realtime_summary,
)
from src.realtime_alert_sound import RealtimeAlertSound
from src.risk_judgement import create_mock_person_detections, evaluate_risk
from src.temporal_filter import TemporalRiskFilter
from src.ui_state import (
    increment_session_state_versions,
    initialize_app_session_state,
    resolve_roi_source_prefix,
    sync_roi_source_artifacts,
)
from src.video_processor import (
    create_dummy_video_from_image,
    create_video_writer,
    get_video_info,
    save_video_summary,
    summarize_frame_result,
)
from src.video_display import (
    get_ffmpeg_executable,
    get_file_size_mb,
    read_video_bytes,
)
from src.visualizer import draw_risk_assessment, draw_temporal_status


STREAMLIT_IMAGE_RESULT_PATH = DEBUG_DIR / "streamlit_image_result.jpg"
STREAMLIT_VIDEO_PATH = RESULT_DIR / "streamlit_processed_video.mp4"
STREAMLIT_VIDEO_WEB_PATH = RESULT_DIR / "streamlit_processed_video_web.mp4"
STREAMLIT_VIDEO_WITH_ALERT_SOUND_PATH = RESULT_DIR / "streamlit_processed_video_with_alert_sound.mp4"
STREAMLIT_VIDEO_WITH_ALERT_SOUND_WEB_PATH = RESULT_DIR / "streamlit_processed_video_with_alert_sound_web.mp4"
STREAMLIT_ALERT_BEEP_WAV_PATH = DEBUG_DIR / "streamlit_alert_beep.wav"
STREAMLIT_VIDEO_WITH_ALERT_AUDIO_PATH = RESULT_DIR / "streamlit_processed_video_alert_audio.mp4"
STREAMLIT_VIDEO_ALERT_AUDIO_TRACK_PATH = DEBUG_DIR / "streamlit_video_alert_audio_track.wav"
STREAMLIT_PREVIEW_FRAME_PATH = DEBUG_DIR / "streamlit_video_preview_frame.jpg"
STREAMLIT_SUMMARY_PATH = DEBUG_DIR / "streamlit_alert_summary.json"
STREAMLIT_LOG_CSV_PATH = LOG_DIR / "streamlit_risk_log.csv"

STREAMLIT_OBS_LAST_FRAME_PATH = DEBUG_DIR / "streamlit_obs_last_frame.jpg"
STREAMLIT_OBS_SUMMARY_PATH = DEBUG_DIR / "streamlit_obs_summary.json"
STREAMLIT_OBS_LOG_CSV_PATH = LOG_DIR / "streamlit_obs_risk_log.csv"
STREAMLIT_OBS_VIDEO_PATH = RESULT_DIR / "streamlit_obs_processed_video.mp4"
STREAMLIT_OBS_ROI_SNAPSHOT_PATH = DEBUG_DIR / "streamlit_obs_roi_snapshot.jpg"
STREAMLIT_VIDEO_ROI_SNAPSHOT_PATH = DEBUG_DIR / "streamlit_video_roi_snapshot.jpg"
STREAMLIT_ROI_SELECTOR_PREVIEW_PATH = DEBUG_DIR / "streamlit_roi_selector_preview.jpg"
PROJECTOR_DEVICE_DIR = PROJECT_ROOT / "data" / "projector_devices"
PROJECTOR_PREVIEW_DIR = DEBUG_DIR / "projector_previews"
PERSON_ONLY_MODEL_NOTICE = (
    "현재 선택한 모델은 person-only 모델입니다. 차량 탐지는 수행하지 않지만, "
    "ROI 안의 사람을 감지하면 위험 경고가 표시됩니다."
)
ROI_ZONE_DIR = PROJECT_ROOT / "data" / "roi_zones"
ROI_BASE_IMAGE_DIR = DEBUG_DIR / "roi_base_images"
ROI_PREVIEW_DIR = DEBUG_DIR / "roi_previews"
REALTIME_SOUND_TRIGGER_LABELS = [
    "While current frame is risky",
    "Stable risk start",
    "Logged alert only",
]
VIDEO_ALERT_SOUND_TRIGGER_LABELS = [
    "Logged alert only",
    "Stable risk start",
    "Every stable risk frame",
]
DEFAULT_VIDEO_ALERT_SOUND_TRIGGER_LABEL = "Every stable risk frame"
VIDEO_ALERT_SOUND_TRIGGER_MAP = {
    "Logged alert only": "logged",
    "Stable risk start": "stable_start",
    "Every stable risk frame": "stable_every",
}
REALTIME_SOUND_TRIGGER_MAP = {
    "While current frame is risky": "stable",
    "Stable risk with cooldown": "stable",
    "Stable risk start": "stable_start",
    "Logged alert only": "logged",
}


def get_realtime_sound_trigger_value(label: str) -> str:
    try:
        return REALTIME_SOUND_TRIGGER_MAP[str(label)]
    except KeyError as exc:
        raise ValueError(f"Unknown realtime sound trigger label: {label}") from exc


def get_default_video_alert_sound_trigger_index() -> int:
    return VIDEO_ALERT_SOUND_TRIGGER_LABELS.index(DEFAULT_VIDEO_ALERT_SOUND_TRIGGER_LABEL)


def get_video_alert_sound_trigger_value(label: str) -> str:
    try:
        return VIDEO_ALERT_SOUND_TRIGGER_MAP[str(label)]
    except KeyError as exc:
        raise ValueError(f"Unknown video alert sound trigger label: {label}") from exc


def model_path_exists(model_path: str) -> bool:
    path = Path(str(model_path)).expanduser()
    if path.is_absolute():
        return path.exists()
    return (PROJECT_ROOT / path).exists() or path.exists()


@st.cache_data(show_spinner=False)
def get_model_options_cached(project_root: str, refresh_token: int = 0) -> list[ModelOption]:
    return build_model_options(Path(project_root))


@st.cache_data(show_spinner=False)
def get_model_class_names_cached(model_path: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def is_person_only_model_option(option: ModelOption | None) -> bool:
    return bool(option and option.model_type == "project_person_only")


def is_person_only_model_type(model_type: str | None) -> bool:
    return str(model_type or "") == "project_person_only"


def get_recommended_confidence(option: ModelOption | None, default_conf: float = CONF_THRESHOLD) -> float:
    if option and getattr(option, "recommended_confidence", None) is not None:
        return float(option.recommended_confidence)
    return float(default_conf)


def get_model_requirement_summary(option: ModelOption | None) -> dict:
    expected_classes = list(option.expected_classes) if option else []
    return {
        "model_type": option.model_type if option else "unknown",
        "expected_classes": expected_classes,
        "person_detection_required": True,
        "car_detection_required": False,
        "person_only": is_person_only_model_option(option),
    }


def build_roi_artifact_paths(prefix: str) -> dict:
    safe_prefix = sanitize_filename_component(prefix)
    ROI_ZONE_DIR.mkdir(parents=True, exist_ok=True)
    ROI_BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ROI_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "danger_zone_json_path": display_path(ROI_ZONE_DIR / f"{safe_prefix}_danger_zone.json"),
        "roi_base_image_path": display_path(ROI_BASE_IMAGE_DIR / f"{safe_prefix}_roi_base_image.jpg"),
        "roi_preview_output_path": display_path(ROI_PREVIEW_DIR / f"{safe_prefix}_roi_selector_preview.jpg"),
    }


def build_projector_artifact_paths(prefix: str) -> dict:
    paths = build_projector_artifact_path(prefix)
    PROJECTOR_DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTOR_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "projector_json_path": display_path(Path(paths["projector_json_path"])),
        "projector_preview_output_path": display_path(Path(paths["projector_preview_output_path"])),
    }


def save_uploaded_file(uploaded_file, output_path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(uploaded_file, "getbuffer"):
        data = uploaded_file.getbuffer()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        data = uploaded_file

    path.write_bytes(bytes(data))
    return str(path)


def open_camera_capture(camera_index: int):
    camera_index = int(camera_index)
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open camera index {camera_index}.")
    return capture


def capture_camera_snapshot(
    camera_index: int,
    output_path: str,
    resize_width: int | None = None,
) -> str:
    ensure_directories()
    capture = open_camera_capture(camera_index)
    try:
        success, frame = capture.read()
    finally:
        capture.release()

    if not success or frame is None:
        raise ValueError(f"Failed to read a frame from camera index {camera_index}.")

    frame = resize_frame_keep_aspect(frame, resize_width)
    return save_debug_image(frame, output_path)


def capture_video_frame_for_roi(
    video_path: str,
    frame_index: int,
    output_path: str,
    resize_width: int | None = None,
) -> str:
    ensure_directories()
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    frame_index = int(frame_index)
    if frame_index < 0:
        raise ValueError("frame_index must be greater than or equal to 0.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        used_frame_index = frame_index
        if frame_count > 0 and used_frame_index >= frame_count:
            used_frame_index = frame_count - 1

        capture.set(cv2.CAP_PROP_POS_FRAMES, used_frame_index)
        success, frame = capture.read()
    finally:
        capture.release()

    if not success or frame is None:
        raise ValueError(f"Failed to read frame {frame_index} from video: {path}")

    frame = resize_frame_keep_aspect(frame, resize_width)
    return save_debug_image(frame, output_path)


def build_roi_selector_command(
    python_executable: str,
    image_path: str,
    output_path: str,
    preview_output_path: str,
    load_existing: bool = True,
    max_display_width: int = 960,
    max_display_height: int = 720,
) -> list[str]:
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "run_step11_roi_selector.py"),
        "--image",
        str(image_path),
        "--output",
        str(output_path),
        "--preview-output",
        str(preview_output_path),
        "--max-display-width",
        str(int(max_display_width)),
        "--max-display-height",
        str(int(max_display_height)),
    ]
    if load_existing:
        command.append("--load-existing")
    return command


def build_projector_selector_command(
    python_executable: str,
    image_path: str,
    output_path: str,
    preview_output_path: str,
    zone_path: str | None = None,
    load_existing: bool = True,
    max_display_width: int = 1280,
    max_display_height: int = 800,
) -> list[str]:
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "run_projector_selector.py"),
        "--image",
        str(image_path),
        "--output",
        str(output_path),
        "--preview-output",
        str(preview_output_path),
        "--max-display-width",
        str(int(max_display_width)),
        "--max-display-height",
        str(int(max_display_height)),
    ]
    if zone_path:
        command.extend(["--zone", str(zone_path)])
    if load_existing:
        command.append("--load-existing")
    return command


def launch_roi_selector(command: list[str], wait: bool = True) -> dict:
    command_text = " ".join(str(item) for item in command)
    try:
        if wait:
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
            return {
                "started": True,
                "finished": True,
                "returncode": int(completed.returncode),
                "command": command_text,
            }
        subprocess.Popen(command, cwd=str(PROJECT_ROOT))
    except Exception as exc:
        return {
            "started": False,
            "finished": False,
            "error": str(exc),
            "command": command_text,
        }
    return {
        "started": True,
        "finished": False,
        "returncode": None,
        "command": command_text,
    }


def load_roi_display_data(
    roi_json_path: str,
    roi_preview_path: str | None = None,
    roi_base_image_path: str | None = None,
) -> dict:
    json_path = Path(roi_json_path)
    preview_path = Path(roi_preview_path) if roi_preview_path else None
    data = {
        "roi_exists": False,
        "roi_json_path": str(roi_json_path),
        "roi_preview_exists": False,
        "roi_preview_path": str(roi_preview_path) if roi_preview_path else None,
        "roi_points": [],
        "roi_point_count": 0,
        "roi_json_data": None,
        "error": None,
        "preview_error": None,
    }

    if json_path.exists():
        try:
            roi_json_data = json.loads(json_path.read_text(encoding="utf-8"))
            roi_points = load_danger_zone(str(json_path))
            data.update(
                {
                    "roi_exists": True,
                    "roi_points": roi_points,
                    "roi_point_count": len(roi_points),
                    "roi_json_data": roi_json_data,
                }
            )
        except Exception as exc:
            data["error"] = str(exc)
    else:
        data["error"] = f"ROI JSON file does not exist: {json_path}"

    if (
        preview_path is not None
        and not preview_path.exists()
        and data["roi_exists"]
        and roi_base_image_path
        and Path(roi_base_image_path).exists()
    ):
        try:
            base_image = load_image(roi_base_image_path)
            preview_image = draw_danger_zone(base_image, data["roi_points"])
            save_debug_image(preview_image, str(preview_path))
        except Exception as exc:
            data["preview_error"] = str(exc)

    if preview_path is not None:
        data["roi_preview_exists"] = preview_path.exists()

    return data


def refresh_roi_after_selector(
    roi_json_path: str,
    roi_preview_path: str,
    roi_base_image_path: str | None = None,
) -> dict:
    st.session_state["selected_roi_output_path"] = str(roi_json_path)
    st.session_state["selected_danger_zone_json_path"] = str(roi_json_path)
    st.session_state["last_roi_json_path"] = str(roi_json_path)
    st.session_state["last_roi_preview_path"] = str(roi_preview_path)
    roi_display_data = load_roi_display_data(
        roi_json_path=roi_json_path,
        roi_preview_path=roi_preview_path,
        roi_base_image_path=roi_base_image_path,
    )
    st.session_state["latest_roi_display_data"] = roi_display_data
    return roi_display_data


def load_projector_display_data(projector_json_path: str, projector_preview_path: str | None = None) -> dict:
    json_path = Path(projector_json_path)
    preview_path = Path(projector_preview_path) if projector_preview_path else None
    data = {
        "projector_exists": False,
        "projector_json_path": str(projector_json_path),
        "projector_preview_exists": False,
        "projector_preview_path": str(projector_preview_path) if projector_preview_path else None,
        "devices": [],
        "device_count": 0,
        "enabled_device_count": 0,
        "projector_json_data": None,
        "error": None,
    }

    if json_path.exists():
        try:
            config = load_projector_config(json_path)
            devices = config.get("devices", [])
            data.update(
                {
                    "projector_exists": True,
                    "devices": devices,
                    "device_count": len(devices),
                    "enabled_device_count": len(enabled_projector_devices(devices)),
                    "projector_json_data": config,
                }
            )
        except Exception as exc:
            data["error"] = str(exc)
    else:
        data["error"] = f"Projector JSON file does not exist: {json_path}"

    if preview_path is not None:
        data["projector_preview_exists"] = preview_path.exists()

    return data


def refresh_projector_after_selector(projector_json_path: str, projector_preview_path: str) -> dict:
    st.session_state["selected_projector_json_path"] = str(projector_json_path)
    st.session_state["selected_projector_preview_output_path"] = str(projector_preview_path)
    st.session_state["last_projector_json_path"] = str(projector_json_path)
    st.session_state["last_projector_preview_path"] = str(projector_preview_path)
    projector_display_data = load_projector_display_data(
        projector_json_path=projector_json_path,
        projector_preview_path=projector_preview_path,
    )
    st.session_state["latest_projector_display_data"] = projector_display_data
    return projector_display_data


def display_projector_status(projector_display_data: dict) -> None:
    preview_path_value = projector_display_data.get("projector_preview_path")
    preview_path = Path(preview_path_value) if preview_path_value else None

    if projector_display_data.get("projector_preview_exists") and preview_path is not None:
        st.image(str(preview_path), caption="Projector Devices Preview", use_container_width=True)
    else:
        st.info("Projector preview image is not available yet.")

    st.caption(f"Projector JSON Path: {projector_display_data.get('projector_json_path')}")
    st.caption(f"Projector Preview Path: {projector_display_data.get('projector_preview_path')}")
    st.caption(f"Projector Device Count: {projector_display_data.get('device_count', 0)}")
    st.caption(f"Enabled Projector Count: {projector_display_data.get('enabled_device_count', 0)}")

    devices = projector_display_data.get("devices") or []
    if devices:
        st.dataframe(
            pd.DataFrame(devices, columns=["id", "name", "x", "y", "enabled", "endpoint"]),
            use_container_width=True,
        )
    elif projector_display_data.get("error"):
        st.warning(projector_display_data["error"])
    else:
        st.info("No projector devices are configured yet.")


def get_risk_person_points(risk_result: dict) -> list[list[int]]:
    points = []
    for detection in risk_result.get("enhanced_detections", []):
        if detection.get("class_name") != "person" or not bool(detection.get("is_risk", False)):
            continue
        anchor_point = detection.get("anchor_point")
        if isinstance(anchor_point, list) and len(anchor_point) == 2:
            points.append([int(anchor_point[0]), int(anchor_point[1])])
    return points


def load_projector_devices_for_pipeline(projector_config_path: str | None) -> tuple[list[dict], str | None]:
    if not projector_config_path:
        return [], None
    try:
        config = load_projector_config(projector_config_path)
        return config.get("devices", []), None
    except Exception as exc:
        return [], str(exc)


def attach_projector_selection_to_risk_result(risk_result: dict, projector_devices: list[dict]) -> dict:
    enriched = dict(risk_result)
    risk_person_points = get_risk_person_points(risk_result)
    assignments = find_nearest_projectors_for_risk_persons(risk_person_points, projector_devices)
    selected_projector = selected_projector_from_assignments(assignments, projector_devices)

    enriched["risk_person_points"] = risk_person_points
    enriched["selected_projector"] = selected_projector
    enriched["selected_projectors"] = assignments
    if risk_person_points and selected_projector is None:
        enriched["projector_warning"] = "No enabled projector devices configured."
    else:
        enriched["projector_warning"] = None
    return enriched


def format_projector_event_fields(risk_result: dict, dispatch_result: dict | None = None) -> dict:
    projector = risk_result.get("selected_projector")
    dispatch_status = (dispatch_result or {}).get("status", "not_dispatched")
    if not projector:
        return {
            "selected_projector_id": "",
            "selected_projector_name": "",
            "selected_projector_x": "",
            "selected_projector_y": "",
            "selected_projector_distance": "",
            "projector_dispatch_status": dispatch_status,
        }
    return {
        "selected_projector_id": projector.get("id", ""),
        "selected_projector_name": projector.get("name", ""),
        "selected_projector_x": int(projector.get("x", 0)),
        "selected_projector_y": int(projector.get("y", 0)),
        "selected_projector_distance": float(projector.get("distance", 0.0)),
        "projector_dispatch_status": dispatch_status,
    }


def display_roi_status(roi_display_data: dict) -> None:
    st.subheader("Current ROI Preview")

    preview_path_value = roi_display_data.get("roi_preview_path")
    preview_path = Path(preview_path_value) if preview_path_value else None
    if roi_display_data.get("roi_preview_exists") and preview_path is not None:
        st.image(str(preview_path), caption="Current ROI Preview", use_container_width=True)
    else:
        st.info("ROI preview image is not available yet.")
        if roi_display_data.get("preview_error"):
            st.caption(f"Preview generation error: {roi_display_data['preview_error']}")

    st.caption(f"ROI JSON Path: {roi_display_data.get('roi_json_path')}")
    st.caption(f"ROI Preview Path: {roi_display_data.get('roi_preview_path')}")
    st.caption(f"ROI Point Count: {roi_display_data.get('roi_point_count', 0)}")

    with st.expander("Show ROI JSON content", expanded=False):
        if roi_display_data.get("roi_json_data") is not None:
            st.json(roi_display_data["roi_json_data"])
        elif roi_display_data.get("error"):
            st.warning(roi_display_data["error"])
        else:
            st.info("ROI JSON content is not available yet.")


def resolve_video_frame_limit(max_frames, process_entire_video: bool = False) -> int | None:
    if process_entire_video or max_frames is None:
        return None

    frame_limit = int(max_frames)
    if frame_limit <= 0:
        return None
    return frame_limit


def build_continuous_realtime_command(
    python_executable: str,
    camera_index: int,
    zone_path: str,
    model_name: str,
    conf: float,
    frame_step: int,
    window_size: int,
    min_risk_count: int,
    cooldown_frames: int,
    resize_width: int,
    use_mock_person: bool,
    save_output_video: bool,
    roi_source_image: str | None = None,
    enable_alert_sound: bool = True,
    sound_trigger: str = "stable",
    sound_cooldown_frames: int = 10,
    beep_frequency: int = 1000,
    beep_duration_ms: int = 250,
) -> list[str]:
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "run_step10_realtime_test.py"),
        "--source",
        "webcam",
        "--camera-index",
        str(int(camera_index)),
        "--zone",
        str(zone_path),
        "--model",
        str(model_name),
        "--conf",
        str(float(conf)),
        "--frame-step",
        str(int(frame_step)),
        "--window-size",
        str(int(window_size)),
        "--min-risk-count",
        str(int(min_risk_count)),
        "--cooldown-frames",
        str(int(cooldown_frames)),
        "--resize-width",
        str(int(resize_width)),
        "--max-frames",
        "0",
    ]
    if use_mock_person:
        command.append("--use-mock-person")
    if save_output_video:
        command.append("--save-output-video")
    if roi_source_image:
        command.extend(["--roi-source-image", str(roi_source_image)])
    if enable_alert_sound:
        command.append("--enable-alert-sound")
    else:
        command.append("--disable-alert-sound")
    command.extend(
        [
            "--sound-trigger",
            str(sound_trigger),
            "--sound-cooldown-frames",
            str(int(sound_cooldown_frames)),
            "--beep-frequency",
            str(int(beep_frequency)),
            "--beep-duration-ms",
            str(int(beep_duration_ms)),
        ]
    )
    return command


def start_continuous_realtime_process(command: list[str]) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to start continuous realtime analysis: {exc}") from exc


def stop_continuous_realtime_process(process) -> dict:
    if process is None:
        return {"stopped": False, "already_stopped": True, "message": "No realtime process is active."}

    if process.poll() is not None:
        return {
            "stopped": False,
            "already_stopped": True,
            "message": "Realtime process is already stopped.",
        }

    try:
        process.terminate()
        process.wait(timeout=5)
        return {"stopped": True, "killed": False, "message": "Realtime process stopped."}
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        return {"stopped": True, "killed": True, "message": "Realtime process killed after timeout."}
    except Exception as exc:
        return {"stopped": False, "already_stopped": False, "message": str(exc)}


def run_image_pipeline(image_path, zone_path, model_name, conf, use_mock_person, projector_config_path: str | None = None) -> dict:
    ensure_directories()
    image = load_image(str(image_path))
    polygon = load_danger_zone(str(zone_path))
    projector_devices, projector_error = load_projector_devices_for_pipeline(projector_config_path)

    if use_mock_person:
        detections = create_mock_person_detections()
    else:
        detector = ObjectDetector(model_name=model_name, conf_threshold=conf)
        detections = detector.detect(image)

    risk_result = evaluate_risk(detections, polygon)
    risk_result = attach_projector_selection_to_risk_result(risk_result, projector_devices)
    dispatch_result = dispatch_projector_alert(risk_result.get("selected_projector"), risk_result, mode="mock") if risk_result["risk_detected"] else {"status": "not_dispatched"}
    risk_result["projector_dispatch_status"] = dispatch_result["status"]
    if projector_error:
        risk_result["projector_warning"] = projector_error
    visualized = draw_risk_assessment(
        image,
        risk_result,
        polygon,
        projector_devices=projector_devices,
        selected_projectors=risk_result.get("selected_projectors"),
    )
    result_image_path = Path(save_debug_image(visualized, str(STREAMLIT_IMAGE_RESULT_PATH)))

    return {
        "image_path": str(image_path),
        "result_image_path": str(result_image_path),
        "original_bgr": image,
        "result_bgr": visualized,
        "risk_result": risk_result,
        "projector_config_path": projector_config_path,
        "projector_error": projector_error,
    }


def run_video_pipeline(
    video_path,
    zone_path,
    model_name,
    conf,
    max_frames,
    frame_step,
    window_size,
    min_risk_count,
    cooldown_frames,
    use_mock_person,
    process_entire_video: bool = False,
    projector_config_path: str | None = None,
) -> dict:
    ensure_directories()
    max_frames_limit = resolve_video_frame_limit(max_frames, process_entire_video)
    process_entire_video = max_frames_limit is None

    video_path = Path(video_path)
    if not video_path.exists():
        fallback_image = SAMPLE_IMAGE_DIR / "test.jpg"
        create_dummy_video_from_image(str(fallback_image), str(video_path), frame_count=30, fps=10.0)

    if STREAMLIT_LOG_CSV_PATH.exists():
        STREAMLIT_LOG_CSV_PATH.unlink()

    polygon = load_danger_zone(str(zone_path))
    projector_devices, projector_error = load_projector_devices_for_pipeline(projector_config_path)
    video_info = get_video_info(str(video_path))
    temporal_filter = TemporalRiskFilter(window_size=window_size, min_risk_count=min_risk_count)
    alert_logger = AlertLogger(
        log_csv_path=str(STREAMLIT_LOG_CSV_PATH),
        alert_image_dir=str(ALERT_DIR),
        cooldown_frames=cooldown_frames,
        image_prefix="streamlit_alert_frame",
    )

    detector = None
    if not use_mock_person:
        detector = ObjectDetector(model_name=model_name, conf_threshold=conf)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    writer = create_video_writer(
        str(STREAMLIT_VIDEO_PATH),
        video_info["width"],
        video_info["height"],
        video_info["fps"],
    )

    mode = "MOCK_PERSON" if use_mock_person else "YOLO"
    frame_results = []
    processed_frames = 0
    raw_risk_frames = 0
    stable_risk_frames = 0
    logged_alert_count = 0
    max_person_count = 0
    max_car_count = 0
    max_risk_person_count = 0
    frame_index = 0
    preview_saved = False

    try:
        while max_frames_limit is None or processed_frames < max_frames_limit:
            success, frame = capture.read()
            if not success or frame is None:
                break

            if frame_index % int(frame_step) != 0:
                frame_index += 1
                continue

            detections = create_mock_person_detections() if use_mock_person else detector.detect(frame)
            risk_result = evaluate_risk(detections, polygon)
            risk_result = attach_projector_selection_to_risk_result(risk_result, projector_devices)
            temporal_result = temporal_filter.update(risk_result["risk_detected"])
            risk_summary = summarize_frame_result(frame_index, risk_result)

            dispatch_result = (
                dispatch_projector_alert(risk_result.get("selected_projector"), risk_result, mode="mock")
                if temporal_result["stable_risk"]
                else {"status": "not_dispatched"}
            )
            visualized = draw_risk_assessment(
                frame,
                risk_result,
                polygon,
                projector_devices=projector_devices,
                selected_projectors=risk_result.get("selected_projectors"),
                compact_summary=True,
            )

            timestamp_sec = frame_index / video_info["fps"] if video_info["fps"] else 0.0
            event = {
                "frame_index": int(frame_index),
                "timestamp_sec": round(float(timestamp_sec), 4),
                "raw_risk": bool(temporal_result["raw_risk"]),
                "stable_risk": bool(temporal_result["stable_risk"]),
                "risk_level": risk_summary["risk_level"],
                "person_count": int(risk_summary["person_count"]),
                "car_count": int(risk_summary["car_count"]),
                "risk_person_count": int(risk_summary["risk_person_count"]),
                "risk_count": int(temporal_result["risk_count"]),
                "window_size": int(temporal_result["window_size"]),
                "min_risk_count": int(temporal_result["min_risk_count"]),
            }
            event.update(format_projector_event_fields(risk_result, dispatch_result))
            event["selected_projector"] = risk_result.get("selected_projector")
            event["selected_projectors"] = risk_result.get("selected_projectors", [])
            event["projector_warning"] = risk_result.get("projector_warning")
            log_result = alert_logger.log_if_needed(visualized, event)
            if log_result["logged"]:
                logged_alert_count += 1
                cv2.putText(
                    visualized,
                    "ALERT LOGGED",
                    (16, visualized.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(visualized)
            if not preview_saved:
                save_debug_image(visualized, str(STREAMLIT_PREVIEW_FRAME_PATH))
                preview_saved = True

            frame_summary = {
                **event,
                "logged": bool(log_result["logged"]),
                "alert_image_path": log_result["image_path"],
            }
            frame_results.append(frame_summary)
            raw_risk_frames += int(frame_summary["raw_risk"])
            stable_risk_frames += int(frame_summary["stable_risk"])
            max_person_count = max(max_person_count, frame_summary["person_count"])
            max_car_count = max(max_car_count, frame_summary["car_count"])
            max_risk_person_count = max(max_risk_person_count, frame_summary["risk_person_count"])
            processed_frames += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise ValueError("No frames were processed from the input video.")

    summary = {
        "mode": mode,
        "video_path": display_path(video_path),
        "zone_path": display_path(Path(zone_path)),
        "process_entire_video": bool(process_entire_video),
        "max_frames_limit": int(max_frames_limit) if max_frames_limit is not None else None,
        "output_fps": float(video_info["fps"]),
        "processed_frames": int(processed_frames),
        "raw_risk_frames": int(raw_risk_frames),
        "stable_risk_frames": int(stable_risk_frames),
        "logged_alert_count": int(logged_alert_count),
        "max_person_count": int(max_person_count),
        "max_car_count": int(max_car_count),
        "max_risk_person_count": int(max_risk_person_count),
        "window_size": int(window_size),
        "min_risk_count": int(min_risk_count),
        "cooldown_frames": int(cooldown_frames),
        "projector_config_path": display_path(Path(projector_config_path)) if projector_config_path else None,
        "projector_device_count": int(len(projector_devices)),
        "projector_error": projector_error,
        "log_csv_path": display_path(STREAMLIT_LOG_CSV_PATH),
        "alert_image_dir": display_path(ALERT_DIR),
        "frame_results": frame_results,
    }
    save_video_summary(summary, str(STREAMLIT_SUMMARY_PATH))

    return {
        "summary": summary,
        "video_path": str(STREAMLIT_VIDEO_PATH),
        "output_video_path": str(STREAMLIT_VIDEO_PATH),
        "web_video_path": None,
        "preview_frame_path": str(STREAMLIT_PREVIEW_FRAME_PATH),
        "summary_path": str(STREAMLIT_SUMMARY_PATH),
        "csv_path": str(STREAMLIT_LOG_CSV_PATH),
        "csv_log_path": str(STREAMLIT_LOG_CSV_PATH),
        "video_display_status": "original",
    }


def run_obs_camera_pipeline(
    camera_index: int,
    zone_path: str,
    model_name: str,
    conf: float,
    max_live_frames: int,
    frame_step: int,
    window_size: int,
    min_risk_count: int,
    cooldown_frames: int,
    resize_width: int,
    use_mock_person: bool,
    save_output_video: bool,
    roi_source_image_path: str | None = None,
    enable_alert_sound: bool = True,
    sound_trigger: str = "stable",
    sound_cooldown_frames: int = 10,
    beep_frequency: int = 1000,
    beep_duration_ms: int = 250,
    projector_config_path: str | None = None,
    frame_placeholder=None,
) -> dict:
    ensure_directories()
    max_live_frames = int(max_live_frames)
    frame_step = int(frame_step)
    frame_limit = None if max_live_frames <= 0 else max_live_frames
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1.")

    if STREAMLIT_OBS_LOG_CSV_PATH.exists():
        STREAMLIT_OBS_LOG_CSV_PATH.unlink()

    polygon = load_danger_zone(str(zone_path))
    projector_devices_original, projector_error = load_projector_devices_for_pipeline(projector_config_path)
    roi_source_size = None
    roi_source_size_error = None
    if roi_source_image_path:
        try:
            roi_source_size = get_image_size(roi_source_image_path)
        except Exception as exc:
            roi_source_size_error = str(exc)
    temporal_filter = TemporalRiskFilter(window_size=window_size, min_risk_count=min_risk_count)
    alert_logger = AlertLogger(
        log_csv_path=str(STREAMLIT_OBS_LOG_CSV_PATH),
        alert_image_dir=str(ALERT_DIR),
        cooldown_frames=cooldown_frames,
        image_prefix="streamlit_obs_alert_frame",
    )
    alert_sound = RealtimeAlertSound(
        enabled=enable_alert_sound,
        trigger_mode=sound_trigger,
        cooldown_frames=sound_cooldown_frames,
        frequency=beep_frequency,
        duration_ms=beep_duration_ms,
    )
    stats = RealtimeStats()
    max_person_count = 0
    max_car_count = 0

    detector = None
    if not use_mock_person:
        detector = ObjectDetector(model_name=model_name, conf_threshold=conf)

    capture = open_camera_capture(camera_index)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 20.0

    writer = None
    last_visualized_frame = None
    frame_results = []
    read_frames = 0
    frame_index = 0
    mode = "MOCK_PERSON" if use_mock_person else "YOLO"

    try:
        while frame_limit is None or read_frames < frame_limit:
            success, frame = capture.read()
            if not success or frame is None:
                break

            read_frames += 1
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            original_height, original_width = frame.shape[:2]
            resized = resize_frame_keep_aspect(frame, resize_width)
            analysis_height, analysis_width = resized.shape[:2]
            roi_source_width, roi_source_height = roi_source_size or (original_width, original_height)
            scaled_polygon = scale_polygon_to_frame(
                polygon,
                source_width=roi_source_width,
                source_height=roi_source_height,
                target_width=analysis_width,
                target_height=analysis_height,
            )
            projector_devices = scale_projector_devices_to_frame(
                projector_devices_original,
                source_width=roi_source_width,
                source_height=roi_source_height,
                target_width=analysis_width,
                target_height=analysis_height,
            )
            detections = create_mock_person_detections() if use_mock_person else detector.detect(resized)
            risk_result = evaluate_risk(detections, scaled_polygon)
            risk_result = attach_projector_selection_to_risk_result(risk_result, projector_devices)
            temporal_result = temporal_filter.update(risk_result["risk_detected"])
            risk_summary = summarize_frame_result(frame_index, risk_result)

            dispatch_result = (
                dispatch_projector_alert(risk_result.get("selected_projector"), risk_result, mode="mock")
                if temporal_result["stable_risk"]
                else {"status": "not_dispatched"}
            )
            base_visualized = draw_risk_assessment(
                resized,
                risk_result,
                scaled_polygon,
                projector_devices=projector_devices,
                selected_projectors=risk_result.get("selected_projectors"),
            )
            base_visualized = draw_temporal_status(base_visualized, temporal_result)

            timestamp_sec = frame_index / source_fps if source_fps else 0.0
            event = {
                "frame_index": int(frame_index),
                "timestamp_sec": round(float(timestamp_sec), 4),
                "raw_risk": bool(temporal_result["raw_risk"]),
                "stable_risk": bool(temporal_result["stable_risk"]),
                "risk_level": risk_summary["risk_level"],
                "person_count": int(risk_summary["person_count"]),
                "car_count": int(risk_summary["car_count"]),
                "risk_person_count": int(risk_summary["risk_person_count"]),
                "risk_count": int(temporal_result["risk_count"]),
                "window_size": int(temporal_result["window_size"]),
                "min_risk_count": int(temporal_result["min_risk_count"]),
            }
            event.update(format_projector_event_fields(risk_result, dispatch_result))
            event["selected_projector"] = risk_result.get("selected_projector")
            event["selected_projectors"] = risk_result.get("selected_projectors", [])
            event["projector_warning"] = risk_result.get("projector_warning")
            will_log = alert_logger.should_log(
                frame_index=frame_index,
                stable_risk=event["stable_risk"],
            )
            stats.update(risk_result, temporal_result, logged=will_log)
            fps = stats.calculate_fps()
            visualized = draw_realtime_overlay(
                base_visualized,
                frame_index=frame_index,
                fps=fps,
                source_name="OBS Camera",
            )
            if will_log:
                cv2.putText(
                    visualized,
                    "OBS ALERT LOGGED",
                    (16, visualized.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            log_result = alert_logger.log_if_needed(visualized, event)
            if will_log and not log_result["logged"]:
                stats.logged_alert_count = max(0, stats.logged_alert_count - 1)
            sound_result = alert_sound.update_and_play(
                frame_index=frame_index,
                stable_risk=event["stable_risk"],
                logged=bool(log_result["logged"]),
            )
            if sound_result["sound_played"]:
                cv2.putText(
                    visualized,
                    "ALERT SOUND",
                    (16, max(28, visualized.shape[0] - 56)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if save_output_video and writer is None:
                height, width = visualized.shape[:2]
                writer = create_video_writer(str(STREAMLIT_OBS_VIDEO_PATH), width, height, source_fps)
            if writer is not None:
                writer.write(visualized)

            frame_summary = {
                **event,
                "logged": bool(log_result["logged"]),
                "sound_played": bool(sound_result["sound_played"]),
                "alert_image_path": log_result["image_path"],
            }
            frame_results.append(frame_summary)
            max_person_count = max(max_person_count, frame_summary["person_count"])
            max_car_count = max(max_car_count, frame_summary["car_count"])
            last_visualized_frame = visualized

            if frame_placeholder is not None:
                frame_placeholder.image(
                    cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB),
                    caption=f"OBS Camera Frame {frame_index}",
                    use_container_width=True,
                )

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if stats.processed_frames == 0 or last_visualized_frame is None:
        raise ValueError("No frames were processed from the OBS camera input.")

    save_debug_image(last_visualized_frame, str(STREAMLIT_OBS_LAST_FRAME_PATH))
    avg_fps = stats.calculate_fps()

    summary = {
        "mode": mode,
        "source": "OBS Camera",
        "camera_index": int(camera_index),
        "zone_path": display_path(Path(zone_path)),
        "processed_frames": int(stats.processed_frames),
        "raw_risk_frames": int(stats.raw_risk_frames),
        "stable_risk_frames": int(stats.stable_risk_frames),
        "logged_alert_count": int(stats.logged_alert_count),
        "max_person_count": int(max_person_count),
        "max_car_count": int(max_car_count),
        "max_risk_person_count": int(stats.max_risk_person_count),
        "avg_fps": float(avg_fps),
        "max_live_frames": int(max_live_frames),
        "run_continuously": bool(frame_limit is None),
        "frame_step": int(frame_step),
        "resize_width": int(resize_width),
        "roi_source_image_path": display_path(roi_source_image_path) if roi_source_image_path else None,
        "roi_source_width": int(roi_source_size[0]) if roi_source_size else None,
        "roi_source_height": int(roi_source_size[1]) if roi_source_size else None,
        "roi_source_size_error": roi_source_size_error,
        "projector_config_path": display_path(Path(projector_config_path)) if projector_config_path else None,
        "projector_device_count": int(len(projector_devices_original)),
        "projector_error": projector_error,
        "window_size": int(window_size),
        "min_risk_count": int(min_risk_count),
        "cooldown_frames": int(cooldown_frames),
        "alert_sound_enabled": bool(enable_alert_sound),
        "alert_sound_trigger": str(sound_trigger),
        "alert_sound_count": int(alert_sound.sound_count),
        "log_csv_path": display_path(STREAMLIT_OBS_LOG_CSV_PATH),
        "last_frame_path": display_path(STREAMLIT_OBS_LAST_FRAME_PATH),
        "output_video_path": display_path(STREAMLIT_OBS_VIDEO_PATH) if save_output_video else None,
        "alert_image_dir": display_path(ALERT_DIR),
        "frame_results": frame_results,
    }
    save_realtime_summary(summary, str(STREAMLIT_OBS_SUMMARY_PATH))

    return {
        "summary": summary,
        "summary_path": str(STREAMLIT_OBS_SUMMARY_PATH),
        "last_frame_path": str(STREAMLIT_OBS_LAST_FRAME_PATH),
        "csv_path": str(STREAMLIT_OBS_LOG_CSV_PATH),
        "output_video_path": str(STREAMLIT_OBS_VIDEO_PATH) if save_output_video else None,
    }


def load_csv_if_exists(csv_path) -> pd.DataFrame | None:
    path = Path(csv_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def convert_video_for_streamlit(input_path: str | Path, output_path: str | Path) -> str:
    input_path = Path(input_path)
    output_path = Path(output_path)
    read_video_bytes(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = get_ffmpeg_executable()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg executable was not found. Install imageio-ffmpeg or add ffmpeg to PATH.")

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        details = completed.stderr[-700:] or completed.stdout[-700:] or "ffmpeg conversion failed."
        raise RuntimeError(details)
    read_video_bytes(output_path)
    return str(output_path)


def create_alarm_wav_bytes(
    duration_sec: float = 0.75,
    sample_rate: int = 44100,
    frequency_hz: float = 880.0,
    volume: float = 0.45,
) -> bytes:
    duration_sec = float(duration_sec)
    sample_rate = int(sample_rate)
    frequency_hz = float(frequency_hz)
    volume = float(volume)
    if duration_sec <= 0:
        raise ValueError("duration_sec must be greater than 0.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be greater than 0.")

    sample_count = max(1, int(round(duration_sec * sample_rate)))
    timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    tone = np.sin(2.0 * math.pi * frequency_hz * timeline)
    pulse = np.where((timeline % 0.24) < 0.16, 1.0, 0.0)

    fade_samples = min(int(sample_rate * 0.02), max(1, sample_count // 2))
    envelope = np.ones(sample_count, dtype=np.float32)
    envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples)
    envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)

    audio = np.clip(tone * pulse * envelope * volume, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def summary_has_audio_alert(summary: dict) -> bool:
    return int(summary.get("logged_alert_count", 0) or 0) > 0 or int(
        summary.get("stable_risk_frames", 0) or 0
    ) > 0


def collect_alert_sound_times(summary: dict) -> list[float]:
    frame_results = list(summary.get("frame_results", []) or [])
    output_fps = float(summary.get("output_fps") or summary.get("fps") or 10.0)
    if output_fps <= 0:
        output_fps = 10.0

    logged_indices = [index for index, item in enumerate(frame_results) if item.get("logged")]
    if logged_indices:
        return [round(index / output_fps, 4) for index in logged_indices]

    alert_indices = []
    previous_stable_risk = False
    for index, item in enumerate(frame_results):
        stable_risk = bool(item.get("stable_risk"))
        if stable_risk and not previous_stable_risk:
            alert_indices.append(index)
        previous_stable_risk = stable_risk
    return [round(index / output_fps, 4) for index in alert_indices]


def write_alert_audio_track(
    alert_times_sec: list[float],
    duration_sec: float,
    output_path: str | Path,
    sample_rate: int = 44100,
    beep_duration_sec: float = 0.55,
) -> str:
    if not alert_times_sec:
        raise ValueError("alert_times_sec must contain at least one alert time.")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be greater than 0.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_duration = max(float(duration_sec), max(alert_times_sec) + beep_duration_sec)
    total_samples = max(1, int(math.ceil(total_duration * sample_rate)))
    track = np.zeros(total_samples, dtype=np.float32)

    beep_bytes = create_alarm_wav_bytes(
        duration_sec=beep_duration_sec,
        sample_rate=sample_rate,
    )
    with wave.open(io.BytesIO(beep_bytes), "rb") as wav_file:
        beep = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).astype(np.float32)
    beep = beep / 32767.0

    for alert_time in alert_times_sec:
        start = max(0, int(round(float(alert_time) * sample_rate)))
        end = min(total_samples, start + len(beep))
        if end <= start:
            continue
        track[start:end] = np.maximum(track[start:end], beep[: end - start])

    pcm = (np.clip(track, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return str(output_path)


def create_video_with_alert_audio(video_path: str, summary: dict) -> dict:
    if not summary_has_audio_alert(summary):
        return {"created": False, "reason": "No stable risk or logged alert was detected."}

    alert_times = collect_alert_sound_times(summary)
    if not alert_times:
        return {"created": False, "reason": "No alert timestamps were available."}

    output_fps = float(summary.get("output_fps") or summary.get("fps") or 10.0)
    if output_fps <= 0:
        output_fps = 10.0
    processed_frames = int(summary.get("processed_frames", 0) or 0)
    duration_sec = max(processed_frames / output_fps, max(alert_times) + 0.6)

    audio_path = write_alert_audio_track(
        alert_times,
        duration_sec=duration_sec,
        output_path=STREAMLIT_VIDEO_ALERT_AUDIO_TRACK_PATH,
    )

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {
            "created": False,
            "reason": "ffmpeg was not found, so the alert sound could not be embedded in the MP4.",
            "audio_path": audio_path,
            "alert_times_sec": alert_times,
        }

    output_path = STREAMLIT_VIDEO_WITH_ALERT_AUDIO_PATH
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return {
            "created": False,
            "reason": completed.stderr[-500:] or "ffmpeg failed to create the alert-audio video.",
            "audio_path": audio_path,
            "alert_times_sec": alert_times,
        }
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return {
            "created": False,
            "reason": "ffmpeg finished but the alert-audio video file was not created.",
            "audio_path": audio_path,
            "alert_times_sec": alert_times,
        }
    return {
        "created": True,
        "video_path": str(output_path),
        "audio_path": audio_path,
        "alert_times_sec": alert_times,
    }


def display_web_alert_sound(summary: dict, source_label: str) -> bool:
    if not summary_has_audio_alert(summary):
        return False

    audio_bytes = create_alarm_wav_bytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    st.warning(
        f"{source_label}: danger detected. "
        f"stable_risk_frames={summary.get('stable_risk_frames', 0)}, "
        f"logged_alert_count={summary.get('logged_alert_count', 0)}"
    )
    st.audio(audio_bytes, format="audio/wav")
    components.html(
        f"""
        <audio id="parking-alert-audio" autoplay>
          <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        <script>
          const audio = document.getElementById("parking-alert-audio");
          if (audio) {{
            audio.play().catch(() => null);
          }}
        </script>
        """,
        height=0,
    )
    st.caption("If the browser blocks autoplay, press play on the audio control above.")
    return True


def show_video_with_fallback(
    video_path: str | Path,
    preview_image_path: str | Path | None = None,
    caption: str = "Processed video",
    download_label: str | None = None,
) -> bool:
    path = Path(video_path)
    preview_path = Path(preview_image_path) if preview_image_path else None
    video_bytes = None
    file_size = 0

    st.caption(f"{caption} path: {display_path(path)}")
    if preview_path is not None:
        st.caption(f"preview frame path: {display_path(preview_path)}")

    try:
        file_size_mb = get_file_size_mb(path)
        file_size = path.stat().st_size
        st.caption(f"{caption} file size: {file_size_mb:.2f} MB")

        video_bytes = read_video_bytes(path)
        try:
            st.video(video_bytes)
        except Exception as exc:
            st.warning(f"Video preview failed: {exc}")
            st.info(
                "The browser may not support the video codec. "
                "Use the download button below or check the preview frame."
            )
        finally:
            st.download_button(
                label=download_label or f"Download {caption}",
                data=video_bytes,
                file_name=path.name,
                mime="video/mp4",
                key=f"download_{sanitize_filename_component(caption)}_{path.name}_{file_size}",
            )
    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except OSError as exc:
        st.error(f"Failed to read video file: {exc}")
    except Exception as exc:
        st.warning(f"Video display failed: {exc}")
    finally:
        if preview_path is not None:
            try:
                if preview_path.exists():
                    st.image(str(preview_path), caption="Preview frame", use_container_width=True)
                else:
                    st.warning(f"Preview frame file does not exist: {display_path(preview_path)}")
            except Exception as exc:
                st.warning(f"Failed to display preview frame: {exc}")

    return video_bytes is not None


def display_video_result(video_path: str, preview_image_path: str | None = None):
    return show_video_with_fallback(
        video_path=video_path,
        preview_image_path=preview_image_path,
        caption="Result video",
        download_label="Download Result Video",
    )


def display_summary_metrics(summary: dict, model_type: str | None = None) -> None:
    metric_keys = [
        "processed_frames",
        "raw_risk_frames",
        "stable_risk_frames",
        "logged_alert_count",
        "max_person_count",
        "max_risk_person_count",
    ]
    if not is_person_only_model_type(model_type):
        metric_keys.insert(5, "max_car_count")

    cols = st.columns(len(metric_keys))
    for col, key in zip(cols, metric_keys):
        col.metric(key, summary.get(key, 0))
    if is_person_only_model_type(model_type):
        st.caption("car detection is not used for risk judgement.")


def display_obs_summary_metrics(summary: dict, model_type: str | None = None) -> None:
    metric_keys = [
        "processed_frames",
        "raw_risk_frames",
        "stable_risk_frames",
        "logged_alert_count",
        "max_person_count",
        "max_risk_person_count",
        "avg_fps",
    ]
    if not is_person_only_model_type(model_type):
        metric_keys.insert(5, "max_car_count")

    cols = st.columns(len(metric_keys))
    for col, key in zip(cols, metric_keys):
        value = summary.get(key, 0)
        col.metric(key, f"{float(value):.2f}" if key == "avg_fps" else value)
    if is_person_only_model_type(model_type):
        st.caption("car detection is not used for risk judgement.")


def display_image_risk_metrics(risk_result: dict, model_type: str | None = None) -> None:
    metric_keys = [
        "risk_detected",
        "risk_level",
        "person_count",
        "risk_person_count",
    ]
    if not is_person_only_model_type(model_type):
        metric_keys.insert(3, "car_count")

    cols = st.columns(len(metric_keys))
    for col, key in zip(cols, metric_keys):
        col.metric(key, risk_result.get(key, 0))
    if is_person_only_model_type(model_type):
        st.caption("car detection is not used for risk judgement.")


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_image_mode(uploaded_file, settings) -> None:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        image_path = Path(save_uploaded_file(uploaded_file, DEBUG_DIR / f"streamlit_uploaded_image{suffix}"))
    else:
        image_path = SAMPLE_IMAGE_DIR / "test.jpg"

    with st.spinner("Running image risk detection..."):
        result = run_image_pipeline(
            image_path=image_path,
            zone_path=settings["zone_path"],
            model_name=settings["model_name"],
            conf=settings["conf"],
            use_mock_person=settings["use_mock_person"],
            projector_config_path=settings.get("projector_config_path"),
        )

    risk_result = result["risk_result"]
    st.subheader("Image Result")
    left, right = st.columns(2)
    left.image(bgr_to_rgb(result["original_bgr"]), caption="Original Image", use_container_width=True)
    right.image(bgr_to_rgb(result["result_bgr"]), caption="Risk Result", use_container_width=True)

    display_image_risk_metrics(risk_result, settings.get("selected_model_type"))
    st.json(risk_result["enhanced_detections"])
    st.caption(f"Saved result image: {display_path(Path(result['result_image_path']))}")


def render_video_mode(uploaded_file, settings) -> None:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        video_path = Path(save_uploaded_file(uploaded_file, RESULT_DIR / f"streamlit_uploaded_video{suffix}"))
    else:
        video_path = SAMPLE_VIDEO_DIR / "test.mp4"

    with st.spinner("Running video alert pipeline..."):
        result = run_video_pipeline(
            video_path=video_path,
            zone_path=settings["zone_path"],
            model_name=settings["model_name"],
            conf=settings["conf"],
            max_frames=settings["max_frames"],
            frame_step=settings["frame_step"],
            window_size=settings["window_size"],
            min_risk_count=settings["min_risk_count"],
            cooldown_frames=settings["cooldown_frames"],
            use_mock_person=settings["use_mock_person"],
            process_entire_video=settings["process_entire_video"],
            projector_config_path=settings.get("projector_config_path"),
        )

    summary = result["summary"]
    st.subheader("Video Result")
    display_summary_metrics(summary, settings.get("selected_model_type"))

    st.subheader("Original Uploaded Video" if uploaded_file is not None else "Original Input Video")
    show_video_with_fallback(
        video_path=video_path,
        caption="Original Uploaded Video" if uploaded_file is not None else "Original Input Video",
        download_label="Download original video",
    )

    st.subheader("Processed Result Video")
    processed_video_path = result["output_video_path"]
    alert_sound_result = {
        "with_sound": False,
        "reason": "disabled",
        "video_path": processed_video_path,
        "alert_timestamps": [],
    }
    processed_source_video_path = processed_video_path
    processed_caption = "Processed Result Video"

    if settings.get("enable_alert_sound", False):
        st.caption(f"alert sound trigger: {settings.get('alert_sound_trigger_mode', 'logged')}")
        with st.spinner("Adding alert sound to the result video..."):
            alert_sound_result = add_alert_sound_to_video(
                video_path=processed_video_path,
                summary=summary,
                output_video_path=str(STREAMLIT_VIDEO_WITH_ALERT_SOUND_PATH),
                output_wav_path=str(STREAMLIT_ALERT_BEEP_WAV_PATH),
                trigger_mode=settings.get("alert_sound_trigger_mode", "logged"),
                beep_frequency=settings.get("beep_frequency", 1000.0),
                beep_duration_sec=settings.get("beep_duration_sec", 0.25),
                volume=settings.get("beep_volume", 0.5),
            )

        result["alert_sound_result"] = alert_sound_result
        if alert_sound_result.get("with_sound"):
            processed_source_video_path = alert_sound_result["video_with_sound_path"]
            processed_caption = "Processed Result Video with Alert Sound"
            result["alert_sound_status"] = "success"
            st.success("Alert sound was embedded in the processed result video.")
            st.caption(f"alert timestamps sec: {alert_sound_result.get('alert_timestamps', [])}")
            st.caption(f"alert wav path: {display_path(Path(alert_sound_result['audio_wav_path']))}")
        else:
            reason = alert_sound_result.get("reason", "unknown error")
            result["alert_sound_status"] = "skipped" if reason == "no_alert_timestamps" else "failed"
            if reason == "no_alert_timestamps":
                st.info("No alert timestamps were found for the selected trigger mode. Showing the silent result video.")
            else:
                st.warning(f"Could not embed alert sound. Showing the silent result video. {reason}")
    else:
        result["alert_sound_result"] = alert_sound_result
        result["alert_sound_status"] = "disabled"

    st.caption(f"result video path: {display_path(Path(processed_source_video_path))}")
    try:
        st.caption(f"file size in MB: {get_file_size_mb(processed_source_video_path):.2f}")
    except Exception as exc:
        st.warning(f"Could not read result video size: {exc}")
    st.caption(f"preview frame path: {display_path(Path(result['preview_frame_path']))}")

    display_video_path = processed_source_video_path
    web_video_path = None
    video_display_status = "original"
    try:
        with st.spinner("Converting processed video for browser playback..."):
            web_output_path = (
                STREAMLIT_VIDEO_WITH_ALERT_SOUND_WEB_PATH
                if alert_sound_result.get("with_sound")
                else STREAMLIT_VIDEO_WEB_PATH
            )
            web_video_path = convert_video_for_streamlit(processed_source_video_path, web_output_path)
        display_video_path = web_video_path
        video_display_status = "converted"
        result["web_video_path"] = web_video_path
        result["video_display_status"] = video_display_status
        st.success("Created a web-compatible MP4 for Streamlit playback.")
    except Exception as exc:
        video_display_status = "fallback"
        result["video_display_status"] = video_display_status
        st.warning(f"Could not create a web-compatible MP4. Showing the original processed video instead. {exc}")

    show_video_with_fallback(
        display_video_path,
        result["preview_frame_path"],
        caption=processed_caption,
        download_label="Download processed video",
    )
    if web_video_path:
        try:
            web_video_bytes = read_video_bytes(web_video_path)
            st.download_button(
                label="Download web-compatible video",
                data=web_video_bytes,
                file_name=Path(web_video_path).name,
                mime="video/mp4",
                key=f"download_web_compatible_{Path(web_video_path).name}_{Path(web_video_path).stat().st_size}",
            )
        except Exception as exc:
            st.warning(f"Web-compatible video download is unavailable: {exc}")

    st.subheader("Summary JSON")
    st.json(summary)

    csv_df = load_csv_if_exists(result["csv_path"])
    if csv_df is not None:
        st.subheader("CSV Log")
        st.dataframe(csv_df, use_container_width=True)

    alert_paths = [
        item.get("alert_image_path")
        for item in summary.get("frame_results", [])
        if item.get("logged") and item.get("alert_image_path")
    ]
    if alert_paths:
        st.subheader("Captured Alert Images")
        for alert_path in alert_paths:
            st.image(alert_path, caption=display_path(Path(alert_path)), use_container_width=True)

    st.caption(f"Saved result video: {display_path(Path(result['video_path']))}")
    if alert_sound_result.get("with_sound"):
        st.caption(f"Saved alert-sound video: {display_path(Path(alert_sound_result['video_with_sound_path']))}")
    if web_video_path:
        st.caption(f"Saved web-compatible video: {display_path(Path(web_video_path))}")
    st.caption(f"video_display_status: {video_display_status}")
    st.caption(f"alert_sound_status: {result['alert_sound_status']}")
    st.caption(f"Saved summary: {display_path(Path(result['summary_path']))}")
    st.caption(f"Saved CSV log: {display_path(Path(result['csv_path']))}")


def render_obs_camera_mode(settings) -> None:
    st.subheader("OBS Camera Result")
    if settings["max_live_frames"] <= 0:
        st.info(
            "Continuous OBS mode is configured with no frame limit. "
            "Use Start Continuous OBS Analysis in the sidebar, then stop it with Q or the stop button."
        )
        return

    st.info(
        "Streamlit OBS mode processes only max_live_frames for safety. "
        "For longer realtime runs, use scripts/run_step10_realtime_test.py."
    )
    frame_placeholder = st.empty()

    with st.spinner("Running OBS camera analysis..."):
        result = run_obs_camera_pipeline(
            camera_index=settings["camera_index"],
            zone_path=settings["zone_path"],
            model_name=settings["model_name"],
            conf=settings["conf"],
            max_live_frames=settings["max_live_frames"],
            frame_step=settings["frame_step"],
            window_size=settings["window_size"],
            min_risk_count=settings["min_risk_count"],
            cooldown_frames=settings["cooldown_frames"],
            resize_width=settings["resize_width"],
            use_mock_person=settings["use_mock_person"],
            save_output_video=settings["save_output_video"],
            roi_source_image_path=settings["roi_source_image_path"],
            enable_alert_sound=settings["enable_realtime_alert_sound"],
            sound_trigger=settings["realtime_sound_trigger"],
            sound_cooldown_frames=settings["sound_cooldown_frames"],
            beep_frequency=settings["realtime_beep_frequency"],
            beep_duration_ms=settings["realtime_beep_duration_ms"],
            projector_config_path=settings.get("projector_config_path"),
            frame_placeholder=frame_placeholder,
        )

    summary = result["summary"]
    display_obs_summary_metrics(summary, settings.get("selected_model_type"))
    display_web_alert_sound(summary, "OBS Camera Web Alert")

    last_frame_path = Path(result["last_frame_path"])
    if last_frame_path.exists():
        st.image(str(last_frame_path), caption="Last Processed OBS Frame", use_container_width=True)
    else:
        st.warning(f"Last frame file does not exist: {display_path(last_frame_path)}")

    if result["output_video_path"]:
        st.subheader("OBS Output Video")
        display_video_result(result["output_video_path"], result["last_frame_path"])

    st.subheader("Summary JSON")
    st.json(summary)

    csv_df = load_csv_if_exists(result["csv_path"])
    if csv_df is not None:
        st.subheader("CSV Log")
        st.dataframe(csv_df, use_container_width=True)

    alert_paths = [
        item.get("alert_image_path")
        for item in summary.get("frame_results", [])
        if item.get("logged") and item.get("alert_image_path")
    ]
    if alert_paths:
        st.subheader("Captured Alert Images")
        for alert_path in alert_paths:
            st.image(alert_path, caption=display_path(Path(alert_path)), use_container_width=True)

    st.caption(f"Saved last frame: {display_path(Path(result['last_frame_path']))}")
    st.caption(f"Saved summary: {display_path(Path(result['summary_path']))}")
    st.caption(f"Saved CSV log: {display_path(Path(result['csv_path']))}")


def main() -> None:
    st.set_page_config(
        page_title="Parking Pedestrian Warning",
        layout="wide",
    )
    st.title("Parking Pedestrian Warning Demo")
    st.write(
        "YOLO person/car detection, danger-zone ROI judgement, temporal stable-risk filtering, "
        "alert capture, CSV logging, video playback, OBS camera input, and ROI setup helpers."
    )

    with st.sidebar:
        st.header("Settings")
        input_mode = st.radio("Input Mode", ["Image", "Video", "OBS Camera"])
        if input_mode == "Image":
            uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
        elif input_mode == "Video":
            uploaded_file = st.file_uploader("Upload File", type=["mp4", "avi", "mov"])
        else:
            uploaded_file = None
            st.caption("OBS Camera mode reads frames from OpenCV camera_index.")

        initialize_app_session_state(
            st.session_state,
            realtime_prefix_factory=get_realtime_prefix,
            roi_preview_output_path=display_path(STREAMLIT_ROI_SELECTOR_PREVIEW_PATH),
            projector_preview_output_path=display_path(
                PROJECTOR_PREVIEW_DIR / "test_projectors_preview.jpg"
            ),
        )

        if input_mode == "OBS Camera":
            st.caption(f"Current realtime ROI prefix: {st.session_state['obs_roi_prefix']}")
            if st.button("New Realtime ROI Session"):
                st.session_state["obs_roi_prefix"] = get_realtime_prefix()
                st.session_state["active_roi_source_prefix"] = None
                increment_session_state_versions(st.session_state)
                st.rerun()

        roi_source_prefix = resolve_roi_source_prefix(
            input_mode,
            uploaded_name=uploaded_file.name if uploaded_file is not None else None,
            obs_prefix=st.session_state["obs_roi_prefix"],
            sample_image_path=SAMPLE_IMAGE_DIR / "test.jpg",
            sample_video_path=SAMPLE_VIDEO_DIR / "test.mp4",
        )

        if st.session_state["active_roi_source_prefix"] != roi_source_prefix:
            roi_artifact_paths = build_roi_artifact_paths(roi_source_prefix)
            projector_artifact_paths = build_projector_artifact_paths(roi_source_prefix)
            sync_roi_source_artifacts(
                st.session_state,
                source_prefix=roi_source_prefix,
                roi_artifact_paths=roi_artifact_paths,
                projector_artifact_paths=projector_artifact_paths,
            )

        st.subheader("Model Selection")
        if st.button("Refresh model list"):
            st.session_state["model_registry_refresh_token"] += 1
            get_model_options_cached.clear()
            st.rerun()

        model_options = get_model_options_cached(
            str(PROJECT_ROOT),
            int(st.session_state["model_registry_refresh_token"]),
        )
        option_keys = [option.key for option in model_options]
        default_model_key = get_default_model_key(model_options)
        previous_selected_model_key = st.session_state.get("selected_model_key", default_model_key)
        if previous_selected_model_key in option_keys:
            default_index = option_keys.index(previous_selected_model_key)
        elif default_model_key in option_keys:
            default_index = option_keys.index(default_model_key)
        else:
            default_index = 0

        selected_option = st.selectbox(
            "YOLO Model",
            model_options,
            index=default_index,
            format_func=lambda option: option.display_name,
            key="yolo_model_selectbox",
        )
        selected_model_key = selected_option.key
        selected_model_path = get_model_path_from_selection(selected_model_key, model_options)
        selected_option = get_model_option_from_selection(selected_model_key, model_options)
        st.session_state["selected_model_key"] = selected_model_key
        st.session_state["selected_model_path"] = selected_model_path
        requirement_summary = get_model_requirement_summary(selected_option)

        if requirement_summary["person_only"]:
            st.info(PERSON_ONLY_MODEL_NOTICE)
        if selected_option and selected_option.warning:
            st.warning(selected_option.warning)

        with st.expander("Advanced manual model path", expanded=False):
            use_manual_model_path = st.checkbox("Use manual model path", value=False)
            manual_model_path = st.text_input("Manual model path", value=DEFAULT_MODEL_NAME)
        model_name = resolve_selected_model_path(
            use_manual_model_path=bool(use_manual_model_path),
            manual_model_path=manual_model_path,
            selected_model_path=selected_model_path,
        )
        block_model_execution, block_model_message = should_block_model_execution(
            selected_option,
            use_manual_model_path=bool(use_manual_model_path),
        )
        if block_model_execution:
            st.error(block_model_message)
        if use_manual_model_path:
            st.caption(f"Manual model path in use: {model_name}")

        recommended_conf = get_recommended_confidence(selected_option)
        conf = st.slider(
            "Confidence",
            min_value=0.1,
            max_value=0.9,
            value=float(recommended_conf),
            step=0.05,
            key=f"confidence_slider_{selected_model_key}",
        )

        with st.expander("ROI Options", expanded=False):
            st.caption(f"ROI source prefix: {roi_source_prefix}")
            available_roi_json_files = list_available_roi_json_files()
            selected_existing_roi = None
            if available_roi_json_files:
                selected_existing_roi = st.selectbox(
                    "Existing ROI JSON",
                    available_roi_json_files,
                    key="existing_roi_json_selectbox",
                )
                if st.button("Use selected ROI JSON"):
                    st.session_state["selected_danger_zone_json_path"] = selected_existing_roi
                    st.session_state["selected_roi_output_path"] = selected_existing_roi
                    bump_roi_path_widgets()
                    st.rerun()
            else:
                st.caption("No per-source ROI JSON files found in data/roi_zones yet.")

            danger_zone_path_input = st.text_input(
                "Danger Zone JSON Path",
                value=st.session_state["selected_danger_zone_json_path"],
                key=f"danger_zone_path_widget_{st.session_state['danger_zone_path_widget_version']}",
            )
            st.session_state["selected_danger_zone_json_path"] = danger_zone_path_input.strip() or "data/danger_zone.json"
            zone_path_existed_before = Path(st.session_state["selected_danger_zone_json_path"]).exists()
            fallback_exists = Path("data/danger_zone.json").exists()
            zone_path = ensure_roi_json_exists_or_fallback(st.session_state["selected_danger_zone_json_path"])
            st.session_state["selected_danger_zone_json_path"] = zone_path
            if not zone_path_existed_before and Path(zone_path).exists() and fallback_exists:
                st.warning(f"ROI JSON did not exist yet. Copied fallback data/danger_zone.json to {zone_path}.")
            elif not Path(zone_path).exists():
                st.info("This ROI JSON does not exist yet. Capture a frame and run the ROI setup tool to create it.")

        with st.expander("ROI Setup", expanded=False):
            roi_base_image_path_input = st.text_input(
                "ROI base image path",
                value=st.session_state["selected_roi_base_image_path"],
                key=f"roi_base_image_path_widget_{st.session_state['roi_base_image_path_widget_version']}",
            )
            st.session_state["selected_roi_base_image_path"] = (
                roi_base_image_path_input.strip() or st.session_state["selected_roi_base_image_path"]
            )
            st.session_state["selected_roi_image_path"] = st.session_state["selected_roi_base_image_path"]
            roi_output_path_input = st.text_input(
                "ROI output path",
                value=st.session_state["selected_roi_output_path"],
                key=f"roi_output_path_widget_{st.session_state['roi_output_path_widget_version']}",
            )
            st.session_state["selected_roi_output_path"] = (
                roi_output_path_input.strip() or st.session_state["selected_danger_zone_json_path"]
            )
            roi_preview_path_input = st.text_input(
                "ROI preview output path",
                value=st.session_state["selected_roi_preview_output_path"],
                key=f"roi_preview_output_path_widget_{st.session_state['roi_preview_output_path_widget_version']}",
            )
            st.session_state["selected_roi_preview_output_path"] = (
                roi_preview_path_input.strip() or st.session_state["selected_roi_preview_output_path"]
            )
            roi_output_path = st.session_state["selected_roi_output_path"]
            roi_preview_path = st.session_state["selected_roi_preview_output_path"]
            st.caption(
                "ROI GUI opens only on a local Windows desktop. "
                "Remote/cloud Streamlit sessions may not show the OpenCV window."
            )

            if st.button("Run ROI Setup Tool"):
                st.info(
                    "ROI selector window is opening. Streamlit may pause while it is open. "
                    "Select the ROI, press S to save, then close the ROI window to refresh this page."
                )
                roi_image_path_for_selector = st.session_state["selected_roi_base_image_path"]
                command = build_roi_selector_command(
                    python_executable=sys.executable,
                    image_path=roi_image_path_for_selector,
                    output_path=roi_output_path,
                    preview_output_path=roi_preview_path,
                    load_existing=True,
                )
                launch_result = launch_roi_selector(command, wait=True)
                if launch_result["started"]:
                    st.caption(launch_result["command"])
                    if launch_result.get("returncode") == 0:
                        roi_display_data = refresh_roi_after_selector(
                            roi_json_path=roi_output_path,
                            roi_preview_path=roi_preview_path,
                            roi_base_image_path=roi_image_path_for_selector,
                        )
                        if roi_display_data["roi_exists"]:
                            st.success("ROI updated successfully.")
                        else:
                            st.warning("ROI selector finished, but ROI JSON was not found.")
                    else:
                        st.warning(f"ROI selector finished with exit code {launch_result.get('returncode')}.")
                        refresh_roi_after_selector(
                            roi_json_path=roi_output_path,
                            roi_preview_path=roi_preview_path,
                            roi_base_image_path=roi_image_path_for_selector,
                        )
                else:
                    st.error(f"Failed to start ROI selector: {launch_result['error']}")

            if st.button("Reload ROI Manually"):
                try:
                    roi_display_data = refresh_roi_after_selector(
                        roi_json_path=roi_output_path,
                        roi_preview_path=roi_preview_path,
                        roi_base_image_path=st.session_state["selected_roi_base_image_path"],
                    )
                    if roi_display_data["roi_exists"]:
                        st.success("ROI reloaded.")
                    else:
                        st.warning(roi_display_data["error"] or "ROI JSON is not available.")
                except Exception as exc:
                    st.error(f"Failed to load ROI: {exc}")

            roi_display_data = st.session_state["latest_roi_display_data"]
            if (
                roi_display_data is None
                or roi_display_data.get("roi_json_path") != roi_output_path
                or roi_display_data.get("roi_preview_path") != roi_preview_path
            ):
                roi_display_data = load_roi_display_data(
                    roi_json_path=roi_output_path,
                    roi_preview_path=roi_preview_path,
                    roi_base_image_path=st.session_state["selected_roi_base_image_path"],
            )
            st.session_state["latest_roi_display_data"] = roi_display_data
            display_roi_status(roi_display_data)

        with st.expander("Projector Device Setup", expanded=False):
            projector_json_path_input = st.text_input(
                "Projector devices JSON path",
                value=st.session_state["selected_projector_json_path"],
                key=f"projector_devices_json_path_widget_{st.session_state['projector_json_path_widget_version']}",
            )
            st.session_state["selected_projector_json_path"] = (
                projector_json_path_input.strip() or st.session_state["selected_projector_json_path"]
            )
            projector_preview_path_input = st.text_input(
                "Projector preview output path",
                value=st.session_state["selected_projector_preview_output_path"],
                key=f"projector_preview_output_path_widget_{st.session_state['projector_preview_output_path_widget_version']}",
            )
            st.session_state["selected_projector_preview_output_path"] = (
                projector_preview_path_input.strip() or st.session_state["selected_projector_preview_output_path"]
            )
            projector_json_path = st.session_state["selected_projector_json_path"]
            projector_preview_path = st.session_state["selected_projector_preview_output_path"]

            if st.button("Setup Projector Devices"):
                st.info(
                    "Projector selector window is opening. Click projector positions, press S to save, "
                    "then close the selector window to refresh this page."
                )
                command = build_projector_selector_command(
                    python_executable=sys.executable,
                    image_path=st.session_state["selected_roi_base_image_path"],
                    output_path=projector_json_path,
                    preview_output_path=projector_preview_path,
                    zone_path=zone_path,
                    load_existing=True,
                )
                launch_result = launch_roi_selector(command, wait=True)
                if launch_result["started"]:
                    st.caption(launch_result["command"])
                    if launch_result.get("returncode") == 0:
                        projector_display_data = refresh_projector_after_selector(
                            projector_json_path=projector_json_path,
                            projector_preview_path=projector_preview_path,
                        )
                        if projector_display_data["projector_exists"]:
                            st.success("Projector devices updated successfully.")
                        else:
                            st.warning("Projector selector finished, but projector JSON was not found.")
                    else:
                        st.warning(f"Projector selector finished with exit code {launch_result.get('returncode')}.")
                        refresh_projector_after_selector(
                            projector_json_path=projector_json_path,
                            projector_preview_path=projector_preview_path,
                        )
                else:
                    st.error(f"Failed to start projector selector: {launch_result['error']}")

            if st.button("Refresh Projector Devices"):
                refresh_projector_after_selector(
                    projector_json_path=projector_json_path,
                    projector_preview_path=projector_preview_path,
                )

            projector_display_data = st.session_state["latest_projector_display_data"]
            if (
                projector_display_data is None
                or projector_display_data.get("projector_json_path") != projector_json_path
                or projector_display_data.get("projector_preview_path") != projector_preview_path
            ):
                projector_display_data = load_projector_display_data(
                    projector_json_path=projector_json_path,
                    projector_preview_path=projector_preview_path,
                )
                st.session_state["latest_projector_display_data"] = projector_display_data
            display_projector_status(projector_display_data)

        camera_index = 0
        resize_width = 720
        max_live_frames = 0
        run_continuously = False
        save_output_video = False
        enable_realtime_alert_sound = False
        realtime_sound_trigger = "stable"
        realtime_beep_frequency = 1000
        realtime_beep_duration_ms = 250
        sound_cooldown_frames = 10
        if input_mode == "Video":
            with st.expander("Video Frame ROI Capture", expanded=False):
                st.caption(
                    "Capture a frame from the selected video and use it as the ROI setup base image. "
                    "Pick a frame where parked cars and the driveway area are clearly visible."
                )
                video_roi_frame_index = st.number_input(
                    "ROI capture frame index",
                    min_value=0,
                    max_value=100000,
                    value=0,
                    step=1,
                )
                video_path_for_roi = SAMPLE_VIDEO_DIR / "test.mp4"
                if uploaded_file is not None:
                    suffix = Path(uploaded_file.name).suffix or ".mp4"
                    video_path_for_roi = Path(
                        save_uploaded_file(uploaded_file, RESULT_DIR / f"streamlit_uploaded_video{suffix}")
                    )
                st.caption(f"ROI capture video path: {display_path(Path(video_path_for_roi))}")

                if st.button("Capture Frame From Video for ROI"):
                    try:
                        saved_snapshot = capture_video_frame_for_roi(
                            video_path=str(video_path_for_roi),
                            frame_index=int(video_roi_frame_index),
                            output_path=st.session_state["selected_roi_base_image_path"],
                            resize_width=None,
                        )
                        snapshot_display_path = display_path(Path(saved_snapshot))
                        st.session_state["last_video_roi_snapshot_path"] = snapshot_display_path
                        st.session_state["selected_roi_base_image_path"] = snapshot_display_path
                        st.session_state["selected_roi_image_path"] = snapshot_display_path
                        st.session_state["roi_base_image_path_widget_version"] += 1
                        st.success(f"Captured video ROI frame: {snapshot_display_path}")
                        st.info("Now run the ROI setup tool to mark a danger zone on this frame.")
                        st.rerun()
                    except Exception as exc:
                        st.error(
                            "Failed to capture frame from video. "
                            f"Check uploaded video, frame index, and video path. Error: {exc}"
                        )

                if st.session_state["last_video_roi_snapshot_path"]:
                    snapshot_path = Path(st.session_state["last_video_roi_snapshot_path"])
                    st.info(f"Recent video ROI capture: {st.session_state['last_video_roi_snapshot_path']}")
                    if snapshot_path.exists():
                        st.image(str(snapshot_path), caption="Recent Video ROI Snapshot", use_container_width=True)

        if input_mode == "OBS Camera":
            st.subheader("OBS Camera")
            camera_index = st.number_input("camera_index", min_value=0, max_value=20, value=0, step=1)
            st.caption("On Windows, OBS Virtual Camera may appear as camera index 0, 1, or 2.")
            run_continuously = st.checkbox("Run continuously until stopped", value=True)
            max_live_frames_input = st.number_input(
                "Max live frames",
                min_value=0,
                max_value=5000,
                value=0 if run_continuously else 100,
                step=1,
                disabled=run_continuously,
            )
            if run_continuously:
                max_live_frames = 0
                st.caption("Continuous mode passes --max-frames 0 to Step 10 and runs until Q or Stop.")
            else:
                max_live_frames = int(max_live_frames_input)
            resize_width = st.number_input("resize_width", min_value=160, max_value=1920, value=720, step=20)
            save_output_video = st.checkbox("save_output_video", value=False)

            with st.expander("Alert Sound", expanded=False):
                enable_realtime_alert_sound = st.checkbox("Enable realtime alert sound", value=True)
                realtime_sound_trigger_label = st.selectbox(
                    "Realtime alert sound trigger",
                    REALTIME_SOUND_TRIGGER_LABELS,
                    index=0,
                )
                realtime_sound_trigger = get_realtime_sound_trigger_value(realtime_sound_trigger_label)
                realtime_beep_frequency = st.number_input(
                    "Realtime beep frequency",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=50,
                )
                realtime_beep_duration_ms = st.number_input(
                    "Realtime beep duration ms",
                    min_value=50,
                    max_value=2000,
                    value=250,
                    step=50,
                )
                sound_cooldown_frames = st.number_input(
                    "Sound cooldown frames",
                    min_value=0,
                    max_value=1000,
                    value=10,
                    step=1,
                )

            if st.button("Capture Current OBS Frame"):
                try:
                    saved_snapshot = capture_camera_snapshot(
                        camera_index=int(camera_index),
                        output_path=st.session_state["selected_roi_base_image_path"],
                        resize_width=int(resize_width),
                    )
                    snapshot_display_path = display_path(Path(saved_snapshot))
                    st.session_state["last_obs_snapshot_path"] = snapshot_display_path
                    st.session_state["selected_roi_base_image_path"] = snapshot_display_path
                    st.session_state["selected_roi_image_path"] = snapshot_display_path
                    st.session_state["selected_danger_zone_json_path"] = zone_path
                    st.session_state["roi_base_image_path_widget_version"] += 1
                    st.success(f"Captured OBS frame: {display_path(Path(saved_snapshot))}")
                    st.info("Use the captured snapshot path as the ROI base image, then run the ROI setup tool.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to capture OBS frame: {exc}")

        if st.session_state["last_obs_snapshot_path"]:
            snapshot_path = Path(st.session_state["last_obs_snapshot_path"])
            st.info(f"Recent captured image: {st.session_state['last_obs_snapshot_path']}")
            if snapshot_path.exists():
                st.image(str(snapshot_path), caption="Recent OBS Snapshot", use_container_width=True)

        with st.expander("Frame Processing", expanded=False):
            if input_mode == "Video":
                max_frames = st.number_input("max_frames", min_value=1, max_value=1000, value=30, step=1)
                process_entire_video = st.checkbox("Process entire video", value=True)
                if process_entire_video:
                    st.warning("Entire video processing ignores max_frames and runs until the input video ends.")
            else:
                max_frames = 30
                process_entire_video = False
            frame_step = st.number_input("frame_step", min_value=1, max_value=100, value=1, step=1)
            window_size = st.number_input("window_size", min_value=1, max_value=100, value=5, step=1)
            min_risk_count = st.number_input("min_risk_count", min_value=1, max_value=100, value=3, step=1)
            cooldown_default = 30 if input_mode == "OBS Camera" else 10
            cooldown_frames = st.number_input(
                "cooldown_frames",
                min_value=0,
                max_value=1000,
                value=cooldown_default,
                step=1,
            )
            use_mock_person = st.checkbox("use_mock_person", value=False)

        enable_alert_sound = False
        alert_sound_trigger_mode = "logged"
        beep_frequency = 1000.0
        beep_duration_sec = 0.25
        beep_volume = 0.5
        if input_mode == "Video":
            with st.expander("Alert Sound", expanded=False):
                enable_alert_sound = st.checkbox("Enable alert sound in processed video", value=True)
                alert_sound_trigger_label = st.selectbox(
                    "Alert sound trigger",
                    VIDEO_ALERT_SOUND_TRIGGER_LABELS,
                    index=get_default_video_alert_sound_trigger_index(),
                )
                alert_sound_trigger_mode = get_video_alert_sound_trigger_value(alert_sound_trigger_label)
                beep_frequency = st.number_input(
                    "beep frequency Hz",
                    min_value=100.0,
                    max_value=5000.0,
                    value=1000.0,
                    step=50.0,
                )
                beep_duration_sec = st.number_input(
                    "beep duration sec",
                    min_value=0.05,
                    max_value=2.0,
                    value=0.25,
                    step=0.05,
                )
                beep_volume = st.slider("beep volume", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        if input_mode == "OBS Camera":
            st.subheader("Continuous Realtime Analysis")
            realtime_process = st.session_state["continuous_realtime_process"]
            realtime_running = realtime_process is not None and realtime_process.poll() is None
            st.session_state["continuous_realtime_running"] = bool(realtime_running)

            if realtime_running:
                st.success("Continuous realtime analysis is running.")
                st.caption("Use the OpenCV window and press Q, or click the stop button below.")
            else:
                st.info("Continuous realtime analysis is not running.")

            continuous_command = build_continuous_realtime_command(
                python_executable=sys.executable,
                camera_index=int(camera_index),
                zone_path=st.session_state["selected_danger_zone_json_path"],
                model_name=model_name,
                conf=float(conf),
                frame_step=int(frame_step),
                window_size=int(window_size),
                min_risk_count=int(min_risk_count),
                cooldown_frames=int(cooldown_frames),
                resize_width=int(resize_width),
                use_mock_person=bool(use_mock_person),
                save_output_video=bool(save_output_video),
                roi_source_image=st.session_state["selected_roi_base_image_path"],
                enable_alert_sound=bool(enable_realtime_alert_sound),
                sound_trigger=realtime_sound_trigger,
                sound_cooldown_frames=int(sound_cooldown_frames),
                beep_frequency=int(realtime_beep_frequency),
                beep_duration_ms=int(realtime_beep_duration_ms),
            )

            if st.button("Start Continuous OBS Analysis", disabled=bool(block_model_execution)):
                if realtime_running:
                    st.warning("Continuous realtime analysis is already running.")
                elif block_model_execution:
                    st.error(block_model_message)
                else:
                    try:
                        started_process = start_continuous_realtime_process(continuous_command)
                        st.session_state["continuous_realtime_process"] = started_process
                        st.session_state["continuous_realtime_command"] = continuous_command
                        st.session_state["continuous_realtime_running"] = True
                        st.success("Started continuous OBS realtime analysis.")
                    except RuntimeError as exc:
                        st.error(str(exc))

            if st.button("Stop Continuous OBS Analysis"):
                stop_result = stop_continuous_realtime_process(
                    st.session_state["continuous_realtime_process"]
                )
                st.session_state["continuous_realtime_process"] = None
                st.session_state["continuous_realtime_running"] = False
                if stop_result["stopped"]:
                    st.success(stop_result["message"])
                else:
                    st.warning(stop_result["message"])

            last_command = st.session_state["continuous_realtime_command"] or continuous_command
            st.caption("Last continuous realtime command:")
            st.code(" ".join(str(item) for item in last_command))

        run_label = "Run OBS Camera Analysis" if input_mode == "OBS Camera" else "Run Detection"
        run_button = st.button(run_label, type="primary", disabled=bool(block_model_execution))

    settings = {
        "model_name": model_name,
        "selected_model_key": selected_model_key,
        "selected_model_type": requirement_summary["model_type"],
        "selected_model_expected_classes": requirement_summary["expected_classes"],
        "block_model_execution": bool(block_model_execution),
        "block_model_message": block_model_message,
        "conf": float(conf),
        "zone_path": st.session_state["selected_danger_zone_json_path"],
        "projector_config_path": st.session_state["selected_projector_json_path"],
        "max_frames": int(max_frames),
        "process_entire_video": bool(process_entire_video),
        "frame_step": int(frame_step),
        "window_size": int(window_size),
        "min_risk_count": int(min_risk_count),
        "cooldown_frames": int(cooldown_frames),
        "use_mock_person": bool(use_mock_person),
        "enable_alert_sound": bool(enable_alert_sound),
        "alert_sound_trigger_mode": alert_sound_trigger_mode,
        "beep_frequency": float(beep_frequency),
        "beep_duration_sec": float(beep_duration_sec),
        "beep_volume": float(beep_volume),
        "camera_index": int(camera_index),
        "max_live_frames": int(max_live_frames),
        "run_continuously": bool(run_continuously),
        "resize_width": int(resize_width),
        "save_output_video": bool(save_output_video),
        "roi_source_image_path": st.session_state["selected_roi_base_image_path"],
        "enable_realtime_alert_sound": bool(enable_realtime_alert_sound),
        "realtime_sound_trigger": realtime_sound_trigger,
        "realtime_beep_frequency": int(realtime_beep_frequency),
        "realtime_beep_duration_ms": int(realtime_beep_duration_ms),
        "sound_cooldown_frames": int(sound_cooldown_frames),
    }

    if not run_button:
        st.info("Upload a file, use the default sample, or choose OBS Camera, then press the run button.")
        return
    if settings["block_model_execution"]:
        st.error(settings["block_model_message"])
        return

    try:
        if input_mode == "Image":
            render_image_mode(uploaded_file, settings)
        elif input_mode == "Video":
            render_video_mode(uploaded_file, settings)
        else:
            render_obs_camera_mode(settings)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    main()
