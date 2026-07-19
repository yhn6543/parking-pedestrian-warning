from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from src.app_paths import get_video_prefix


ROI_PATH_VERSION_KEYS = (
    "danger_zone_path_widget_version",
    "roi_base_image_path_widget_version",
    "roi_output_path_widget_version",
    "roi_preview_output_path_widget_version",
    "projector_json_path_widget_version",
    "projector_preview_output_path_widget_version",
)


def initialize_app_session_state(
    state: MutableMapping[str, Any],
    *,
    realtime_prefix_factory: Callable[[], str],
    roi_preview_output_path: str,
    projector_preview_output_path: str,
) -> None:
    for key in ROI_PATH_VERSION_KEYS:
        if key not in state:
            state[key] = 0

    if "obs_roi_prefix" not in state:
        state["obs_roi_prefix"] = realtime_prefix_factory()

    defaults = {
        "active_roi_source_prefix": None,
        "selected_danger_zone_json_path": "data/danger_zone.json",
        "selected_roi_base_image_path": "data/sample_images/test.jpg",
        "selected_roi_output_path": "data/danger_zone.json",
        "selected_roi_preview_output_path": roi_preview_output_path,
        "last_obs_snapshot_path": None,
        "last_video_roi_snapshot_path": None,
        "last_roi_json_path": None,
        "last_roi_preview_path": None,
        "latest_roi_display_data": None,
        "selected_projector_json_path": "data/projector_devices/test_projectors.json",
        "selected_projector_preview_output_path": projector_preview_output_path,
        "last_projector_json_path": None,
        "last_projector_preview_path": None,
        "latest_projector_display_data": None,
        "continuous_realtime_process": None,
        "continuous_realtime_command": None,
        "continuous_realtime_running": False,
        "model_registry_refresh_token": 0,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value

    if "selected_roi_image_path" not in state:
        state["selected_roi_image_path"] = state["selected_roi_base_image_path"]


def increment_session_state_versions(state: MutableMapping[str, Any]) -> None:
    for key in ROI_PATH_VERSION_KEYS:
        state[key] += 1


def resolve_roi_source_prefix(
    input_mode: str,
    *,
    uploaded_name: str | None,
    obs_prefix: str,
    sample_image_path: str | Path,
    sample_video_path: str | Path,
) -> str:
    if input_mode == "Video":
        source_name = uploaded_name if uploaded_name is not None else str(sample_video_path)
        return get_video_prefix(source_name)
    if input_mode == "Image":
        source_name = uploaded_name if uploaded_name is not None else str(sample_image_path)
        return get_video_prefix(source_name)
    return obs_prefix


def sync_roi_source_artifacts(
    state: MutableMapping[str, Any],
    *,
    source_prefix: str,
    roi_artifact_paths: dict[str, str],
    projector_artifact_paths: dict[str, str],
) -> bool:
    if state["active_roi_source_prefix"] == source_prefix:
        return False

    state["selected_danger_zone_json_path"] = roi_artifact_paths["danger_zone_json_path"]
    state["selected_roi_base_image_path"] = roi_artifact_paths["roi_base_image_path"]
    state["selected_roi_image_path"] = roi_artifact_paths["roi_base_image_path"]
    state["selected_roi_output_path"] = roi_artifact_paths["danger_zone_json_path"]
    state["selected_roi_preview_output_path"] = roi_artifact_paths["roi_preview_output_path"]
    state["selected_projector_json_path"] = projector_artifact_paths["projector_json_path"]
    state["selected_projector_preview_output_path"] = projector_artifact_paths[
        "projector_preview_output_path"
    ]
    state["latest_projector_display_data"] = None
    state["active_roi_source_prefix"] = source_prefix
    increment_session_state_versions(state)
    return True
