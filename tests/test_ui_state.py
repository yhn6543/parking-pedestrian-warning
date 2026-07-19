from src.ui_state import (
    ROI_PATH_VERSION_KEYS,
    increment_session_state_versions,
    initialize_app_session_state,
    resolve_roi_source_prefix,
    sync_roi_source_artifacts,
)


def initialize(state, factory=lambda: "2026-07-19_12-00-00") -> None:
    initialize_app_session_state(
        state,
        realtime_prefix_factory=factory,
        roi_preview_output_path="outputs/debug/streamlit_roi_selector_preview.jpg",
        projector_preview_output_path="outputs/debug/projectors/test.jpg",
    )


def test_initialize_app_session_state_populates_missing_defaults() -> None:
    state = {}

    initialize(state)

    assert all(state[key] == 0 for key in ROI_PATH_VERSION_KEYS)
    assert state["obs_roi_prefix"] == "2026-07-19_12-00-00"
    assert state["selected_danger_zone_json_path"] == "data/danger_zone.json"
    assert state["selected_roi_image_path"] == "data/sample_images/test.jpg"
    assert state["continuous_realtime_running"] is False


def test_initialize_app_session_state_preserves_existing_values() -> None:
    calls = []
    state = {
        "obs_roi_prefix": "existing-prefix",
        "selected_roi_base_image_path": "custom/base.jpg",
        "selected_roi_image_path": "custom/image.jpg",
        "model_registry_refresh_token": 7,
    }

    initialize(state, factory=lambda: calls.append(True) or "new-prefix")

    assert calls == []
    assert state["obs_roi_prefix"] == "existing-prefix"
    assert state["selected_roi_base_image_path"] == "custom/base.jpg"
    assert state["selected_roi_image_path"] == "custom/image.jpg"
    assert state["model_registry_refresh_token"] == 7


def test_initialize_derives_roi_image_path_from_existing_base_path() -> None:
    state = {"selected_roi_base_image_path": "custom/base.jpg"}

    initialize(state)

    assert state["selected_roi_image_path"] == "custom/base.jpg"


def test_increment_session_state_versions_updates_every_widget_version() -> None:
    state = {key: index for index, key in enumerate(ROI_PATH_VERSION_KEYS)}

    increment_session_state_versions(state)

    assert [state[key] for key in ROI_PATH_VERSION_KEYS] == list(range(1, len(ROI_PATH_VERSION_KEYS) + 1))


def test_resolve_roi_source_prefix_matches_each_input_mode() -> None:
    common = {
        "uploaded_name": None,
        "obs_prefix": "obs-session",
        "sample_image_path": "data/sample_images/test.jpg",
        "sample_video_path": "data/sample_videos/test.mp4",
    }

    assert resolve_roi_source_prefix("Image", **common) == "test"
    assert resolve_roi_source_prefix("Video", **common) == "test"
    assert resolve_roi_source_prefix("OBS Camera", **common) == "obs-session"
    assert resolve_roi_source_prefix("Video", **{**common, "uploaded_name": "parking lot.mp4"}) == "parking_lot"


def test_sync_roi_source_artifacts_updates_once_and_preserves_same_prefix() -> None:
    state = {}
    initialize(state)
    roi_paths = {
        "danger_zone_json_path": "data/roi_zones/camera_danger_zone.json",
        "roi_base_image_path": "outputs/debug/roi_base_images/camera.jpg",
        "roi_preview_output_path": "outputs/debug/roi_previews/camera.jpg",
    }
    projector_paths = {
        "projector_json_path": "data/projector_devices/camera.json",
        "projector_preview_output_path": "outputs/debug/projectors/camera.jpg",
    }

    changed = sync_roi_source_artifacts(
        state,
        source_prefix="camera",
        roi_artifact_paths=roi_paths,
        projector_artifact_paths=projector_paths,
    )

    assert changed is True
    assert state["active_roi_source_prefix"] == "camera"
    assert state["selected_danger_zone_json_path"] == roi_paths["danger_zone_json_path"]
    assert state["selected_projector_json_path"] == projector_paths["projector_json_path"]
    assert all(state[key] == 1 for key in ROI_PATH_VERSION_KEYS)

    snapshot = dict(state)
    changed_again = sync_roi_source_artifacts(
        state,
        source_prefix="camera",
        roi_artifact_paths={**roi_paths, "danger_zone_json_path": "unexpected.json"},
        projector_artifact_paths=projector_paths,
    )

    assert changed_again is False
    assert state == snapshot
