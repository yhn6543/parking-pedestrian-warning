from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKED_TEXT = (
    "C:" + "\\Users\\" + "sky" + "2311",
    "C:" + "\\" + "yolov26s_data",
    "C:" + "\\" + "yolo_runs",
)
BLOCKED_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".tflite",
    ".ckpt",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".wav",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
}

ROOT_FILES = (
    ".gitignore",
    "README.md",
    "LICENSE_PENDING.md",
    "config.example.yaml",
    "requirements.txt",
    "requirements-dev.txt",
    "app.py",
)
PUBLIC_SCRIPTS = (
    "scripts/build_public_package.py",
    "scripts/build_manual_exclusion_review.py",
    "scripts/manual_exclusion_bbox_server.py",
    "scripts/manual_exclusion_review_assets/bbox_review.html",
    "scripts/manual_exclusion_review_assets/bbox_review_app.css",
    "scripts/manual_exclusion_review_assets/bbox_review_app.js",
    "scripts/manual_exclusion_review_assets/bbox_review_shortcuts.js",
    "scripts/run_projector_selector.py",
    "scripts/run_step01_input_test.py",
    "scripts/run_step02_detection_test.py",
    "scripts/run_step03_roi_test.py",
    "scripts/run_step04_risk_test.py",
    "scripts/run_step05_video_test.py",
    "scripts/run_step06_temporal_test.py",
    "scripts/run_step07_alert_logging_test.py",
    "scripts/run_step09_report_test.py",
    "scripts/run_step10_realtime_test.py",
    "scripts/run_step11_roi_selector.py",
)
PUBLIC_TESTS = (
    "tests/test_alert_audio.py",
    "tests/test_alert_logger.py",
    "tests/test_app_config.py",
    "tests/test_app_model_selection.py",
    "tests/test_build_public_package.py",
    "tests/test_danger_zone.py",
    "tests/test_detector.py",
    "tests/test_detector_person_only.py",
    "tests/test_input_loader.py",
    "tests/test_manual_exclusion_review.py",
    "tests/test_model_registry.py",
    "tests/test_projector_devices.py",
    "tests/test_realtime_alert_sound.py",
    "tests/test_realtime_processor.py",
    "tests/test_report_generator.py",
    "tests/test_risk_judgement.py",
    "tests/test_risk_judgement_person_only.py",
    "tests/test_roi_path_management.py",
    "tests/test_roi_selector.py",
    "tests/test_roi_ui_helpers.py",
    "tests/test_step10_realtime_cli.py",
    "tests/test_streamlit_obs_helpers.py",
    "tests/test_streamlit_video_roi_helpers.py",
    "tests/test_temporal_filter.py",
    "tests/test_ui_state.py",
    "tests/test_video_display_helpers.py",
    "tests/test_video_processor.py",
    "tests/test_visualizer_overlay.py",
)


def public_files(source_root: Path) -> list[Path]:
    source_root = source_root.resolve()
    source_files = sorted(
        path.relative_to(source_root)
        for path in (source_root / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    paths = [Path(path) for path in (*ROOT_FILES, *PUBLIC_SCRIPTS, *PUBLIC_TESTS)]
    return sorted(set(paths + source_files), key=lambda path: path.as_posix())


def validate_sources(source_root: Path, paths: list[Path]) -> list[str]:
    errors = []
    for relative_path in paths:
        source_path = source_root / relative_path
        if not source_path.is_file():
            errors.append(f"missing allowlist file: {relative_path.as_posix()}")
            continue
        if source_path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"blocked extension: {relative_path.as_posix()}")
            continue
        try:
            text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-text allowlist file: {relative_path.as_posix()}")
            continue
        for blocked in BLOCKED_TEXT:
            if blocked.casefold() in text.casefold():
                errors.append(f"blocked path in {relative_path.as_posix()}: {blocked}")
    return errors


def ensure_destination_is_safe(destination: Path) -> None:
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"destination must be absent or empty: {destination}")


def build_package(source_root: Path, destination: Path, *, apply: bool) -> dict[str, object]:
    source_root = source_root.resolve()
    destination = destination.resolve()
    paths = public_files(source_root)
    errors = validate_sources(source_root, paths)
    if errors:
        raise ValueError("; ".join(errors))
    ensure_destination_is_safe(destination)

    total_bytes = sum((source_root / path).stat().st_size for path in paths)
    if apply:
        destination.mkdir(parents=True, exist_ok=True)
        for relative_path in paths:
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / relative_path, target)
    return {
        "status": "applied" if apply else "dry_run_passed",
        "source": str(source_root),
        "destination": str(destination),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "files": [path.as_posix() for path in paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the allowlist-only public package.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_package(args.source, args.destination, apply=args.apply)
    except ValueError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
