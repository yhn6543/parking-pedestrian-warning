import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONF_THRESHOLD,
    DEBUG_DIR,
    DEFAULT_MODEL_NAME,
    RESULT_DIR,
    SAMPLE_IMAGE_DIR,
    SAMPLE_VIDEO_DIR,
    ensure_directories,
)
from src.danger_zone import load_danger_zone
from src.detector import ObjectDetector
from src.input_loader import save_debug_image
from src.risk_judgement import create_mock_person_detections, evaluate_risk
from src.temporal_filter import TemporalRiskFilter
from src.video_processor import (
    create_dummy_video_from_image,
    create_video_writer,
    get_video_info,
    save_video_summary,
    summarize_frame_result,
)
from src.visualizer import draw_risk_assessment, draw_temporal_status


DEFAULT_VIDEO_PATH = SAMPLE_VIDEO_DIR / "test.mp4"
DEFAULT_IMAGE_FOR_DUMMY_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_ZONE_PATH = PROJECT_ROOT / "data" / "danger_zone.json"
RESULT_VIDEO_PATH = RESULT_DIR / "step06_temporal_processed_video.mp4"
PREVIEW_FRAME_PATH = DEBUG_DIR / "step06_temporal_preview_frame.jpg"
SUMMARY_JSON_PATH = DEBUG_DIR / "step06_temporal_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 06 temporal risk filtering test")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--image-for-dummy", default=str(DEFAULT_IMAGE_FOR_DUMMY_PATH))
    parser.add_argument("--zone", default=str(DEFAULT_ZONE_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--use-mock-person", action="store_true")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-risk-count", type=int, default=3)
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than 0.")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be greater than 0.")


def print_header(args: argparse.Namespace, mode: str) -> None:
    print("[STEP] Step 06 - Temporal Risk Filtering")
    print(f"[MODE] {mode}")
    print()
    print("[INPUT]")
    print(f"* video: {display_path(Path(args.video))}")
    print(f"* zone: {display_path(Path(args.zone))}")
    if mode == "YOLO":
        print(f"* model: {args.model}")
        print(f"* conf: {args.conf}")
    print(f"* max_frames: {args.max_frames}")
    print(f"* frame_step: {args.frame_step}")
    print(f"* window_size: {args.window_size}")
    print(f"* min_risk_count: {args.min_risk_count}")
    print()


def print_video_info(video_info: dict) -> None:
    print("[VIDEO INFO]")
    print(f"* width: {video_info['width']}")
    print(f"* height: {video_info['height']}")
    print(f"* fps: {video_info['fps']}")
    print(f"* frame_count: {video_info['frame_count']}")
    print()


def print_success(summary: dict) -> None:
    print("[PROCESS SUMMARY]")
    print(f"* processed_frames: {summary['processed_frames']}")
    print(f"* raw_risk_frames: {summary['raw_risk_frames']}")
    print(f"* stable_risk_frames: {summary['stable_risk_frames']}")
    print(f"* max_risk_person_count: {summary['max_risk_person_count']}")
    print()
    print("[OUTPUT]")
    print(f"* result video: {display_path(RESULT_VIDEO_PATH)}")
    print(f"* preview frame: {display_path(PREVIEW_FRAME_PATH)}")
    print(f"* summary json: {display_path(SUMMARY_JSON_PATH)}")
    print()
    print("[STATUS] SUCCESS")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] 영상 경로, danger_zone.json, YOLO 모델 파일, temporal filter 설정을 확인하세요.")


def main() -> int:
    args = parse_args()
    mode = "MOCK_PERSON" if args.use_mock_person else "YOLO"
    print_header(args, mode)

    try:
        validate_args(args)
        ensure_directories()

        video_path = Path(args.video)
        if not video_path.exists():
            create_dummy_video_from_image(args.image_for_dummy, str(video_path), frame_count=30, fps=10.0)

        polygon = load_danger_zone(args.zone)
        video_info = get_video_info(str(video_path))
        print_video_info(video_info)

        temporal_filter = TemporalRiskFilter(
            window_size=args.window_size,
            min_risk_count=args.min_risk_count,
        )

        detector = None
        if not args.use_mock_person:
            detector = ObjectDetector(model_name=args.model, conf_threshold=args.conf)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        writer = create_video_writer(
            str(RESULT_VIDEO_PATH),
            video_info["width"],
            video_info["height"],
            video_info["fps"],
        )

        frame_results = []
        processed_frames = 0
        raw_risk_frames = 0
        stable_risk_frames = 0
        max_risk_person_count = 0
        frame_index = 0
        preview_saved = False

        try:
            while processed_frames < args.max_frames:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                if frame_index % args.frame_step != 0:
                    frame_index += 1
                    continue

                if args.use_mock_person:
                    detections = create_mock_person_detections()
                else:
                    detections = detector.detect(frame)

                risk_result = evaluate_risk(detections, polygon)
                temporal_result = temporal_filter.update(risk_result["risk_detected"])

                visualized = draw_risk_assessment(frame, risk_result, polygon)
                visualized = draw_temporal_status(visualized, temporal_result)
                writer.write(visualized)

                if not preview_saved:
                    save_debug_image(visualized, str(PREVIEW_FRAME_PATH))
                    preview_saved = True

                risk_summary = summarize_frame_result(frame_index, risk_result)
                frame_summary = {
                    "frame_index": int(frame_index),
                    "raw_risk": bool(temporal_result["raw_risk"]),
                    "stable_risk": bool(temporal_result["stable_risk"]),
                    "risk_count": int(temporal_result["risk_count"]),
                    "window_size": int(temporal_result["window_size"]),
                    "min_risk_count": int(temporal_result["min_risk_count"]),
                    "risk_detected": bool(risk_summary["risk_detected"]),
                    "risk_level": risk_summary["risk_level"],
                    "person_count": int(risk_summary["person_count"]),
                    "car_count": int(risk_summary["car_count"]),
                    "risk_person_count": int(risk_summary["risk_person_count"]),
                }
                frame_results.append(frame_summary)

                if frame_summary["raw_risk"]:
                    raw_risk_frames += 1
                if frame_summary["stable_risk"]:
                    stable_risk_frames += 1
                max_risk_person_count = max(
                    max_risk_person_count,
                    frame_summary["risk_person_count"],
                )

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
            "zone_path": display_path(Path(args.zone)),
            "processed_frames": int(processed_frames),
            "raw_risk_frames": int(raw_risk_frames),
            "stable_risk_frames": int(stable_risk_frames),
            "max_risk_person_count": int(max_risk_person_count),
            "window_size": int(args.window_size),
            "min_risk_count": int(args.min_risk_count),
            "frame_results": frame_results,
        }
        save_video_summary(summary, str(SUMMARY_JSON_PATH))
        print_success(summary)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
