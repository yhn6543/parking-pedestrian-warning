import argparse
import math
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_logger import AlertLogger
from src.realtime_alert_sound import RealtimeAlertSound
from src.config import (
    ALERT_DIR,
    CONF_THRESHOLD,
    DEBUG_DIR,
    DEFAULT_MODEL_NAME,
    LOG_DIR,
    RESULT_DIR,
    ensure_directories,
)
from src.danger_zone import load_danger_zone
from src.detector import ObjectDetector
from src.input_loader import save_debug_image
from src.realtime_processor import (
    RealtimeStats,
    draw_realtime_overlay,
    get_image_size,
    open_realtime_source,
    resize_frame_keep_aspect,
    scale_polygon_to_frame,
    save_realtime_summary,
)
from src.risk_judgement import create_mock_person_detections, evaluate_risk
from src.temporal_filter import TemporalRiskFilter
from src.video_processor import create_video_writer
from src.visualizer import draw_risk_assessment, draw_temporal_status


DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "sample_videos" / "test.mp4"
DEFAULT_ZONE_PATH = PROJECT_ROOT / "data" / "danger_zone.json"
LAST_FRAME_PATH = DEBUG_DIR / "step10_realtime_last_frame.jpg"
SUMMARY_JSON_PATH = DEBUG_DIR / "step10_realtime_summary.json"
LOG_CSV_PATH = LOG_DIR / "step10_realtime_risk_log.csv"
DEFAULT_OUTPUT_VIDEO_PATH = RESULT_DIR / "step10_realtime_output.mp4"
DEFAULT_FPS = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 10 realtime webcam/video risk detection")
    parser.add_argument("--source", choices=["webcam", "video"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--zone", default=str(DEFAULT_ZONE_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-risk-count", type=int, default=3)
    parser.add_argument("--cooldown-frames", type=int, default=30)
    parser.add_argument("--resize-width", type=int, default=720)
    parser.add_argument("--roi-source-image", default=None)
    parser.add_argument("--roi-source-width", type=int, default=None)
    parser.add_argument("--roi-source-height", type=int, default=None)
    sound_group = parser.add_mutually_exclusive_group()
    sound_group.add_argument("--enable-alert-sound", dest="alert_sound_enabled", action="store_true")
    sound_group.add_argument("--disable-alert-sound", dest="alert_sound_enabled", action="store_false")
    parser.set_defaults(alert_sound_enabled=False)
    parser.add_argument("--sound-trigger", choices=["logged", "stable_start", "stable"], default="stable")
    parser.add_argument("--sound-cooldown-frames", type=int, default=10)
    parser.add_argument("--beep-frequency", type=int, default=1000)
    parser.add_argument("--beep-duration-ms", type=int, default=250)
    parser.add_argument("--use-mock-person", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-output-video", action="store_true")
    parser.add_argument("--output-video", default=str(DEFAULT_OUTPUT_VIDEO_PATH))
    return parser.parse_args()


def display_path(path: Path | str | None) -> str:
    if path is None:
        return "none"
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _safe_fps(value: float) -> float:
    fps = float(value or 0.0)
    if fps <= 0 or not math.isfinite(fps):
        return DEFAULT_FPS
    return fps


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be greater than 0.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be greater than or equal to 0.")
    if args.resize_width <= 0:
        raise ValueError("--resize-width must be greater than 0.")
    if (args.roi_source_width is None) != (args.roi_source_height is None):
        raise ValueError("--roi-source-width and --roi-source-height must be provided together.")
    if args.roi_source_width is not None and args.roi_source_width <= 0:
        raise ValueError("--roi-source-width must be greater than 0.")
    if args.roi_source_height is not None and args.roi_source_height <= 0:
        raise ValueError("--roi-source-height must be greater than 0.")
    if args.sound_cooldown_frames < 0:
        raise ValueError("--sound-cooldown-frames must be greater than or equal to 0.")
    if args.beep_frequency <= 0:
        raise ValueError("--beep-frequency must be greater than 0.")
    if args.beep_duration_ms <= 0:
        raise ValueError("--beep-duration-ms must be greater than 0.")


def resolve_roi_source_size(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.roi_source_image:
        try:
            return get_image_size(args.roi_source_image)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARNING] ROI source image could not be read. Falling back to source frame size. {exc}")
            return None
    if args.roi_source_width is not None and args.roi_source_height is not None:
        return int(args.roi_source_width), int(args.roi_source_height)
    return None


def print_header(args: argparse.Namespace, mode: str) -> None:
    print("[STEP] Step 10 - Realtime Webcam Risk Detection")
    print(f"[MODE] {mode}")
    print(f"[SOURCE] {args.source}")
    print()
    print("[INPUT]")
    if args.source == "webcam":
        print(f"* camera_index: {args.camera_index}")
    else:
        print(f"* video: {display_path(args.video)}")
    print(f"* zone: {display_path(args.zone)}")
    if mode == "YOLO":
        print(f"* model: {args.model}")
        print(f"* conf: {args.conf}")
    print(f"* frame_step: {args.frame_step}")
    print(f"* max_frames: {args.max_frames}")
    print(f"* window_size: {args.window_size}")
    print(f"* min_risk_count: {args.min_risk_count}")
    print(f"* cooldown_frames: {args.cooldown_frames}")
    print(f"* resize_width: {args.resize_width}")
    print(f"* roi_source_image: {display_path(args.roi_source_image)}")
    print(f"* roi_source_width: {args.roi_source_width}")
    print(f"* roi_source_height: {args.roi_source_height}")
    print(f"* alert_sound_enabled: {args.alert_sound_enabled}")
    print(f"* sound_trigger: {args.sound_trigger}")
    print(f"* sound_cooldown_frames: {args.sound_cooldown_frames}")
    print(f"* beep_frequency: {args.beep_frequency}")
    print(f"* beep_duration_ms: {args.beep_duration_ms}")
    print(f"* no_display: {args.no_display}")
    print(f"* save_output_video: {args.save_output_video}")
    print()


def print_success(summary: dict) -> None:
    print("[PROCESS SUMMARY]")
    print(f"* processed_frames: {summary['processed_frames']}")
    print(f"* raw_risk_frames: {summary['raw_risk_frames']}")
    print(f"* stable_risk_frames: {summary['stable_risk_frames']}")
    print(f"* logged_alert_count: {summary['logged_alert_count']}")
    print(f"* max_risk_person_count: {summary['max_risk_person_count']}")
    print(f"* alert_sound_count: {summary['alert_sound_count']}")
    print(f"* avg_fps: {summary['avg_fps']:.2f}")
    print()
    print("[OUTPUT]")
    print(f"* last frame: {summary['last_frame_path']}")
    print(f"* summary json: {display_path(SUMMARY_JSON_PATH)}")
    print(f"* csv log: {summary['log_csv_path']}")
    print(f"* captured alerts dir: {display_path(ALERT_DIR)}")
    if summary.get("output_video_path"):
        print(f"* output video: {summary['output_video_path']}")
    print()
    print("[STATUS] SUCCESS")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] video/zone/model 경로, 카메라 연결 상태, 출력 폴더 권한을 확인하세요.")


def draw_detection_skipped_label(frame):
    output = frame.copy()
    cv2.putText(
        output,
        "DETECTION SKIPPED",
        (16, max(28, output.shape[0] - 156)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_realtime_alert_logged_label(frame):
    output = frame.copy()
    cv2.putText(
        output,
        "REALTIME ALERT LOGGED",
        (16, max(28, output.shape[0] - 156)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_realtime_alert_sound_label(frame):
    output = frame.copy()
    cv2.putText(
        output,
        "ALERT SOUND",
        (16, max(28, output.shape[0] - 56)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def show_frame(frame, no_display: bool) -> bool:
    if no_display:
        return False
    cv2.imshow("Step 10 Realtime Risk Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord("q")


def main() -> int:
    args = parse_args()
    mode = "MOCK_PERSON" if args.use_mock_person else "YOLO"
    print_header(args, mode)

    capture = None
    writer = None
    display_enabled = not args.no_display

    try:
        validate_args(args)
        ensure_directories()

        if LOG_CSV_PATH.exists():
            LOG_CSV_PATH.unlink()

        polygon = load_danger_zone(args.zone)
        roi_source_size = resolve_roi_source_size(args)
        video_path = args.video if args.source == "video" else None
        capture = open_realtime_source(
            source=args.source,
            camera_index=args.camera_index,
            video_path=video_path,
        )
        source_fps = _safe_fps(capture.get(cv2.CAP_PROP_FPS))

        temporal_filter = TemporalRiskFilter(
            window_size=args.window_size,
            min_risk_count=args.min_risk_count,
        )
        alert_logger = AlertLogger(
            log_csv_path=str(LOG_CSV_PATH),
            alert_image_dir=str(ALERT_DIR),
            cooldown_frames=args.cooldown_frames,
            image_prefix="realtime_alert_frame",
        )
        alert_sound = RealtimeAlertSound(
            enabled=args.alert_sound_enabled,
            trigger_mode=args.sound_trigger,
            cooldown_frames=args.sound_cooldown_frames,
            frequency=args.beep_frequency,
            duration_ms=args.beep_duration_ms,
        )
        stats = RealtimeStats()

        detector = None
        if not args.use_mock_person:
            detector = ObjectDetector(model_name=args.model, conf_threshold=args.conf)

        def ensure_writer(frame) -> None:
            nonlocal writer
            if not args.save_output_video or writer is not None:
                return
            height, width = frame.shape[:2]
            writer = create_video_writer(str(args.output_video), width, height, source_fps)

        frame_index = 0
        last_visualized_frame = None
        should_stop = False

        while not should_stop:
            success, frame = capture.read()
            if not success or frame is None:
                break

            original_height, original_width = frame.shape[:2]
            resized = resize_frame_keep_aspect(frame, args.resize_width)
            analysis_height, analysis_width = resized.shape[:2]
            roi_source_width, roi_source_height = roi_source_size or (original_width, original_height)
            scaled_polygon = scale_polygon_to_frame(
                polygon,
                source_width=roi_source_width,
                source_height=roi_source_height,
                target_width=analysis_width,
                target_height=analysis_height,
            )

            if frame_index % args.frame_step != 0:
                skipped = draw_realtime_overlay(
                    resized,
                    frame_index=frame_index,
                    fps=stats.last_fps,
                    source_name=args.source,
                )
                skipped = draw_detection_skipped_label(skipped)
                ensure_writer(skipped)
                if writer is not None:
                    writer.write(skipped)
                last_visualized_frame = skipped
                should_stop = show_frame(skipped, args.no_display)
                frame_index += 1
                continue

            detections = (
                create_mock_person_detections()
                if args.use_mock_person
                else detector.detect(resized)
            )
            risk_result = evaluate_risk(detections, scaled_polygon)
            temporal_result = temporal_filter.update(risk_result["risk_detected"])

            timestamp_sec = frame_index / source_fps if source_fps else 0.0
            event = {
                "frame_index": int(frame_index),
                "timestamp_sec": round(float(timestamp_sec), 4),
                "raw_risk": bool(temporal_result["raw_risk"]),
                "stable_risk": bool(temporal_result["stable_risk"]),
                "risk_level": risk_result["risk_level"],
                "person_count": int(risk_result["person_count"]),
                "car_count": int(risk_result["car_count"]),
                "risk_person_count": int(risk_result["risk_person_count"]),
                "risk_count": int(temporal_result["risk_count"]),
                "window_size": int(temporal_result["window_size"]),
                "min_risk_count": int(temporal_result["min_risk_count"]),
            }

            will_log = alert_logger.should_log(
                frame_index=frame_index,
                stable_risk=event["stable_risk"],
            )
            stats.update(risk_result, temporal_result, logged=will_log)
            fps = stats.calculate_fps()

            visualized = draw_risk_assessment(resized, risk_result, scaled_polygon)
            visualized = draw_temporal_status(visualized, temporal_result)
            visualized = draw_realtime_overlay(
                visualized,
                frame_index=frame_index,
                fps=fps,
                source_name=args.source,
            )
            if will_log:
                visualized = draw_realtime_alert_logged_label(visualized)

            log_result = alert_logger.log_if_needed(visualized, event)
            if will_log and not log_result["logged"]:
                stats.logged_alert_count = max(0, stats.logged_alert_count - 1)
            sound_result = alert_sound.update_and_play(
                frame_index=frame_index,
                stable_risk=event["stable_risk"],
                logged=bool(log_result["logged"]),
            )
            if sound_result["sound_played"]:
                visualized = draw_realtime_alert_sound_label(visualized)

            ensure_writer(visualized)
            if writer is not None:
                writer.write(visualized)

            last_visualized_frame = visualized
            should_stop = show_frame(visualized, args.no_display)

            frame_index += 1
            if args.max_frames > 0 and stats.processed_frames >= args.max_frames:
                break

        if stats.processed_frames == 0 or last_visualized_frame is None:
            raise ValueError("No frames were processed from the realtime source.")

        save_debug_image(last_visualized_frame, str(LAST_FRAME_PATH))
        avg_fps = stats.calculate_fps()

        summary = {
            "mode": mode,
            "source": args.source,
            "video_path": display_path(args.video) if args.source == "video" else None,
            "camera_index": int(args.camera_index),
            "zone_path": display_path(args.zone),
            "processed_frames": int(stats.processed_frames),
            "raw_risk_frames": int(stats.raw_risk_frames),
            "stable_risk_frames": int(stats.stable_risk_frames),
            "logged_alert_count": int(stats.logged_alert_count),
            "max_risk_person_count": int(stats.max_risk_person_count),
            "window_size": int(args.window_size),
            "min_risk_count": int(args.min_risk_count),
            "cooldown_frames": int(args.cooldown_frames),
            "resize_width": int(args.resize_width),
            "roi_source_image": display_path(args.roi_source_image),
            "roi_source_width": int(roi_source_size[0]) if roi_source_size else None,
            "roi_source_height": int(roi_source_size[1]) if roi_source_size else None,
            "alert_sound_enabled": bool(args.alert_sound_enabled),
            "alert_sound_trigger": str(args.sound_trigger),
            "alert_sound_count": int(alert_sound.sound_count),
            "avg_fps": float(avg_fps),
            "log_csv_path": display_path(LOG_CSV_PATH),
            "last_frame_path": display_path(LAST_FRAME_PATH),
        }
        if args.save_output_video:
            summary["output_video_path"] = display_path(args.output_video)

        save_realtime_summary(summary, str(SUMMARY_JSON_PATH))
        print_success(summary)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
