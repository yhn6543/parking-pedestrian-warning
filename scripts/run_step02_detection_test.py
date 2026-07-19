import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONF_THRESHOLD, DEBUG_DIR, DEFAULT_MODEL_NAME, SAMPLE_IMAGE_DIR, ensure_directories
from src.detector import ObjectDetector
from src.input_loader import load_image, save_debug_image
from src.visualizer import draw_detections, save_detections_json


DEFAULT_IMAGE_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_RESULT_IMAGE_PATH = DEBUG_DIR / "step02_detection_result.jpg"
DEFAULT_RESULT_JSON_PATH = DEBUG_DIR / "step02_detection_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 02 YOLO person/car detection test")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Input image path. Default: data/sample_images/test.jpg",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"YOLO model name or path. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold. Default: {CONF_THRESHOLD}",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def print_header(image_path: Path, model_name: str, conf_threshold: float) -> None:
    print("[STEP] Step 02 - YOLO Person/Car Detection")
    print(f"[INPUT] {display_path(image_path)}")
    print(f"[MODEL] {model_name}")
    print(f"[CONF_THRESHOLD] {conf_threshold}")
    print()


def print_success(detections: list[dict], result_image_path: Path, result_json_path: Path) -> None:
    counts = Counter(detection["class_name"] for detection in detections)

    print("[DETECTION SUMMARY]")
    print(f"* person: {counts.get('person', 0)}")
    print(f"* car: {counts.get('car', 0)}")
    print(f"* total: {len(detections)}")
    print()
    print("[OUTPUT]")
    print(f"* result image: {display_path(result_image_path)}")
    print(f"* result json: {display_path(result_json_path)}")
    print()
    print("[STATUS] SUCCESS")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] ultralytics 설치 여부, 모델 다운로드 가능 여부, 이미지 경로를 확인하세요.")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    result_image_path = DEFAULT_RESULT_IMAGE_PATH
    result_json_path = DEFAULT_RESULT_JSON_PATH

    print_header(image_path, args.model, args.conf)

    try:
        ensure_directories()
        image = load_image(str(image_path))
        detector = ObjectDetector(model_name=args.model, conf_threshold=args.conf)
        detections = detector.detect(image)
        visualized_image = draw_detections(image, detections)
        saved_image_path = Path(save_debug_image(visualized_image, str(result_image_path)))
        saved_json_path = Path(save_detections_json(detections, str(result_json_path)))
        print_success(detections, saved_image_path, saved_json_path)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
