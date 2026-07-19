import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONF_THRESHOLD, DEBUG_DIR, DEFAULT_MODEL_NAME, SAMPLE_IMAGE_DIR, ensure_directories
from src.danger_zone import load_danger_zone
from src.detector import ObjectDetector
from src.input_loader import load_image, save_debug_image
from src.risk_judgement import create_mock_person_detections, evaluate_risk
from src.visualizer import draw_risk_assessment


DEFAULT_IMAGE_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_ZONE_PATH = PROJECT_ROOT / "data" / "danger_zone.json"
YOLO_RESULT_IMAGE_PATH = DEBUG_DIR / "step04_risk_judgement.jpg"
YOLO_RESULT_JSON_PATH = DEBUG_DIR / "step04_risk_judgement.json"
MOCK_RESULT_IMAGE_PATH = DEBUG_DIR / "step04_risk_judgement_mock.jpg"
MOCK_RESULT_JSON_PATH = DEBUG_DIR / "step04_risk_judgement_mock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 04 single-image risk judgement test")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Input image path. Default: data/sample_images/test.jpg",
    )
    parser.add_argument(
        "--zone",
        default=str(DEFAULT_ZONE_PATH),
        help="Danger zone JSON path. Default: data/danger_zone.json",
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
    parser.add_argument(
        "--use-mock-person",
        action="store_true",
        help="Use mock person detections instead of running YOLO.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def save_risk_result_json(risk_result: dict, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(risk_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def print_header(args: argparse.Namespace, mode: str) -> None:
    print("[STEP] Step 04 - Risk Judgement Test")
    print(f"[MODE] {mode}")
    print()
    print("[INPUT]")
    print(f"* image: {display_path(Path(args.image))}")
    print(f"* zone: {display_path(Path(args.zone))}")
    if mode == "YOLO":
        print(f"* model: {args.model}")
        print(f"* conf: {args.conf}")
    print()


def print_success(risk_result: dict, result_image_path: Path, result_json_path: Path) -> None:
    print("[RISK SUMMARY]")
    print(f"* risk_detected: {risk_result['risk_detected']}")
    print(f"* risk_level: {risk_result['risk_level']}")
    print(f"* person_count: {risk_result['person_count']}")
    print(f"* car_count: {risk_result['car_count']}")
    print(f"* risk_person_count: {risk_result['risk_person_count']}")
    print()
    print("[OUTPUT]")
    print(f"* result image: {display_path(result_image_path)}")
    print(f"* result json: {display_path(result_json_path)}")
    print()
    print("[STATUS] SUCCESS")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] 이미지 경로, danger_zone.json, YOLO 모델 파일, detection 형식을 확인하세요.")


def main() -> int:
    args = parse_args()
    mode = "MOCK_PERSON" if args.use_mock_person else "YOLO"
    print_header(args, mode)

    try:
        ensure_directories()
        image = load_image(args.image)
        polygon = load_danger_zone(args.zone)

        if args.use_mock_person:
            detections = create_mock_person_detections()
            result_image_path = MOCK_RESULT_IMAGE_PATH
            result_json_path = MOCK_RESULT_JSON_PATH
        else:
            detector = ObjectDetector(model_name=args.model, conf_threshold=args.conf)
            detections = detector.detect(image)
            result_image_path = YOLO_RESULT_IMAGE_PATH
            result_json_path = YOLO_RESULT_JSON_PATH

        risk_result = evaluate_risk(detections, polygon)
        visualized = draw_risk_assessment(image, risk_result, polygon)
        saved_image_path = Path(save_debug_image(visualized, str(result_image_path)))
        saved_json_path = Path(save_risk_result_json(risk_result, result_json_path))

        print_success(risk_result, saved_image_path, saved_json_path)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
