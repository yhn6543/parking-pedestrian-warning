import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEBUG_DIR, PROJECT_ROOT, SAMPLE_IMAGE_DIR, ensure_directories
from src.danger_zone import (
    draw_danger_zone,
    draw_test_points,
    is_point_in_polygon,
    load_danger_zone,
)
from src.input_loader import load_image, save_debug_image


DEFAULT_IMAGE_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_ZONE_PATH = PROJECT_ROOT / "data" / "danger_zone.json"
DEFAULT_RESULT_IMAGE_PATH = DEBUG_DIR / "step03_danger_zone.jpg"
DEFAULT_RESULT_JSON_PATH = DEBUG_DIR / "step03_roi_test_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 03 danger zone ROI test")
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
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_test_points(image_shape: tuple[int, ...], polygon: list[list[int]]) -> list[list[int]]:
    height, width = image_shape[:2]
    return [
        [width // 2, height // 2],
        [10, 10],
        [max(width - 10, 0), max(height - 10, 0)],
        polygon[0],
        [width // 2, int(height * 0.75)],
    ]


def save_roi_test_json(polygon: list[list[int]], point_results: list[dict], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "danger_zone": polygon,
        "point_tests": point_results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def print_success(
    image_path: Path,
    zone_path: Path,
    polygon: list[list[int]],
    point_results: list[dict],
    result_image_path: Path,
    result_json_path: Path,
) -> None:
    print("[STEP] Step 03 - Danger Zone ROI Test")
    print("[INPUT]")
    print(f"* image: {display_path(image_path)}")
    print(f"* zone: {display_path(zone_path)}")
    print()
    print("[DANGER ZONE]")
    print(f"* points: {polygon}")
    print()
    print("[POINT TEST]")
    for item in point_results:
        print(f"* point: {item['point']}, inside: {item['inside']}")
    print()
    print("[OUTPUT]")
    print(f"* result image: {display_path(result_image_path)}")
    print(f"* result json: {display_path(result_json_path)}")
    print()
    print("[STATUS] SUCCESS")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] 이미지 경로, danger_zone.json 경로, polygon 좌표 형식을 확인하세요.")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    zone_path = Path(args.zone)

    try:
        ensure_directories()
        image = load_image(str(image_path))
        polygon = load_danger_zone(str(zone_path))

        points = build_test_points(image.shape, polygon)
        point_results = [
            {
                "point": point,
                "inside": is_point_in_polygon((point[0], point[1]), polygon),
            }
            for point in points
        ]

        visualized = draw_danger_zone(image, polygon)
        visualized = draw_test_points(visualized, point_results)
        result_image_path = Path(save_debug_image(visualized, str(DEFAULT_RESULT_IMAGE_PATH)))
        result_json_path = Path(save_roi_test_json(polygon, point_results, DEFAULT_RESULT_JSON_PATH))

        print_success(image_path, zone_path, polygon, point_results, result_image_path, result_json_path)
        return 0
    except (FileNotFoundError, ValueError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
