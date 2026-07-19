import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEBUG_DIR, SAMPLE_IMAGE_DIR, ensure_directories
from src.input_loader import get_image_info, load_image, save_debug_image
from src.roi_selector import (
    create_default_roi_for_image,
    display_point_to_original_point,
    draw_roi_preview,
    load_roi_json,
    original_points_to_display_points,
    resize_image_for_display,
    save_roi_json,
    validate_roi_points,
)


DEFAULT_IMAGE_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "danger_zone.json"
DEFAULT_PREVIEW_OUTPUT_PATH = DEBUG_DIR / "step11_roi_selector_preview.jpg"
WINDOW_NAME = "Step 11 - ROI Selector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 11 mouse click ROI selector")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-output", default=str(DEFAULT_PREVIEW_OUTPUT_PATH))
    parser.add_argument("--use-default", action="store_true")
    parser.add_argument("--load-existing", action="store_true")
    parser.add_argument("--max-display-width", type=int, default=960)
    parser.add_argument("--max-display-height", type=int, default=720)
    return parser.parse_args()


def display_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def print_header(args: argparse.Namespace, mode: str) -> None:
    print("[STEP] Step 11 - Mouse Click ROI Selector")
    print(f"[MODE] {mode}")
    print()
    print("[INPUT]")
    print(f"* image: {display_path(args.image)}")
    print(f"* output: {display_path(args.output)}")
    print(f"* preview_output: {display_path(args.preview_output)}")
    print(f"* max_display_width: {args.max_display_width}")
    print(f"* max_display_height: {args.max_display_height}")
    print()
    if mode == "GUI":
        print("[CONTROL]")
        print("* Left click: add point")
        print("* U: undo last point")
        print("* R: reset points")
        print("* S: save ROI")
        print("* Q/ESC: quit without saving")
        print()


def print_display_info(display_info: dict) -> None:
    print("[DISPLAY]")
    print(f"* original_width: {display_info['original_width']}")
    print(f"* original_height: {display_info['original_height']}")
    print(f"* display_width: {display_info['display_width']}")
    print(f"* display_height: {display_info['display_height']}")
    print(f"* scale_x: {display_info['scale_x']:.4f}")
    print(f"* scale_y: {display_info['scale_y']:.4f}")
    print()


def print_success(points_original: list[list[int]], output_path: str, preview_path: str) -> None:
    print("[ROI SAVED]")
    print(f"* original points: {points_original}")
    print()
    print("[OUTPUT]")
    print(f"* roi json: {display_path(output_path)}")
    print(f"* preview image: {display_path(preview_path)}")
    print()
    print("[STATUS] SUCCESS")


def print_cancelled(points_original: list[list[int]]) -> None:
    print("[ROI]")
    print(f"* original points: {points_original}")
    print()
    print("[STATUS] CANCELLED")
    print("[SUGGESTION] Select at least three ROI points and press S to save.")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] Check image path, output path, ROI point count, and display size options.")


def save_outputs(image, points_original: list[list[int]], output_path: str, preview_path: str) -> tuple[str, str]:
    validate_roi_points(points_original)
    roi_json_path = save_roi_json(points_original, output_path)
    preview = draw_roi_preview(image, points_original, closed=True)
    saved_preview_path = save_debug_image(preview, preview_path)
    return roi_json_path, saved_preview_path


def run_default_mode(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    image_info = get_image_info(image)
    _, display_info = resize_image_for_display(
        image,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
    )
    print_display_info(display_info)
    points_original = create_default_roi_for_image(
        width=image_info["width"],
        height=image_info["height"],
    )
    roi_json_path, preview_path = save_outputs(
        image,
        points_original,
        args.output,
        args.preview_output,
    )
    print_success(points_original, roi_json_path, preview_path)
    return 0


def load_initial_points(args: argparse.Namespace) -> list[list[int]]:
    output_path = Path(args.output)
    if not args.load_existing or not output_path.exists():
        return []
    return load_roi_json(str(output_path))


def run_gui_mode(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    display_image, display_info = resize_image_for_display(
        image,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
    )
    print_display_info(display_info)
    points_original = load_initial_points(args)
    saved = False

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point_original = display_point_to_original_point((int(x), int(y)), display_info)
            points_original.append(point_original)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_info["display_width"], display_info["display_height"])
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            points_display = original_points_to_display_points(points_original, display_info)
            preview = draw_roi_preview(
                display_image,
                points_display,
                closed=len(points_display) >= 3,
                display_info=display_info,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if points_original:
                    points_original.pop()
            elif key == ord("r"):
                points_original.clear()
            elif key == ord("s"):
                try:
                    roi_json_path, preview_path = save_outputs(
                        image,
                        points_original,
                        args.output,
                        args.preview_output,
                    )
                except ValueError as exc:
                    print(f"[ERROR] {exc}")
                    print("[SUGGESTION] Select at least three ROI points, then press S again.")
                    continue
                print_success(points_original, roi_json_path, preview_path)
                saved = True
                break
    finally:
        cv2.destroyAllWindows()

    if not saved:
        print_cancelled(points_original)
    return 0


def main() -> int:
    args = parse_args()
    mode = "DEFAULT_NO_GUI" if args.use_default else "GUI"
    print_header(args, mode)

    try:
        ensure_directories()
        if args.use_default:
            return run_default_mode(args)
        return run_gui_mode(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
