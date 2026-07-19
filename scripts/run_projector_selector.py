import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEBUG_DIR, SAMPLE_IMAGE_DIR, ensure_directories
from src.danger_zone import load_danger_zone
from src.input_loader import get_image_info, load_image, save_debug_image
from src.projector_devices import load_projector_config, save_projector_config
from src.roi_selector import (
    display_point_to_original_point,
    original_points_to_display_points,
    resize_image_for_display,
)


DEFAULT_IMAGE_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "projector_devices" / "test_projectors.json"
DEFAULT_PREVIEW_OUTPUT_PATH = DEBUG_DIR / "projector_previews" / "test_projectors_preview.jpg"
WINDOW_NAME = "Projector Device Selector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mouse click projector device selector")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-output", default=str(DEFAULT_PREVIEW_OUTPUT_PATH))
    parser.add_argument("--zone", default="")
    parser.add_argument("--load-existing", action="store_true")
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=800)
    return parser.parse_args()


def display_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def print_header(args: argparse.Namespace) -> None:
    print("[STEP] Projector Device Selector")
    print()
    print("[INPUT]")
    print(f"* image: {display_path(args.image)}")
    print(f"* output: {display_path(args.output)}")
    print(f"* preview_output: {display_path(args.preview_output)}")
    print(f"* zone: {display_path(args.zone) if args.zone else ''}")
    print(f"* max_display_width: {args.max_display_width}")
    print(f"* max_display_height: {args.max_display_height}")
    print()
    print("[CONTROL]")
    print("* Left click: add projector device")
    print("* U: undo last projector")
    print("* R: reset projectors")
    print("* S: save projectors")
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


def make_device(point: list[int], index: int) -> dict:
    number = index + 1
    return {
        "id": f"projector_{number}",
        "name": f"Projector {number}",
        "x": int(point[0]),
        "y": int(point[1]),
        "enabled": True,
        "endpoint": "",
    }


def renumber_devices(devices: list[dict]) -> list[dict]:
    renumbered = []
    for index, device in enumerate(devices):
        item = dict(device)
        item["id"] = f"projector_{index + 1}"
        item["name"] = item.get("name") or f"Projector {index + 1}"
        renumbered.append(item)
    return renumbered


def load_initial_devices(args: argparse.Namespace) -> list[dict]:
    output_path = Path(args.output)
    if not args.load_existing or not output_path.exists():
        return []
    return load_projector_config(output_path)["devices"]


def load_optional_zone(zone_path: str) -> list[list[int]]:
    if not zone_path:
        return []
    path = Path(zone_path)
    if not path.exists():
        return []
    return load_danger_zone(str(path))


def draw_projector_preview(
    image,
    devices: list[dict],
    zone_points: list[list[int]] | None = None,
    title: str = "Projector Devices",
):
    output = image.copy()
    height, width = output.shape[:2]

    if zone_points and len(zone_points) >= 3:
        import numpy as np

        overlay = output.copy()
        contour = np.array(zone_points, dtype=np.int32)
        cv2.fillPoly(overlay, [contour], (0, 180, 255))
        cv2.addWeighted(overlay, 0.18, output, 0.82, 0, output)
        cv2.polylines(output, [contour], isClosed=True, color=(0, 90, 255), thickness=2)

    for index, device in enumerate(devices, start=1):
        x = int(device["x"])
        y = int(device["y"])
        label = str(device.get("id") or f"projector_{index}")
        cv2.circle(output, (x, y), 8, (255, 180, 0), thickness=-1)
        cv2.circle(output, (x, y), 12, (255, 255, 255), thickness=2)

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_x = min(max(0, x + 14), max(0, width - label_size[0] - 8))
        label_y = min(max(label_size[1] + baseline + 2, y - 8), height - 4)
        cv2.rectangle(
            output,
            (label_x - 4, label_y - label_size[1] - baseline - 4),
            (label_x + label_size[0] + 4, label_y + baseline),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (label_x, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    panel_lines = [
        "Left click: add projector",
        "U: undo   R: reset   S: save",
        "Q/ESC: quit",
    ]
    panel_width = min(max(width - 16, 1), 560)
    panel_height = 34 + len(panel_lines) * 18
    cv2.rectangle(output, (8, 8), (8 + panel_width, 8 + panel_height), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"{title} | devices: {len(devices)}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for line_index, line in enumerate(panel_lines):
        cv2.putText(
            output,
            line,
            (18, 52 + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def save_outputs(image, devices: list[dict], image_size: dict, output_path: str, preview_path: str, zone_points) -> tuple[str, str]:
    devices = renumber_devices(devices)
    config_path = save_projector_config(output_path, devices, image_size)
    preview = draw_projector_preview(image, devices, zone_points=zone_points)
    saved_preview_path = save_debug_image(preview, preview_path)
    return config_path, saved_preview_path


def print_success(devices: list[dict], output_path: str, preview_path: str) -> None:
    print("[PROJECTORS SAVED]")
    print(f"* device_count: {len(devices)}")
    print()
    print("[OUTPUT]")
    print(f"* projector json: {display_path(output_path)}")
    print(f"* preview image: {display_path(preview_path)}")
    print()
    print("[STATUS] SUCCESS")


def print_cancelled(devices: list[dict]) -> None:
    print("[PROJECTORS]")
    print(f"* device_count: {len(devices)}")
    print()
    print("[STATUS] CANCELLED")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] Check image path, output path, and display size options.")


def run_gui_mode(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    image_info = get_image_info(image)
    display_image, display_info = resize_image_for_display(
        image,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
    )
    print_display_info(display_info)

    devices_original = load_initial_devices(args)
    zone_points_original = load_optional_zone(args.zone)
    zone_points_display = original_points_to_display_points(zone_points_original, display_info) if zone_points_original else []
    saved = False

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point_original = display_point_to_original_point((int(x), int(y)), display_info)
            devices_original.append(make_device(point_original, len(devices_original)))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_info["display_width"], display_info["display_height"])
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            devices_display = []
            for device in renumber_devices(devices_original):
                point_display = original_points_to_display_points([[device["x"], device["y"]]], display_info)[0]
                item = dict(device)
                item["x"] = point_display[0]
                item["y"] = point_display[1]
                devices_display.append(item)

            preview = draw_projector_preview(
                display_image,
                devices_display,
                zone_points=zone_points_display,
                title="Projector Device Selector",
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if devices_original:
                    devices_original.pop()
            elif key == ord("r"):
                devices_original.clear()
            elif key == ord("s"):
                config_path, preview_path = save_outputs(
                    image,
                    devices_original,
                    {"width": image_info["width"], "height": image_info["height"]},
                    args.output,
                    args.preview_output,
                    zone_points_original,
                )
                print_success(devices_original, config_path, preview_path)
                saved = True
                break
    finally:
        cv2.destroyAllWindows()

    if not saved:
        print_cancelled(devices_original)
    return 0


def main() -> int:
    args = parse_args()
    print_header(args)

    try:
        ensure_directories()
        return run_gui_mode(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
