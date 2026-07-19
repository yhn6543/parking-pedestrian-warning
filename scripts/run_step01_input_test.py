import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEBUG_DIR, SAMPLE_IMAGE_DIR, ensure_directories
from src.input_loader import get_image_info, load_image, save_debug_image


DEFAULT_IMAGE_PATH = SAMPLE_IMAGE_DIR / "test.jpg"
DEFAULT_OUTPUT_PATH = DEBUG_DIR / "step01_input_preview.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 01 input file test")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Input image path. Default: data/sample_images/test.jpg",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def print_success(image_path: Path, output_path: Path, info: dict) -> None:
    print("[STEP] Step 01 - Input File Test")
    print(f"[INPUT] {display_path(image_path)}")
    print()
    print("[IMAGE INFO]")
    print(f"- width: {info['width']}")
    print(f"- height: {info['height']}")
    print(f"- channels: {info['channels']}")
    print(f"- dtype: {info['dtype']}")
    print()
    print("[OUTPUT]")
    print(f"- debug image: {display_path(output_path)}")
    print()
    print("[STATUS] SUCCESS")


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] data/sample_images 폴더에 test.jpg 파일이 있는지 확인하세요.")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_path = DEFAULT_OUTPUT_PATH

    try:
        ensure_directories()
        image = load_image(str(image_path))
        info = get_image_info(image)
        saved_path = Path(save_debug_image(image, str(output_path)))
        print_success(image_path, saved_path, info)
        return 0
    except (FileNotFoundError, ValueError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
