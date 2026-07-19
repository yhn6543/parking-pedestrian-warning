from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as an OpenCV BGR array."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image file: {path}")

    return image


def save_debug_image(image: np.ndarray, output_path: str) -> str:
    """Save an image to disk and return the saved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Failed to save debug image: {path}")

    return str(path)


def get_image_info(image: np.ndarray) -> dict:
    """Return basic image metadata for terminal debug output."""
    if image is None or not hasattr(image, "shape"):
        raise ValueError("Invalid image: expected a numpy.ndarray with shape information.")

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(image.dtype),
    }


def load_video_first_frame(video_path: str) -> np.ndarray:
    """Load the first frame of a video file as an OpenCV BGR array."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        success, frame = capture.read()
    finally:
        capture.release()

    if not success or frame is None:
        raise ValueError(f"Failed to read first frame from video: {path}")

    return frame
