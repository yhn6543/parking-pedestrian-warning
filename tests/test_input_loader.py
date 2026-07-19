import numpy as np
import pytest

from src.input_loader import get_image_info, load_image


def test_get_image_info_returns_expected_dict() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    info = get_image_info(image)

    assert info == {
        "width": 32,
        "height": 24,
        "channels": 3,
        "dtype": "uint8",
    }


def test_load_image_raises_file_not_found_for_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        load_image("data/sample_images/does_not_exist.jpg")
