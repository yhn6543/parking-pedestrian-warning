import json

import numpy as np
import pytest

from src.danger_zone import load_danger_zone
from src.roi_selector import (
    calculate_display_size,
    create_default_roi_for_image,
    display_point_to_original_point,
    draw_roi_preview,
    load_roi_json,
    original_points_to_display_points,
    resize_image_for_display,
    save_roi_json,
    validate_roi_points,
)


def test_validate_roi_points_accepts_three_or_more_points() -> None:
    points = [[0, 0], [100, 0], [100, 100]]

    assert validate_roi_points(points) is True


def test_validate_roi_points_rejects_two_points() -> None:
    with pytest.raises(ValueError):
        validate_roi_points([[0, 0], [100, 100]])


def test_validate_roi_points_rejects_invalid_point_format() -> None:
    with pytest.raises(ValueError):
        validate_roi_points([[0, 0], [100, 0], [100]])


def test_save_roi_json_writes_danger_zone_key(tmp_path) -> None:
    output_path = tmp_path / "nested" / "danger_zone.json"
    points = [[0, 0], [100, 0], [100, 100]]

    saved_path = save_roi_json(points, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_path == str(output_path)
    assert data == {"danger_zone": points}


def test_load_roi_json_reads_saved_roi(tmp_path) -> None:
    output_path = tmp_path / "danger_zone.json"
    points = [[0, 0], [100, 0], [100, 100]]
    save_roi_json(points, str(output_path))

    loaded_points = load_roi_json(str(output_path))

    assert loaded_points == points
    assert load_danger_zone(str(output_path)) == points


def test_draw_roi_preview_returns_same_shape_image() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    points = [[10, 70], [120, 70], [130, 110], [5, 110]]

    result = draw_roi_preview(image, points, closed=True)

    assert result.shape == image.shape
    assert not np.shares_memory(result, image)
    assert result.sum() > 0


def test_calculate_display_size_scales_wide_image_to_limit() -> None:
    info = calculate_display_size(width=1920, height=1080, max_display_width=960, max_display_height=720)

    assert info["display_width"] == 960
    assert info["display_height"] == 540
    assert info["scale_x"] == pytest.approx(2.0)
    assert info["scale_y"] == pytest.approx(2.0)


def test_calculate_display_size_scales_tall_image_to_limit() -> None:
    info = calculate_display_size(width=1000, height=2000, max_display_width=960, max_display_height=720)

    assert info["display_height"] == 720
    assert info["display_width"] == 360
    assert info["scale_x"] == pytest.approx(1000 / 360)
    assert info["scale_y"] == pytest.approx(2000 / 720)


def test_calculate_display_size_keeps_small_image_original_size() -> None:
    info = calculate_display_size(width=640, height=480, max_display_width=960, max_display_height=720)

    assert info["display_width"] == 640
    assert info["display_height"] == 480
    assert info["scale_x"] == pytest.approx(1.0)
    assert info["scale_y"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("width", "height", "max_width", "max_height"),
    [
        (0, 480, 960, 720),
        (640, 0, 960, 720),
        (640, 480, 0, 720),
        (640, 480, 960, 0),
    ],
)
def test_calculate_display_size_rejects_invalid_sizes(width, height, max_width, max_height) -> None:
    with pytest.raises(ValueError):
        calculate_display_size(
            width=width,
            height=height,
            max_display_width=max_width,
            max_display_height=max_height,
        )


def test_resize_image_for_display_returns_scaled_copy() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    display_image, display_info = resize_image_for_display(
        image,
        max_display_width=960,
        max_display_height=720,
    )

    assert display_image.shape == (540, 960, 3)
    assert display_info["scale_x"] == pytest.approx(2.0)
    assert display_info["scale_y"] == pytest.approx(2.0)
    assert not np.shares_memory(display_image, image)


def test_display_point_to_original_point_uses_scale() -> None:
    display_info = {
        "original_width": 1920,
        "original_height": 1080,
        "display_width": 960,
        "display_height": 540,
        "scale_x": 2.0,
        "scale_y": 2.0,
    }

    assert display_point_to_original_point((480, 270), display_info) == [960, 540]


def test_display_point_to_original_point_clamps_negative_and_large_values() -> None:
    display_info = {
        "original_width": 100,
        "original_height": 80,
        "display_width": 50,
        "display_height": 40,
        "scale_x": 2.0,
        "scale_y": 2.0,
    }

    assert display_point_to_original_point((-5, -1), display_info) == [0, 0]
    assert display_point_to_original_point((999, 999), display_info) == [99, 79]


def test_original_points_to_display_points_uses_scale() -> None:
    display_info = {
        "original_width": 1920,
        "original_height": 1080,
        "display_width": 960,
        "display_height": 540,
        "scale_x": 2.0,
        "scale_y": 2.0,
    }

    assert original_points_to_display_points([[960, 540]], display_info) == [[480, 270]]


def test_create_default_roi_for_image_returns_expected_sample_points() -> None:
    points = create_default_roi_for_image(width=640, height=480)

    assert points == [[96, 264], [544, 264], [576, 456], [64, 456]]


def test_create_default_roi_for_image_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError):
        create_default_roi_for_image(width=0, height=480)

    with pytest.raises(ValueError):
        create_default_roi_for_image(width=640, height=0)
