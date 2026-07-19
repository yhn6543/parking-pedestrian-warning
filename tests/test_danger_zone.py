import json

import numpy as np
import pytest

from src.danger_zone import (
    draw_danger_zone,
    is_point_in_polygon,
    load_danger_zone,
    validate_polygon,
)


def test_validate_polygon_returns_true_for_valid_polygon() -> None:
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]

    assert validate_polygon(polygon)


def test_is_point_in_polygon_returns_true_for_inside_point() -> None:
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]

    assert is_point_in_polygon((50, 50), polygon)


def test_is_point_in_polygon_returns_false_for_outside_point() -> None:
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]

    assert not is_point_in_polygon((150, 50), polygon)


def test_is_point_in_polygon_treats_boundary_as_inside() -> None:
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]

    assert is_point_in_polygon((0, 0), polygon)


def test_validate_polygon_raises_value_error_for_two_points() -> None:
    with pytest.raises(ValueError):
        validate_polygon([[0, 0], [100, 100]])


def test_load_danger_zone_raises_value_error_when_key_is_missing(tmp_path) -> None:
    json_path = tmp_path / "danger_zone.json"
    json_path.write_text(json.dumps({"points": [[0, 0], [1, 0], [1, 1]]}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_danger_zone(str(json_path))


def test_draw_danger_zone_returns_image_with_same_shape() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    polygon = [[10, 10], [100, 10], [120, 80], [20, 90]]

    result = draw_danger_zone(image, polygon)

    assert result.shape == image.shape
    assert not np.shares_memory(result, image)
