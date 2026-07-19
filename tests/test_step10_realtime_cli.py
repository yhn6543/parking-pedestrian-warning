from argparse import Namespace

from scripts.run_step10_realtime_test import resolve_roi_source_size


def test_resolve_roi_source_size_falls_back_when_image_is_missing(tmp_path) -> None:
    args = Namespace(
        roi_source_image=str(tmp_path / "missing.jpg"),
        roi_source_width=None,
        roi_source_height=None,
    )

    assert resolve_roi_source_size(args) is None


def test_resolve_roi_source_size_uses_explicit_dimensions() -> None:
    args = Namespace(
        roi_source_image=None,
        roi_source_width=1920,
        roi_source_height=1080,
    )

    assert resolve_roi_source_size(args) == (1920, 1080)
