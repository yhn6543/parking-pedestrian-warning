from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_manual_exclusion_review.py"
SERVER_SCRIPT = ROOT / "scripts" / "manual_exclusion_bbox_server.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_candidate_csv(path: Path, image_path: Path, label_path: Path) -> None:
    columns = [
        "normalized_id",
        "image_path",
        "label_path",
        "split",
        "source_dataset",
        "output_group",
        "candidate_reasons",
        "person_count",
        "small_bbox_ratio",
        "false_negative_count",
        "overlay_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "normalized_id": "sample",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "split": "val",
                "source_dataset": "fixture",
                "output_group": "fixture",
                "candidate_reasons": "small_person",
                "person_count": "1",
                "small_bbox_ratio": "1",
                "false_negative_count": "1",
                "overlay_path": str(image_path),
            }
        )


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    candidate_path = tmp_path / "candidates.csv"
    cv2.imwrite(str(image_path), np.zeros((100, 120, 3), dtype=np.uint8))
    label_path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    write_candidate_csv(candidate_path, image_path, label_path)
    return candidate_path, image_path, label_path


def test_decision_aliases_are_safe_and_v_is_rejected() -> None:
    builder = load_module(BUILD_SCRIPT, "manual_review_builder_aliases")
    server = load_module(SERVER_SCRIPT, "manual_review_server_aliases")

    expected = ["KEEP", "KEEP", "KEEP", "DROP", "DROP", "HOLD", "HOLD"]
    values = ["", "KEEP", "K", "DROP", "F", "HOLD", "C"]
    assert [builder.normalize_decision(value) for value in values] == expected
    assert [server.normalize_decision(value) for value in values] == expected
    with pytest.raises(ValueError):
        builder.normalize_decision("V")
    with pytest.raises(ValueError):
        server.normalize_decision("V")


def test_generated_html_loads_bbox_app_then_shortcuts(tmp_path: Path) -> None:
    builder = load_module(BUILD_SCRIPT, "manual_review_builder_assets")
    candidate_path, _, _ = make_fixture(tmp_path)
    output = tmp_path / "review"

    builder.write_assets(output, candidate_path)

    html = (output / "bbox_review.html").read_text(encoding="utf-8")
    assert 'src="bbox_review_app.js"' in html
    assert 'src="bbox_review_shortcuts.js"' in html
    assert html.index('src="bbox_review_app.js"') < html.index('src="bbox_review_shortcuts.js"')
    assert "KEEP (K)" in html
    assert "BBox 표시 중 (V)" in html
    assert "KEEP (V)" not in html
    assert "bbox_review.html" in (output / "index.html").read_text(encoding="utf-8")
    assert "bbox_review.html" in (output / "manual_exclusion_review.html").read_text(
        encoding="utf-8"
    )


def test_shortcut_contract_and_text_input_guard_are_present() -> None:
    asset_dir = ROOT / "scripts" / "manual_exclusion_review_assets"
    app_source = (asset_dir / "bbox_review_app.js").read_text(encoding="utf-8")
    shortcuts = (asset_dir / "bbox_review_shortcuts.js").read_text(encoding="utf-8")

    for key in ("'v'", "'a'", "'w'", "'s'", "'arrowleft'", "'arrowright'", "'arrowup'", "'arrowdown'"):
        assert key in app_source
    for key in ("'k'", "'f'", "'c'", "'q'", "'e'"):
        assert key in shortcuts
    assert "event.code === 'Space'" in shortcuts
    assert "guardedDecision('DROP', true)" in shortcuts
    assert "['INPUT', 'TEXTAREA', 'SELECT']" in shortcuts
    assert "saveBeforeLeaving" in shortcuts
    assert "beforeunload" in shortcuts
    assert "guardedDecision('KEEP')" in shortcuts
    assert "guardedDecision('DROP')" in shortcuts
    assert "guardedDecision('HOLD')" in shortcuts


def test_generated_launcher_opens_latest_page_without_personal_project_path(tmp_path: Path) -> None:
    builder = load_module(BUILD_SCRIPT, "manual_review_builder_launcher")
    candidate_path, _, _ = make_fixture(tmp_path)
    output = tmp_path / "review"

    builder.write_assets(output, candidate_path)

    launcher = (output / "start_manual_exclusion_review.bat").read_text(encoding="utf-8")
    assert "bbox_review.html" in launcher
    assert '"%~dp0manual_exclusion_bbox_server.py"' in launcher
    assert "manual_exclusion_review.html" not in launcher
    assert r"Documents\parking_pedestrian_warning" not in launcher


def test_generator_preserves_existing_decisions_and_bbox_database(tmp_path: Path) -> None:
    builder = load_module(BUILD_SCRIPT, "manual_review_builder_preserve")
    candidate_path, _, _ = make_fixture(tmp_path)
    output = tmp_path / "review"
    output.mkdir()
    decisions = output / "manual_exclusion_decisions.csv"
    database = output / "manual_exclusion_bbox_edits.sqlite"
    decisions.write_bytes(b"existing decisions")
    database.write_bytes(b"existing database")

    builder.write_assets(output, candidate_path)

    assert decisions.read_bytes() == b"existing decisions"
    assert database.read_bytes() == b"existing database"


def test_bbox_store_edits_only_separate_database_and_keeps_source_label(tmp_path: Path) -> None:
    server = load_module(SERVER_SCRIPT, "manual_review_server_store")
    candidate_path, _, label_path = make_fixture(tmp_path)
    review_root = tmp_path / "review"
    review_root.mkdir()
    original_label = label_path.read_bytes()
    store = server.ReviewStore(review_root, candidate_path)
    row = store.candidate("sample")
    assert row is not None

    initial = store.load_boxes(row)
    saved = store.save_boxes(
        row,
        {
            "revision": initial["revision"],
            "boxes": [{"class_id": 0, "x1": 20, "y1": 20, "x2": 50, "y2": 70}],
        },
    )

    assert saved["revision"] == 1
    assert (review_root / "manual_exclusion_bbox_edits.sqlite").is_file()
    assert label_path.read_bytes() == original_label
    with pytest.raises(ValueError, match="stale"):
        store.save_boxes(row, {"revision": 0, "boxes": saved["boxes"]})
    with pytest.raises(ValueError, match="empty"):
        store.save_boxes(row, {"revision": 1, "boxes": []})


def test_k_f_c_decisions_are_saved_without_v_alias(tmp_path: Path) -> None:
    server = load_module(SERVER_SCRIPT, "manual_review_server_decisions")
    candidate_path, _, _ = make_fixture(tmp_path)
    review_root = tmp_path / "review"
    review_root.mkdir()
    store = server.ReviewStore(review_root, candidate_path)
    row = store.candidate("sample")
    assert row is not None

    assert store.save_decision(row, {"decision": "K"})["decision"] == "KEEP"
    assert store.save_decision(row, {"decision": "F"})["decision"] == "DROP"
    assert store.save_decision(row, {"decision": "C"})["decision"] == "HOLD"
    with pytest.raises(ValueError):
        store.save_decision(row, {"decision": "V"})
