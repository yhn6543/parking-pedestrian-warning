from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_public_package.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_public_package", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_package_dry_run_has_only_existing_safe_files(tmp_path: Path) -> None:
    module = load_module()
    destination = tmp_path / "public"

    result = module.build_package(ROOT, destination, apply=False)

    assert result["status"] == "dry_run_passed"
    assert result["file_count"] > 40
    assert not destination.exists()
    assert all(not Path(path).is_absolute() for path in result["files"])
    assert not any(path.startswith(("data/", "reports/", "runs/", "models/")) for path in result["files"])


def test_public_package_apply_copies_allowlist_without_git_or_weights(tmp_path: Path) -> None:
    module = load_module()
    destination = tmp_path / "public"

    result = module.build_package(ROOT, destination, apply=True)

    copied = [path for path in destination.rglob("*") if path.is_file()]
    assert len(copied) == result["file_count"]
    assert (destination / "app.py").is_file()
    assert (destination / "src" / "detector.py").is_file()
    assert not (destination / ".git").exists()
    assert not any(path.suffix.lower() in module.BLOCKED_SUFFIXES for path in copied)


def test_public_package_refuses_nonempty_destination(tmp_path: Path) -> None:
    module = load_module()
    destination = tmp_path / "public"
    destination.mkdir()
    (destination / "keep.txt").write_text("user file", encoding="utf-8")

    with pytest.raises(ValueError, match="absent or empty"):
        module.build_package(ROOT, destination, apply=True)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "user file"
