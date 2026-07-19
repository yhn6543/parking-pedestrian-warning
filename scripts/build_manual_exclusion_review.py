from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).with_name("manual_exclusion_review_assets")
SERVER_SOURCE = Path(__file__).with_name("manual_exclusion_bbox_server.py")
DEFAULT_CANDIDATES = Path(
    os.environ.get(
        "PARKING_RECHECK_CANDIDATES",
        "reports/v3_dataset_recheck_candidates/recheck_candidates.csv",
    )
)
DEFAULT_OUT = Path(
    os.environ.get("PARKING_MANUAL_REVIEW_DIR", "reports/v3_manual_exclusion_review")
)
DECISION_COLUMNS = [
    "normalized_id",
    "image_path",
    "label_path",
    "split",
    "source_dataset",
    "output_group",
    "candidate_reasons",
    "decision",
    "decision_label",
    "note",
    "updated_at",
]
STATIC_ASSETS = (
    "bbox_review.html",
    "bbox_review_app.css",
    "bbox_review_app.js",
    "bbox_review_shortcuts.js",
)


def normalize_decision(value: str) -> str:
    key = value.strip().upper()
    if key in {"", "KEEP", "OK", "K"}:
        return "KEEP"
    if key in {"DROP", "F"}:
        return "DROP"
    if key in {"HOLD", "C"}:
        return "HOLD"
    raise ValueError(f"unsupported decision: {value}")


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
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
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"candidate CSV missing columns: {sorted(missing)}")
    normalized_ids = [row["normalized_id"] for row in rows]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("candidate CSV has duplicate normalized_id")
    return rows


def decision_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(normalize_decision(row.get("decision", "")) for row in rows))


def _start_script() -> str:
    return (
        "@echo off\n"
        "cd /d \"%~dp0\"\n"
        "start \"\" http://127.0.0.1:8788/bbox_review.html\n"
        "python \"%~dp0manual_exclusion_bbox_server.py\" "
        "--root \"%~dp0\" --summary \"%~dp0review_setup_summary.json\"\n"
    )


def _write_static_assets(out: Path) -> None:
    for name in STATIC_ASSETS:
        shutil.copyfile(ASSET_DIR / name, out / name)

    redirect = (
        '<!doctype html><meta charset="utf-8"><title>BBox Review</title>'
        '<script>location.replace("bbox_review.html")</script>\n'
    )
    (out / "index.html").write_text(redirect, encoding="utf-8")
    (out / "manual_exclusion_review.html").write_text(redirect, encoding="utf-8")
    (out / "bbox_review_with_shortcuts.html").write_text(redirect, encoding="utf-8")
    (out / "manual_exclusion_review_app.js").write_text(
        "// Legacy page redirects to bbox_review.html.\n",
        encoding="utf-8",
    )


def write_assets(out: Path, candidate_csv: Path) -> dict[str, Any]:
    rows = read_candidates(candidate_csv)
    out.mkdir(parents=True, exist_ok=True)
    _write_static_assets(out)
    shutil.copyfile(SERVER_SOURCE, out / "manual_exclusion_bbox_server.py")
    shutil.copyfile(SERVER_SOURCE, out / "manual_exclusion_review_server.py")
    (out / "start_manual_exclusion_review.bat").write_text(
        _start_script(),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# v3 Manual Exclusion and BBox Review\n\n"
        "Open `bbox_review.html` through `start_manual_exclusion_review.bat`. "
        "Blank decisions are implicit KEEP. K/F/C save KEEP/DROP/HOLD; V only toggles "
        "BBox visibility. BBox edits are stored separately in SQLite, and source v3 labels "
        "are never modified.\n",
        encoding="utf-8",
    )

    decisions_path = out / "manual_exclusion_decisions.csv"
    if not decisions_path.exists():
        with decisions_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=DECISION_COLUMNS).writeheader()

    summary = {
        "candidate_csv": str(candidate_csv.resolve()),
        "total_candidates": len(rows),
        "decision_counts": decision_counts([]),
        "server_port": 8788,
        "default_page": "bbox_review.html",
    }
    (out / "review_setup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the v3 manual exclusion and BBox review UI without changing source data."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv or sys.argv[1:])

    if not args.candidates.is_file():
        print(f"[ERROR] missing candidate CSV: {args.candidates}")
        return 1
    rows = read_candidates(args.candidates)
    if args.dry_run:
        print(
            json.dumps(
                {"status": "passed", "total_candidates": len(rows), "out": str(args.out)},
                ensure_ascii=False,
            )
        )
        return 0
    print(json.dumps(write_assets(args.out, args.candidates), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
