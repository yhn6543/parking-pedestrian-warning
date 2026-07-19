from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


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


def normalize_decision(value: str | None) -> str:
    key = str(value or "").strip().upper()
    if key in {"", "KEEP", "OK", "K"}:
        return "KEEP"
    if key in {"DROP", "F"}:
        return "DROP"
    if key in {"HOLD", "C"}:
        return "HOLD"
    raise ValueError("decision must be KEEP, DROP, or HOLD")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, root: Path, candidates_path: Path) -> None:
        self.root = Path(root).resolve()
        self.candidates_path = Path(candidates_path).resolve()
        self.decisions_path = self.root / "manual_exclusion_decisions.csv"
        self.bbox_db_path = self.root / "manual_exclusion_bbox_edits.sqlite"
        self.initialize_bbox_db()

    def initialize_bbox_db(self) -> None:
        with sqlite3.connect(self.bbox_db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bbox_edit_sessions(
                    normalized_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    label_path TEXT NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    boxes_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def candidates(self) -> list[dict[str, str]]:
        with self.candidates_path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def candidate(self, normalized_id: str) -> dict[str, str] | None:
        return next(
            (row for row in self.candidates() if row["normalized_id"] == normalized_id),
            None,
        )

    def decisions(self) -> dict[str, dict[str, str]]:
        if not self.decisions_path.exists():
            return {}
        with self.decisions_path.open(newline="", encoding="utf-8-sig") as handle:
            return {
                row["normalized_id"]: row
                for row in csv.DictReader(handle)
                if row.get("normalized_id")
            }

    @staticmethod
    def _image_shape(image_path: str) -> tuple[int, int]:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        height, width = image.shape[:2]
        return width, height

    def load_boxes(self, row: dict[str, str]) -> dict[str, Any]:
        with sqlite3.connect(self.bbox_db_path) as connection:
            saved = connection.execute(
                "SELECT revision, boxes_json, image_width, image_height "
                "FROM bbox_edit_sessions WHERE normalized_id = ?",
                (row["normalized_id"],),
            ).fetchone()
        if saved:
            return {
                "revision": saved[0],
                "boxes": json.loads(saved[1]),
                "image_width": saved[2],
                "image_height": saved[3],
                "edited": True,
            }

        width, height = self._image_shape(row["image_path"])
        boxes = []
        for line in Path(row["label_path"]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, x_center, y_center, box_width, box_height = line.split()
            if int(class_id) != 0:
                raise ValueError("only class 0 person boxes are supported")
            x_center, y_center, box_width, box_height = map(
                float,
                (x_center, y_center, box_width, box_height),
            )
            boxes.append(
                {
                    "class_id": 0,
                    "x1": max(0.0, (x_center - box_width / 2) * width),
                    "y1": max(0.0, (y_center - box_height / 2) * height),
                    "x2": min(float(width), (x_center + box_width / 2) * width),
                    "y2": min(float(height), (y_center + box_height / 2) * height),
                }
            )
        return {
            "revision": 0,
            "boxes": boxes,
            "image_width": width,
            "image_height": height,
            "edited": False,
        }

    def save_boxes(self, row: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        width, height = self._image_shape(row["image_path"])
        boxes = payload.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError("empty boxes are blocked; use DROP for images with no person")

        for box in boxes:
            if int(box.get("class_id", 0)) != 0:
                raise ValueError("only class 0 person boxes are supported")
            x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("bbox coordinates are outside the image")
            if x2 - x1 < 2 or y2 - y1 < 2:
                raise ValueError("bbox is too small")

        requested_revision = int(payload.get("revision", 0))
        with sqlite3.connect(self.bbox_db_path) as connection:
            saved = connection.execute(
                "SELECT revision FROM bbox_edit_sessions WHERE normalized_id = ?",
                (row["normalized_id"],),
            ).fetchone()
            actual_revision = saved[0] if saved else 0
            if requested_revision != actual_revision:
                raise ValueError("stale bbox revision; reload image")
            new_revision = actual_revision + 1
            connection.execute(
                "INSERT OR REPLACE INTO bbox_edit_sessions VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["normalized_id"],
                    row["image_path"],
                    row["label_path"],
                    width,
                    height,
                    new_revision,
                    json.dumps(boxes),
                    utc_now(),
                ),
            )
        return {
            "revision": new_revision,
            "boxes": boxes,
            "image_width": width,
            "image_height": height,
            "edited": True,
        }

    def save_decision(
        self,
        row: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        decisions = self.decisions()
        decision = normalize_decision(payload.get("decision"))
        record = {key: row.get(key, "") for key in DECISION_COLUMNS}
        record.update(
            decision=decision,
            decision_label=decision,
            note=str(payload.get("note", "")).strip(),
            updated_at=utc_now(),
        )
        decisions[row["normalized_id"]] = record

        temporary_path = self.decisions_path.with_suffix(".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DECISION_COLUMNS)
            writer.writeheader()
            writer.writerows(decisions[key] for key in sorted(decisions))
        os.replace(temporary_path, self.decisions_path)
        return record


class ReviewHandler(BaseHTTPRequestHandler):
    @property
    def store(self) -> ReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        decisions = self.store.decisions()

        if parsed.path == "/api/candidates":
            mode = (query.get("filter") or [""])[0]
            group = (query.get("group") or [""])[0]
            reason = (query.get("reason") or [""])[0]
            output = []
            for row in self.store.candidates():
                saved = decisions.get(row["normalized_id"])
                decision = normalize_decision(saved.get("decision")) if saved else ""
                if mode and not (mode == "undecided" and not decision) and decision != mode:
                    continue
                if group and row["output_group"] != group:
                    continue
                if reason and reason not in row["candidate_reasons"].split(";"):
                    continue
                output.append(
                    {
                        **row,
                        "decision": decision,
                        "note": saved.get("note", "") if saved else "",
                    }
                )
            self.send_json({"ok": True, "rows": output})
            return

        if parsed.path == "/api/item":
            row = self.store.candidate((query.get("id") or [""])[0])
            if not row:
                self.send_json({"ok": False, "error": "not found"}, 404)
                return
            saved = decisions.get(row["normalized_id"])
            self.send_json(
                {
                    "ok": True,
                    "item": {
                        **row,
                        **self.store.load_boxes(row),
                        "decision": normalize_decision(saved.get("decision")) if saved else "",
                        "note": saved.get("note", "") if saved else "",
                    },
                }
            )
            return

        if parsed.path in {"/api/image", "/api/overlay"}:
            row = self.store.candidate((query.get("id") or [""])[0])
            if not row:
                self.send_error(404)
                return
            key = "image_path" if parsed.path == "/api/image" else "overlay_path"
            self.serve_file(Path(row[key]))
            return

        relative_path = "bbox_review.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        static_path = (self.store.root / relative_path).resolve()
        if static_path != self.store.root and self.store.root not in static_path.parents:
            self.send_error(403)
            return
        self.serve_file(static_path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/bbox", "/api/decision"}:
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            row = self.store.candidate(str(payload.get("normalized_id", "")))
            if not row:
                raise ValueError("candidate not found")
            if path == "/api/bbox":
                value = self.store.save_boxes(row, payload)
            else:
                value = self.store.save_decision(row, payload)
            self.send_json({"ok": True, "value": value})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the v3 exclusion and BBox review UI.")
    parser.add_argument("--root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", type=Path)
    source.add_argument("--summary", type=Path)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f"review root does not exist: {args.root}")
    candidates_path = args.candidates
    if args.summary:
        if not args.summary.is_file():
            parser.error(f"review summary does not exist: {args.summary}")
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        candidates_path = Path(summary.get("candidate_csv", ""))
    if not candidates_path or not candidates_path.is_file():
        parser.error(f"candidate CSV does not exist: {candidates_path}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    server.store = ReviewStore(args.root, candidates_path)  # type: ignore[attr-defined]
    print(f"BBox review: http://127.0.0.1:{args.port}/bbox_review.html")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
