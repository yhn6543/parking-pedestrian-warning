import csv
import json
import math
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_NAME_TO_ID = {"person": 0, "car": 1}


def normalize_class_name(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"^\d+\s*[-_:]\s*", "", value)
    value = re.sub(r"[-_]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _to_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def load_yolo_data_yaml(yaml_path: str | Path) -> dict:
    path = _to_path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"data.yaml not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"data.yaml must contain a mapping: {path}")
    data["_yaml_path"] = str(path)
    data["_yaml_dir"] = str(path.parent)
    data["_names_by_id"] = parse_names(data.get("names", {}))
    return data


def parse_names(names: Any) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        parsed = {}
        for key, value in names.items():
            parsed[int(key)] = str(value)
        return dict(sorted(parsed.items()))
    raise ValueError("names must be a list or dict.")


def _resolve_relative_path(value: str | Path, dataset_root: Path, yaml_dir: Path) -> Path:
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((dataset_root / path).resolve())
        candidates.append((yaml_dir / path).resolve())
        if str(path).startswith("../"):
            candidates.append((dataset_root / str(path).replace("../", "", 1)).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _split_yaml_key(data_yaml: dict, split: str) -> str | None:
    if split == "valid":
        return "val" if "val" in data_yaml else "valid" if "valid" in data_yaml else None
    return split if split in data_yaml else None


def resolve_split_image_dir(dataset_root: str | Path, data_yaml: dict, split: str) -> Path | None:
    root = _to_path(dataset_root).resolve()
    yaml_dir = Path(data_yaml.get("_yaml_dir", root)).resolve()
    key = _split_yaml_key(data_yaml, split)

    if key and data_yaml.get(key):
        configured = data_yaml[key]
        if isinstance(configured, str):
            path = _resolve_relative_path(configured, root, yaml_dir)
            if path.is_dir():
                return path
            if path.is_file():
                return path

    split_names = ["valid", "val"] if split == "valid" else [split]
    candidates = []
    for name in split_names:
        candidates.extend(
            [
                root / name / "images",
                root / "images" / name,
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def infer_label_dir_from_image_dir(image_dir: Path, dataset_root: Path) -> Path | None:
    image_dir = image_dir.resolve()
    root = dataset_root.resolve()
    parts = list(image_dir.parts)
    candidates = []

    for index, part in enumerate(parts):
        if part.lower() == "images":
            replaced = parts.copy()
            replaced[index] = "labels"
            candidates.append(Path(*replaced))

    if image_dir.name.lower() == "images":
        candidates.append(image_dir.parent / "labels")
    if image_dir.parent.name.lower() == "images":
        candidates.append(image_dir.parent.parent / "labels" / image_dir.name)

    split = image_dir.parent.name if image_dir.name.lower() == "images" else image_dir.name
    candidates.extend(
        [
            root / split / "labels",
            root / "labels" / split,
            image_dir.parent / "labels",
            image_dir.parent.parent / "labels" / image_dir.name,
        ]
    )

    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_dir():
            return normalized
    return None


def list_image_files(image_dir: Path) -> list[Path]:
    if not image_dir or not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def iter_yolo_image_label_pairs(image_dir: Path, label_dir: Path):
    for image_path in list_image_files(image_dir):
        rel = image_path.relative_to(image_dir)
        label_path = (label_dir / rel).with_suffix(".txt")
        if not label_path.exists():
            label_path = label_dir / f"{image_path.stem}.txt"
        yield image_path, label_path if label_path.exists() else None


def _polygon_to_bbox(values: list[float]) -> list[float] | None:
    if len(values) < 6 or len(values) % 2 != 0:
        return None
    xs = values[0::2]
    ys = values[1::2]
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        max_x - min_x,
        max_y - min_y,
    ]


def convert_label_line_to_target(
    line: str,
    source_names: dict[int, str],
    class_map: dict[str, str],
    ignore_classes: list[str],
) -> tuple[str | None, str | None]:
    stripped = line.strip()
    if not stripped:
        return None, "empty_line"
    parts = stripped.split()
    if len(parts) < 5:
        return None, "too_few_values"

    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None, "non_numeric_value"

    if class_id not in source_names:
        return None, f"class_id_out_of_range:{class_id}"

    raw_name = normalize_class_name(source_names[class_id])
    normalized_map = {normalize_class_name(key): value for key, value in class_map.items()}
    normalized_ignore = {normalize_class_name(item) for item in ignore_classes}

    if raw_name in normalized_ignore:
        return None, None
    if raw_name not in normalized_map:
        return None, f"unknown_class:{raw_name}"

    target_name = normalized_map[raw_name]
    if target_name not in TARGET_NAME_TO_ID:
        return None, f"unknown_target_class:{target_name}"
    target_id = TARGET_NAME_TO_ID[target_name]

    if len(values) == 4:
        bbox = values
    else:
        bbox = _polygon_to_bbox(values)
        if bbox is None:
            return None, "invalid_segmentation_values"

    if any(not math.isfinite(value) for value in bbox):
        return None, "non_finite_bbox"
    x_center, y_center, width, height = bbox
    if not all(0.0 <= value <= 1.0 for value in bbox):
        return None, "bbox_out_of_range"
    if width <= 0 or height <= 0:
        return None, "bbox_non_positive_size"

    return f"{target_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", None


def ensure_yolo_split_dirs(root: Path) -> None:
    for split in ["train", "valid", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def write_yolo_data_yaml(root: str | Path, names: dict[int, str] | None = None) -> str:
    root = _to_path(root)
    payload = {
        "path": root.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names or {0: "person", 1: "car"},
    }
    output_path = root / "data.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(output_path)


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> str:
    output = _to_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(output)


def write_json(path: str | Path, payload: Any) -> str:
    output = _to_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


@dataclass
class SplitResolution:
    split: str
    image_dir: str | None
    label_dir: str | None
    image_count: int
    label_count: int


def resolve_dataset_splits(dataset_root: str | Path, data_yaml: dict) -> dict[str, SplitResolution]:
    root = _to_path(dataset_root).resolve()
    result = {}
    for split in ["train", "valid", "test"]:
        image_dir = resolve_split_image_dir(root, data_yaml, split)
        label_dir = infer_label_dir_from_image_dir(image_dir, root) if image_dir else None
        image_count = len(list_image_files(image_dir)) if image_dir else 0
        label_count = len(list(label_dir.rglob("*.txt"))) if label_dir else 0
        result[split] = SplitResolution(
            split=split,
            image_dir=str(image_dir) if image_dir else None,
            label_dir=str(label_dir) if label_dir else None,
            image_count=image_count,
            label_count=label_count,
        )
    return result


def convert_label_file(
    label_path: Path | None,
    output_label_path: Path,
    source_names: dict[int, str],
    class_map: dict[str, str],
    ignore_classes: list[str],
) -> tuple[int, Counter, list[dict]]:
    converted = []
    counts = Counter()
    warnings = []
    if label_path and label_path.exists():
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        lines = []

    for line_index, line in enumerate(lines, start=1):
        converted_line, warning = convert_label_line_to_target(
            line=line,
            source_names=source_names,
            class_map=class_map,
            ignore_classes=ignore_classes,
        )
        if warning:
            warnings.append({"label_path": str(label_path), "line": line_index, "warning": warning})
        if converted_line:
            target_id = int(converted_line.split()[0])
            counts[target_id] += 1
            converted.append(converted_line)

    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    return len(converted), counts, warnings


def copy_image(source: Path, destination: Path, copy_mode: str = "copy") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        if destination.exists():
            destination.unlink()
        destination.symlink_to(source.resolve())
    elif copy_mode == "hardlink":
        if destination.exists():
            destination.unlink()
        try:
            destination.hardlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)


def count_yolo_classes(label_dir: Path) -> Counter:
    counts = Counter()
    if not label_dir.exists():
        return counts
    for label_path in label_dir.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    counts[int(float(parts[0]))] += 1
                except ValueError:
                    pass
    return counts


def count_dataset_classes(dataset_root: Path) -> list[dict]:
    rows = []
    for split in ["train", "valid", "test"]:
        label_dir = dataset_root / split / "labels"
        counts = count_yolo_classes(label_dir)
        for class_id, class_name in [(0, "person"), (1, "car")]:
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox_count": int(counts.get(class_id, 0)),
                }
            )
    return rows


def choose_smoke_images(image_dir: Path, label_dir: Path, limit: int, seed: int) -> list[Path]:
    images = list_image_files(image_dir)
    rng = random.Random(seed)
    scored = []
    for image_path in images:
        label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
        if not label_path.exists():
            label_path = label_dir / f"{image_path.stem}.txt"
        class_ids = set()
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        class_ids.add(int(float(parts[0])))
                    except ValueError:
                        pass
        score = (0 in class_ids) + (1 in class_ids)
        scored.append((score, rng.random(), image_path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[: max(0, int(limit))]]


def create_synthetic_image(path: Path, label_path: Path, index: int, width: int = 320, height: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), color=(35 + index * 7 % 60, 45, 55))
    draw = ImageDraw.Draw(image)

    person_box = (50, 60, 95, 180)
    car_box = (160, 110, 285, 175)
    draw.rectangle(person_box, outline=(255, 230, 80), width=3)
    draw.rectangle(car_box, outline=(80, 190, 255), width=3)
    draw.text((12, 12), f"synthetic {index}", fill=(255, 255, 255))
    image.save(path)

    def yolo_line(class_id: int, box: tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

    label_path.write_text("\n".join([yolo_line(0, person_box), yolo_line(1, car_box)]) + "\n", encoding="utf-8")
