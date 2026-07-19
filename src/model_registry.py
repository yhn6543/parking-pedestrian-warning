from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FINETUNED_DIR = Path("models") / "fine_tuned"
FINETUNED_DEFAULT_PATH = FINETUNED_DIR / "parking_yolo11n_best.pt"
PERSON_ONLY_DEFAULT_PATH = FINETUNED_DIR / "parking_yolo11n_person_only_best.pt"
YOLO26_PERSON_ONLY_DEFAULT_PATH = FINETUNED_DIR / "parking_yolo26n_person_only_best.pt"
FINETUNED_2CLASS_MODEL_KEY = "yolo11n_finetuned_2class"
PERSON_ONLY_MODEL_KEY = "yolo11n_person_only"
YOLO26_PRETRAINED_MODEL_KEY = "yolo26n_pretrained"
YOLO26_PERSON_ONLY_MODEL_KEY = "yolo26n_person_only"
MISSING_FINETUNED_WARNING = (
    "Fine-tuned YOLO11n person/car model file was not found. "
    "Copy best.pt to models/fine_tuned/parking_yolo11n_best.pt."
)
MISSING_PERSON_ONLY_WARNING = (
    "person-only 모델 파일을 찾을 수 없습니다. 학습 완료 후 best.pt를 "
    "models/fine_tuned/parking_yolo11n_person_only_best.pt 위치에 복사해 주세요."
)
MISSING_YOLO26_PERSON_ONLY_WARNING = (
    "YOLO26n person-only 모델 파일을 찾을 수 없습니다. 학습 완료 후 best.pt를 "
    "models/fine_tuned/parking_yolo26n_person_only_best.pt 위치에 복사해 주세요."
)


@dataclass(frozen=True)
class ModelOption:
    key: str
    display_name: str
    path: str
    model_type: str
    expected_classes: tuple[str, ...]
    exists: bool
    recommended_confidence: float = 0.35
    warning: str = ""


def _project_path(project_root: Path, relative_path: Path) -> Path:
    return project_root.resolve() / relative_path


def find_finetuned_best_model(project_root: Path) -> Path | None:
    root = Path(project_root).resolve()
    default_path = _project_path(root, FINETUNED_DEFAULT_PATH)
    if default_path.exists():
        return default_path

    fine_tuned_dir = _project_path(root, FINETUNED_DIR)
    if not fine_tuned_dir.exists():
        return None

    candidates = sorted(
        (path for path in fine_tuned_dir.glob("*.pt") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _fixed_option(
    project_root: Path,
    key: str,
    display_name: str,
    path: str,
    model_type: str,
    expected_classes: tuple[str, ...],
    missing_warning: str | None = None,
    recommended_confidence: float = 0.35,
) -> ModelOption:
    absolute = _project_path(project_root, Path(path))
    exists = absolute.exists()
    return ModelOption(
        key=key,
        display_name=display_name,
        path=path,
        model_type=model_type,
        expected_classes=expected_classes,
        exists=exists,
        recommended_confidence=float(recommended_confidence),
        warning="" if exists else (missing_warning or f"Model file does not exist: {path}"),
    )


def get_fixed_model_options(project_root: Path) -> list[ModelOption]:
    root = Path(project_root).resolve()

    return [
        _fixed_option(
            root,
            key="yolov8n_pretrained",
            display_name="YOLOv8n pretrained",
            path="yolov8n.pt",
            model_type="coco_pretrained",
            expected_classes=("person", "car"),
        ),
        _fixed_option(
            root,
            key="yolo11n_pretrained",
            display_name="YOLO11n pretrained",
            path="yolo11n.pt",
            model_type="coco_pretrained",
            expected_classes=("person", "car"),
        ),
        _fixed_option(
            root,
            key=FINETUNED_2CLASS_MODEL_KEY,
            display_name="Fine-tuned YOLO11n person/car",
            path=FINETUNED_DEFAULT_PATH.as_posix(),
            model_type="project_2class",
            expected_classes=("person", "car"),
            missing_warning=MISSING_FINETUNED_WARNING,
        ),
        _fixed_option(
            root,
            key=PERSON_ONLY_MODEL_KEY,
            display_name="Fine-tuned YOLO11n person-only",
            path=PERSON_ONLY_DEFAULT_PATH.as_posix(),
            model_type="project_person_only",
            expected_classes=("person",),
            missing_warning=MISSING_PERSON_ONLY_WARNING,
            recommended_confidence=0.25,
        ),
        _fixed_option(
            root,
            key=YOLO26_PRETRAINED_MODEL_KEY,
            display_name="YOLO26n pretrained",
            path="yolo26n.pt",
            model_type="coco_pretrained",
            expected_classes=("person", "car"),
            recommended_confidence=0.30,
        ),
        _fixed_option(
            root,
            key=YOLO26_PERSON_ONLY_MODEL_KEY,
            display_name="Fine-tuned YOLO26n person-only",
            path=YOLO26_PERSON_ONLY_DEFAULT_PATH.as_posix(),
            model_type="project_person_only",
            expected_classes=("person",),
            missing_warning=MISSING_YOLO26_PERSON_ONLY_WARNING,
            recommended_confidence=0.25,
        ),
    ]


def build_model_options(project_root: Path) -> list[ModelOption]:
    return get_fixed_model_options(project_root)


def discover_model_files(project_root: Path, external_roots: list[Path] | None = None) -> list[ModelOption]:
    return get_fixed_model_options(project_root)


def get_model_path_from_selection(selected_key: str, options: list[ModelOption]) -> str:
    for option in options:
        if option.key == selected_key:
            return option.path
    return ""


def get_model_option_from_selection(selected_key: str, options: list[ModelOption]) -> ModelOption | None:
    for option in options:
        if option.key == selected_key:
            return option
    return None


def get_default_model_key(options: list[ModelOption]) -> str:
    by_key = {option.key: option for option in options}
    if by_key.get(PERSON_ONLY_MODEL_KEY) and by_key[PERSON_ONLY_MODEL_KEY].exists:
        return PERSON_ONLY_MODEL_KEY
    if by_key.get(FINETUNED_2CLASS_MODEL_KEY) and by_key[FINETUNED_2CLASS_MODEL_KEY].exists:
        return FINETUNED_2CLASS_MODEL_KEY
    if by_key.get("yolo11n_pretrained") and by_key["yolo11n_pretrained"].exists:
        return "yolo11n_pretrained"
    if by_key.get("yolov8n_pretrained") and by_key["yolov8n_pretrained"].exists:
        return "yolov8n_pretrained"
    return "yolo11n_pretrained"


def get_default_model_path(options: list[ModelOption]) -> str:
    return get_model_path_from_selection(get_default_model_key(options), options)


def resolve_selected_model_path(
    use_manual_model_path: bool,
    manual_model_path: str,
    selected_model_path: str,
) -> str:
    if use_manual_model_path:
        return str(manual_model_path).strip()
    return str(selected_model_path).strip()


def should_block_model_execution(
    selected_option: ModelOption | None,
    use_manual_model_path: bool = False,
) -> tuple[bool, str]:
    if use_manual_model_path:
        return False, ""
    if (
        selected_option
        and selected_option.model_type in {"project_2class", "project_person_only"}
        and not selected_option.exists
    ):
        return True, selected_option.warning or f"Model file does not exist: {selected_option.path}"
    return False, ""
