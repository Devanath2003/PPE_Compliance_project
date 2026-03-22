from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "runs"
UPLOADS_DIR = ROOT_DIR / "uploads"
YOLO_CONFIG_DIR = ROOT_DIR / ".yolo_config"

APP_TITLE = "PPE Compliance Monitor"
APP_SUBTITLE = "YOLOv11 + STCA powered industrial safety monitoring"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_COMPLIANCE_THRESHOLD = 0.75
DEFAULT_TEMPORAL_WINDOW = 20
DEFAULT_IMAGE_SIZE = 960

PAPER_HIGHLIGHTS = [
    "YOLOv11 handles the spatial PPE detections while STCA adds temporal reasoning.",
    "TEMF reduces noisy compliance swings by minimizing temporal uncertainty.",
    "ACSF weights recent frames more heavily to produce a stable compliance score.",
    "The paper focused on helmet and vest, but this project generalizes the same logic to every PPE class present in the trained model.",
]


def list_model_paths() -> list[Path]:
    return sorted(ROOT_DIR.glob("*.pt"))


def default_model_name() -> str | None:
    preferred = ROOT_DIR / "best.pt"
    if preferred.exists():
        return preferred.name

    models = list_model_paths()
    return models[0].name if models else None

