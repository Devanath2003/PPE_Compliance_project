from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ROOT_DIR


def _now_slug() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "sample"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return cleaned.strip("_") or "sample"


def _as_relative_url(path: Path) -> str:
    return "/" + path.relative_to(ROOT_DIR).as_posix()


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _to_box(values: np.ndarray, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in values]
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w - 1)),
        max(0, min(y2, h - 1)),
    )


def _intersection(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = min(ax2, bx2) - max(ax1, bx1)
    dy = min(ay2, by2) - max(ay1, by1)
    if dx <= 0 or dy <= 0:
        return 0
    return dx * dy


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    inter = _intersection(box_a, box_b)
    union = _area(box_a) + _area(box_b) - inter
    return inter / union if union else 0.0


def _box_distance_score(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    aw = max(1.0, box_a[2] - box_a[0])
    ah = max(1.0, box_a[3] - box_a[1])
    bw = max(1.0, box_b[2] - box_b[0])
    bh = max(1.0, box_b[3] - box_b[1])
    scale = max(aw, ah, bw, bh)
    distance = np.hypot(ax - bx, ay - by) / scale
    return max(0.0, 1.0 - distance)


def _box_size_score(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    aw = max(1.0, box_a[2] - box_a[0])
    ah = max(1.0, box_a[3] - box_a[1])
    bw = max(1.0, box_b[2] - box_b[0])
    bh = max(1.0, box_b[3] - box_b[1])
    width_score = min(aw, bw) / max(aw, bw)
    height_score = min(ah, bh) / max(ah, bh)
    return 0.5 * (width_score + height_score)


def _extract_box_feature(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _appearance_score(feature_a: np.ndarray | None, feature_b: np.ndarray | None) -> float:
    if feature_a is None or feature_b is None:
        return 0.5
    similarity = cv2.compareHist(feature_a.astype(np.float32), feature_b.astype(np.float32), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))


def _blend_feature(
    base_feature: np.ndarray | None,
    new_feature: np.ndarray | None,
    new_weight: float = 0.30,
) -> np.ndarray | None:
    if new_feature is None:
        return base_feature
    if base_feature is None:
        return new_feature
    return (base_feature * (1.0 - new_weight)) + (new_feature * new_weight)


def _expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    x_ratio: float,
    y_ratio: float,
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    ex = int(bw * x_ratio)
    ey = int(bh * y_ratio)
    return (
        max(0, x1 - ex),
        max(0, y1 - ey),
        min(w - 1, x2 + ex),
        min(h - 1, y2 + ey),
    )


def _default_required_ppe(class_names: list[str]) -> list[str]:
    ppe_names = [name for name in class_names if name.lower() != "person"]
    if not ppe_names:
        return []

    preferred = [name for name in ppe_names if name.lower() != "shoes"]
    return preferred or ppe_names
