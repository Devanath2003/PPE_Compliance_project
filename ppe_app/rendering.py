from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .tracking import Detection


COLOR_PALETTE = {
    "compliant": (45, 189, 108),
    "warning": (25, 175, 235),
    "violation": (51, 85, 255),
    "ppe": (255, 215, 0),
    "panel": (18, 21, 30),
    "panel_alt": (31, 38, 52),
    "text": (245, 247, 250),
    "muted": (181, 188, 201),
}

VIDEO_CODECS = [
    (".mp4", "mp4v"),
    (".webm", "VP80"),
    (".avi", "MJPG"),
    (".mp4", "avc1"),
]

MAX_BROWSER_PREVIEW_FRAMES = 240
PREVIEW_MAX_WIDTH = 960
PREVIEW_JPEG_QUALITY = 72


def render_frame(
    frame: np.ndarray,
    people: list[dict[str, Any]],
    ppe_items: list[Detection],
) -> np.ndarray:
    for report in people:
        x1, y1, x2, y2 = report["box"]
        color = COLOR_PALETTE["compliant"] if report["status_kind"] == "compliant" else COLOR_PALETTE["violation"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = report["track_label"]
        if report["inferred_person_box"]:
            label += " *"
        label_box(frame, label, (x1, max(20, y1 - 8)), color, dark_text=False)

    for item in ppe_items:
        x1, y1, x2, y2 = item.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PALETTE["ppe"], 2)
        label = f"{item.class_name} {item.confidence:.2f}"
        label_box(frame, label, (x1, max(20, y1 - 8)), COLOR_PALETTE["ppe"], dark_text=True)
    return frame


def prepare_preview_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= PREVIEW_MAX_WIDTH:
        return frame
    scale = PREVIEW_MAX_WIDTH / max(width, 1)
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (PREVIEW_MAX_WIDTH, resized_height), interpolation=cv2.INTER_AREA)


def label_box(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    dark_text: bool,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    x, y = origin
    y = max(th + 6, y)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 10, y + baseline), color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    text_color = (20, 20, 20) if dark_text else COLOR_PALETTE["text"]
    cv2.putText(
        frame,
        text,
        (x + 5, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        1,
        cv2.LINE_AA,
    )
