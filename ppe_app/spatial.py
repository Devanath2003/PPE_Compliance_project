from __future__ import annotations

from .helpers import _area, _center, _expand_box, _intersection
from .tracking import Detection


BODY_REGION_RULES = {
    "helmet": (0.00, 0.28),
    "glasses": (0.00, 0.35),
    "mask": (0.08, 0.52),
    "vest": (0.24, 0.80),
    "gloves": (0.25, 0.98),
    "shoes": (0.74, 1.04),
}


def infer_person_detection(
    detections: list[Detection],
    frame_shape: tuple[int, int, int],
) -> Detection:
    xs1 = [item.box[0] for item in detections]
    ys1 = [item.box[1] for item in detections]
    xs2 = [item.box[2] for item in detections]
    ys2 = [item.box[3] for item in detections]
    x1 = min(xs1)
    y1 = min(ys1)
    x2 = max(xs2)
    y2 = max(ys2)
    expanded = _expand_box((x1, y1, x2, y2), frame_shape, x_ratio=0.25, y_ratio=0.45)
    return Detection(class_id=0, class_name="Person", confidence=0.0, box=expanded, inferred=True)


def assignment_score(
    person_box: tuple[int, int, int, int],
    ppe_box: tuple[int, int, int, int],
    class_name: str,
    frame_shape: tuple[int, int, int],
) -> float:
    expanded_person = _expand_box(person_box, frame_shape, x_ratio=0.08, y_ratio=0.06)
    overlap = _intersection(expanded_person, ppe_box)
    if overlap <= 0:
        return 0.0

    item_area = _area(ppe_box)
    person_area = _area(person_box)
    cx, cy = _center(ppe_box)
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    pcx, _ = _center(person_box)

    x_norm = max(0.0, 1.0 - abs(cx - pcx) / (pw * 0.7))
    y_ratio = (cy - py1) / ph
    region = BODY_REGION_RULES.get(class_name.lower(), (0.0, 1.0))
    if region[0] <= y_ratio <= region[1]:
        band_score = 1.0
    else:
        distance = min(abs(y_ratio - region[0]), abs(y_ratio - region[1]))
        band_score = max(0.0, 1.0 - distance * 2.5)

    overlap_ratio = overlap / item_area
    person_overlap = overlap / person_area
    return (0.50 * overlap_ratio) + (0.15 * person_overlap) + (0.20 * x_norm) + (0.15 * band_score)
