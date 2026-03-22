from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import YOLO_CONFIG_DIR

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("YOLO_VERBOSE", "False")

from ultralytics import YOLO

from .helpers import (
    _appearance_score,
    _blend_feature,
    _box_distance_score,
    _box_size_score,
    _extract_box_feature,
    _iou,
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    track_id: int | None = None
    inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        from .helpers import _round

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": _round(self.confidence),
            "box": list(self.box),
            "track_id": self.track_id,
            "inferred": self.inferred,
        }


@dataclass
class ModelRuntime:
    model: YOLO
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class TrackRecord:
    box: tuple[int, int, int, int]
    feature: np.ndarray | None = None
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    missed: int = 0


@dataclass
class WorkerIdentityRecord:
    box: tuple[int, int, int, int]
    feature: np.ndarray | None = None
    last_seen_frame: int = 0
    last_raw_track_key: str | None = None
    frames_seen: int = 0


@dataclass
class PPEPresenceRecord:
    last_seen_frame: dict[str, int] = field(default_factory=dict)


class PPEPresenceSmoother:
    def __init__(self, persistence_frames: int) -> None:
        self.persistence_frames = max(1, persistence_frames)
        self.records: dict[str, PPEPresenceRecord] = {}

    def resolve(
        self,
        track_key: str,
        observed_items: dict[str, Detection],
        required_ppe: list[str],
        frame_index: int,
    ) -> dict[str, Any]:
        record = self.records.setdefault(track_key, PPEPresenceRecord())
        supported_by_memory: list[str] = []
        missing_now: list[str] = []
        effective_scores: dict[str, float] = {}

        for name in required_ppe:
            if name in observed_items:
                record.last_seen_frame[name] = frame_index
                effective_scores[name] = 1.0
                continue

            last_seen = record.last_seen_frame.get(name)
            if last_seen is not None and (frame_index - last_seen) <= self.persistence_frames:
                supported_by_memory.append(name)
                effective_scores[name] = 1.0
                continue

            effective_scores[name] = 0.0
            missing_now.append(name)

        return {
            "effective_scores": effective_scores,
            "missing_now": missing_now,
            "supported_by_memory": supported_by_memory,
        }


class SimpleBoxTracker:
    def __init__(self, iou_threshold: float = 0.08, max_missed: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: dict[int, TrackRecord] = {}

    def _iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        return _iou(box_a, box_b)

    def _distance_score(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        return _box_distance_score(box_a, box_b)

    def _size_score(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        return _box_size_score(box_a, box_b)

    def _predict_box(self, track: TrackRecord) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = track.box
        vx1, vy1, vx2, vy2 = track.velocity
        return (
            int(round(x1 + vx1)),
            int(round(y1 + vy1)),
            int(round(x2 + vx2)),
            int(round(y2 + vy2)),
        )

    def _extract_feature(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
        return _extract_box_feature(frame, box)

    def _appearance_score(self, feature_a: np.ndarray | None, feature_b: np.ndarray | None) -> float:
        return _appearance_score(feature_a, feature_b)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        if not detections:
            self._age_tracks()
            return []

        features = [self._extract_feature(frame, detection.box) for detection in detections]
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                predicted_box = self._predict_box(track)
                iou = self._iou(detection.box, predicted_box)
                distance = self._distance_score(detection.box, predicted_box)
                size = self._size_score(detection.box, predicted_box)
                appearance = self._appearance_score(features[detection_index], track.feature)
                score = (iou * 0.35) + (distance * 0.30) + (size * 0.15) + (appearance * 0.20)
                if iou >= self.iou_threshold or (distance >= 0.52 and appearance >= 0.45):
                    candidates.append((score, detection_index, track_id))

        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(candidates, key=lambda item: item[0], reverse=True):
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            detections[detection_index].track_id = track_id
            track = self.tracks[track_id]
            previous_box = track.box
            current_box = detections[detection_index].box
            track.velocity = tuple(float(current - previous) for current, previous in zip(current_box, previous_box))
            track.box = current_box
            feature = features[detection_index]
            track.feature = _blend_feature(track.feature, feature, new_weight=0.30)
            track.missed = 0
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            detection.track_id = track_id
            self.tracks[track_id] = TrackRecord(box=detection.box, feature=features[detection_index], missed=0)

        for track_id, track in list(self.tracks.items()):
            if track_id in matched_tracks:
                continue
            if not any(detection.track_id == track_id for detection in detections):
                track.missed += 1
            if track.missed > self.max_missed:
                self.tracks.pop(track_id, None)

        return detections

    def _age_tracks(self) -> None:
        for track_id, track in list(self.tracks.items()):
            track.missed += 1
            if track.missed > self.max_missed:
                self.tracks.pop(track_id, None)


class WorkerIdentityResolver:
    def __init__(self, max_gap_frames: int = 90, min_match_score: float = 0.46) -> None:
        self.max_gap_frames = max(12, max_gap_frames)
        self.min_match_score = min_match_score
        self.next_worker_id = 1
        self.track_to_worker: dict[str, int] = {}
        self.records: dict[int, WorkerIdentityRecord] = {}

    def assign(
        self,
        persons: list[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, Any]]:
        if not persons:
            self.prune(frame_index)
            return []

        features = [_extract_box_feature(frame, person.box) for person in persons]
        assigned_workers: set[int] = set()
        assignments: list[dict[str, Any] | None] = [None] * len(persons)

        for index, person in enumerate(persons):
            raw_track_key = str(person.track_id) if person.track_id is not None else f"frame-{frame_index}-person-{index + 1}"
            worker_id = self.track_to_worker.get(raw_track_key)
            if worker_id is None:
                continue
            assignments[index] = {
                "raw_track_key": raw_track_key,
                "worker_id": worker_id,
            }
            assigned_workers.add(worker_id)

        for index, person in enumerate(persons):
            if assignments[index] is not None:
                continue

            raw_track_key = str(person.track_id) if person.track_id is not None else f"frame-{frame_index}-person-{index + 1}"
            worker_id = self._resolve_worker(person.box, features[index], frame_index, assigned_workers)
            if worker_id is None:
                worker_id = self.next_worker_id
                self.next_worker_id += 1

            self.track_to_worker[raw_track_key] = worker_id
            assignments[index] = {
                "raw_track_key": raw_track_key,
                "worker_id": worker_id,
            }
            assigned_workers.add(worker_id)

        resolved: list[dict[str, Any]] = []
        for index, person in enumerate(persons):
            assignment = assignments[index]
            if assignment is None:
                continue

            worker_id = int(assignment["worker_id"])
            raw_track_key = str(assignment["raw_track_key"])
            record = self.records.get(worker_id)
            if record is None:
                record = WorkerIdentityRecord(
                    box=person.box,
                    feature=features[index],
                    last_seen_frame=frame_index,
                    last_raw_track_key=raw_track_key,
                    frames_seen=1,
                )
                self.records[worker_id] = record
            else:
                record.box = person.box
                record.feature = _blend_feature(record.feature, features[index], new_weight=0.25)
                record.last_seen_frame = frame_index
                record.last_raw_track_key = raw_track_key
                record.frames_seen += 1

            resolved.append(
                {
                    "track_key": str(worker_id),
                    "track_id": worker_id,
                    "track_label": f"ID {worker_id}",
                    "raw_track_key": raw_track_key,
                }
            )

        self.prune(frame_index)
        return resolved

    def _resolve_worker(
        self,
        box: tuple[int, int, int, int],
        feature: np.ndarray | None,
        frame_index: int,
        assigned_workers: set[int],
    ) -> int | None:
        best_worker_id: int | None = None
        best_score = 0.0

        for worker_id, record in self.records.items():
            if worker_id in assigned_workers:
                continue

            frame_gap = frame_index - record.last_seen_frame
            if frame_gap < 0 or frame_gap > self.max_gap_frames:
                continue

            iou = _iou(box, record.box)
            distance = _box_distance_score(box, record.box)
            size = _box_size_score(box, record.box)
            appearance = _appearance_score(feature, record.feature)
            time_score = max(0.0, 1.0 - (frame_gap / self.max_gap_frames))
            score = (0.24 * iou) + (0.33 * distance) + (0.18 * size) + (0.20 * appearance) + (0.05 * time_score)

            if score < self.min_match_score and not (
                (distance >= 0.56 and size >= 0.52)
                or (iou >= 0.12 and appearance >= 0.45)
                or (appearance >= 0.72 and size >= 0.55)
            ):
                continue

            if score > best_score:
                best_score = score
                best_worker_id = worker_id

        return best_worker_id

    def prune(self, frame_index: int) -> None:
        stale_worker_ids = [
            worker_id
            for worker_id, record in self.records.items()
            if frame_index - record.last_seen_frame > (self.max_gap_frames * 4)
        ]
        for worker_id in stale_worker_ids:
            self.records.pop(worker_id, None)
