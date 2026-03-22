from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def temporal_entropy(scores: list[float] | np.ndarray, eps: float = 1e-6) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, eps, 1.0 - eps)
    entropy = -(arr * np.log(arr) + (1.0 - arr) * np.log(1.0 - arr))
    return float(np.mean(entropy))


def adaptive_score(scores: list[float] | np.ndarray, decay: float = 0.10) -> float:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    positions = np.arange(arr.size, dtype=np.float32)
    weights = np.exp(decay * (positions - arr.size + 1))
    weights /= np.sum(weights)
    return float(np.dot(arr, weights))


def score_trend(scores: list[float]) -> str:
    if len(scores) < 5:
        return "warming-up"

    recent = float(np.mean(scores[-5:]))
    older = float(np.mean(scores[-10:-5])) if len(scores) >= 10 else recent
    if recent > older + 0.08:
        return "improving"
    if recent < older - 0.08:
        return "declining"
    return "stable"


@dataclass
class TemporalState:
    history: deque[float]
    last_seen_frame: int = 0


@dataclass
class TemporalComplianceTracker:
    window_size: int
    threshold: float
    decay: float | None = None
    states: dict[str, TemporalState] = field(default_factory=dict)

    def update(self, track_key: str, frame_score: float, frame_index: int) -> dict[str, Any]:
        state = self.states.get(track_key)
        if state is None:
            state = TemporalState(history=deque(maxlen=self.window_size))
            self.states[track_key] = state

        state.history.append(float(frame_score))
        state.last_seen_frame = frame_index

        scores = list(state.history)
        decay = self.decay if self.decay is not None else max(0.005, min(0.05, 0.5 / max(self.window_size, 1)))
        adaptive = adaptive_score(scores, decay=decay)
        entropy = temporal_entropy(scores)

        return {
            "frame_score": round(float(frame_score), 4),
            "adaptive_score": round(adaptive, 4),
            "entropy": round(entropy, 4),
            "trend": score_trend(scores),
            "compliant": adaptive >= self.threshold,
            "window_length": len(scores),
            "decay": round(decay, 4),
        }

    def prune(self, frame_index: int, max_idle_frames: int) -> None:
        stale = [
            track_key
            for track_key, state in self.states.items()
            if frame_index - state.last_seen_frame > max_idle_frames
        ]
        for track_key in stale:
            self.states.pop(track_key, None)


def classify_status(missing_items: list[str], adaptive: float, threshold: float) -> dict[str, str]:
    if adaptive < threshold:
        return {"label": "Violation", "kind": "violation"}
    if missing_items:
        return {"label": "At Risk", "kind": "warning"}
    return {"label": "Compliant", "kind": "compliant"}
