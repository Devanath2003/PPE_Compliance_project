from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .helpers import _round


def update_missing_evidence(
    bucket: dict[str, Any],
    required_ppe: list[str],
    missing_now: list[str],
) -> None:
    missing_set = set(missing_now)
    streaks = bucket.setdefault("current_missing_streaks", {})
    max_streaks = bucket.setdefault("max_missing_streaks", Counter())
    missing_frames = bucket.setdefault("missing_frames", Counter())

    for name in required_ppe:
        if name in missing_set:
            missing_frames[name] += 1
            streaks[name] = streaks.get(name, 0) + 1
            max_streaks[name] = max(max_streaks.get(name, 0), streaks[name])
        else:
            streaks[name] = 0


def persistent_missing_items(
    missing_frames: Counter,
    max_streaks: Counter,
    total_frames: int,
    fps: float,
) -> list[str]:
    if total_frames <= 0:
        return []

    min_streak = max(3, int(round(fps * 0.5)))
    min_ratio = 0.35
    persistent: list[tuple[str, int, int]] = []
    for name, frames_missing in missing_frames.items():
        ratio = frames_missing / total_frames
        longest = int(max_streaks.get(name, 0))
        if longest >= min_streak or ratio >= min_ratio:
            persistent.append((name, frames_missing, longest))

    persistent.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return [name for name, _, _ in persistent]


def close_event(event: dict[str, Any], fps: float, required_ppe: list[str]) -> dict[str, Any]:
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    duration_frames = max(1, end_frame - start_frame + 1)
    missing = persistent_missing_items(
        event.get("missing_frames", Counter()),
        event.get("max_missing_streaks", Counter()),
        event.get("frame_count", duration_frames),
        fps,
    )
    if not missing:
        missing = persistent_missing_items(
            event.get("missing_frames", Counter()),
            event.get("max_missing_streaks", Counter()),
            duration_frames,
            max(fps * 0.5, 1.0),
        )
    return {
        "track_key": event["track_key"],
        "track_label": event["track_label"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_seconds": _round(start_frame / fps),
        "end_time_seconds": _round(end_frame / fps),
        "duration_seconds": _round(duration_frames / fps),
        "missing_ppe": missing,
    }


def write_events_csv(output_path: Path, events: list[dict[str, Any]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "track_key",
                "track_label",
                "start_frame",
                "end_frame",
                "start_time_seconds",
                "end_time_seconds",
                "duration_seconds",
                "missing_ppe",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow({**event, "missing_ppe": ", ".join(event["missing_ppe"])})


def finalize_track_rollups(track_rollups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for item in track_rollups.values():
        frames_seen = item["frames_seen"]
        dominant = item["status_counts"].most_common(1)[0][0] if item["status_counts"] else "compliant"
        persistent_missing = persistent_missing_items(
            item.get("missing_frames", Counter()),
            item.get("max_missing_streaks", Counter()),
            frames_seen,
            fps=12.0,
        )
        finalized.append(
            {
                "track_key": item["track_key"],
                "track_label": item["track_label"],
                "frames_seen": frames_seen,
                "average_adaptive_score": _round(item["score_sum"] / frames_seen if frames_seen else 0.0),
                "dominant_status": dominant,
                "status_counts": dict(item["status_counts"]),
                "source_tracks": sorted(str(track_key) for track_key in item.get("source_tracks", set())),
                "persistent_missing": persistent_missing,
            }
        )
    finalized.sort(key=lambda row: int(row["track_key"]) if str(row["track_key"]).isdigit() else row["track_label"])
    return finalized
