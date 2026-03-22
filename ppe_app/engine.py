from __future__ import annotations

import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_COMPLIANCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TEMPORAL_WINDOW,
    PAPER_HIGHLIGHTS,
    ROOT_DIR,
    RUNS_DIR,
    YOLO_CONFIG_DIR,
    default_model_name,
    list_model_paths,
)

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("YOLO_VERBOSE", "False")

from ultralytics import YOLO

from .helpers import (
    _area,
    _as_relative_url,
    _default_required_ppe,
    _now_slug,
    _round,
    _safe_stem,
    _to_box,
)
from .rendering import (
    MAX_BROWSER_PREVIEW_FRAMES,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_MAX_WIDTH,
    VIDEO_CODECS,
    prepare_preview_frame,
    render_frame,
)
from .reporting import (
    close_event,
    finalize_track_rollups,
    update_missing_evidence,
    write_events_csv,
)
from .spatial import assignment_score, infer_person_detection
from .temporal import TemporalComplianceTracker, classify_status
from .tracking import (
    Detection,
    ModelRuntime,
    PPEPresenceSmoother,
    SimpleBoxTracker,
    WorkerIdentityResolver,
)


class PPEComplianceEngine:
    def __init__(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._model_cache: dict[str, ModelRuntime] = {}
        self._device = 0 if torch.cuda.is_available() else "cpu"

    def list_available_models(self) -> list[str]:
        return [path.name for path in list_model_paths()]

    def default_model(self) -> str:
        model_name = default_model_name()
        if not model_name:
            raise RuntimeError("No .pt weights were found in the project folder.")
        return model_name

    def get_model_metadata(self, model_name: str | None = None) -> dict[str, Any]:
        runtime = self._get_runtime(model_name or self.default_model())
        names = self._class_names(runtime.model)
        return {
            "selected_model": model_name or self.default_model(),
            "available_models": self.list_available_models(),
            "class_names": names,
            "ppe_classes": [name for name in names if name.lower() != "person"],
            "default_required_ppe": _default_required_ppe(names),
            "defaults": {
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "compliance_threshold": DEFAULT_COMPLIANCE_THRESHOLD,
                "temporal_window": DEFAULT_TEMPORAL_WINDOW,
            },
            "paper_highlights": PAPER_HIGHLIGHTS,
        }

    def _get_runtime(self, model_name: str) -> ModelRuntime:
        runtime = self._model_cache.get(model_name)
        if runtime is not None:
            return runtime

        model_path = ROOT_DIR / model_name
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_name}")

        model = YOLO(str(model_path))
        runtime = ModelRuntime(model=model)
        self._model_cache[model_name] = runtime
        return runtime

    def _class_names(self, model: YOLO) -> list[str]:
        names = model.names
        if isinstance(names, dict):
            return [names[index] for index in sorted(names)]
        return list(names)

    def _normalize_required_ppe(self, required_ppe: list[str] | None, class_names: list[str]) -> list[str]:
        allowed = {name for name in class_names if name.lower() != "person"}
        selected = [name for name in (required_ppe or _default_required_ppe(class_names)) if name in allowed]
        return selected or _default_required_ppe(class_names)

    def _validate_image(self, input_path: Path) -> None:
        if input_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise RuntimeError(f"Unsupported image format: {input_path.suffix}")

    def _validate_video(self, input_path: Path) -> None:
        if input_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video format: {input_path.suffix}")

    def _prepare_run_dir(self, prefix: str, source_name: str) -> Path:
        run_dir = RUNS_DIR / f"{prefix}_{_now_slug()}_{_safe_stem(source_name)}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _create_video_writer(
        self,
        run_dir: Path,
        width: int,
        height: int,
        fps: float,
        source_name: str,
    ) -> tuple[cv2.VideoWriter, Path]:
        stem = _safe_stem(source_name)
        for extension, codec in VIDEO_CODECS:
            output_path = run_dir / f"{stem}_annotated{extension}"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                max(fps, 1.0),
                (width, height),
            )
            if writer.isOpened():
                return writer, output_path
            writer.release()
        raise RuntimeError("Could not create a video writer for the annotated output.")

    def analyze_image(
        self,
        input_path: Path,
        model_name: str | None = None,
        required_ppe: list[str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compliance_threshold: float = DEFAULT_COMPLIANCE_THRESHOLD,
        temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
    ) -> dict[str, Any]:
        self._validate_image(input_path)
        runtime = self._get_runtime(model_name or self.default_model())
        class_names = self._class_names(runtime.model)
        selected_ppe = self._normalize_required_ppe(required_ppe, class_names)

        frame = cv2.imread(str(input_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {input_path.name}")

        tracker = TemporalComplianceTracker(window_size=temporal_window, threshold=compliance_threshold)
        frame_result = self._analyze_frame(
            runtime=runtime,
            frame=frame,
            frame_index=0,
            confidence_threshold=confidence_threshold,
            required_ppe=selected_ppe,
            compliance_threshold=compliance_threshold,
            tracker=tracker,
            box_tracker=None,
            identity_resolver=None,
            presence_smoother=None,
            use_tracking=False,
        )

        run_dir = self._prepare_run_dir(prefix="image", source_name=input_path.name)
        annotated_path = run_dir / f"{_safe_stem(input_path.name)}_annotated.jpg"
        summary_path = run_dir / "report.json"
        cv2.imwrite(str(annotated_path), frame_result["annotated_frame"])

        payload = {
            "mode": "image",
            "input_name": input_path.name,
            "selected_model": model_name or self.default_model(),
            "required_ppe": selected_ppe,
            "summary": frame_result["summary"],
            "people": frame_result["people"],
            "detections": frame_result["detections"],
            "annotated_media_url": _as_relative_url(annotated_path),
            "report_url": _as_relative_url(summary_path),
        }

        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def analyze_video(
        self,
        input_path: Path,
        model_name: str | None = None,
        required_ppe: list[str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compliance_threshold: float = DEFAULT_COMPLIANCE_THRESHOLD,
        temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
    ) -> dict[str, Any]:
        self._validate_video(input_path)
        runtime = self._get_runtime(model_name or self.default_model())
        class_names = self._class_names(runtime.model)
        selected_ppe = self._normalize_required_ppe(required_ppe, class_names)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {input_path.name}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        run_dir = self._prepare_run_dir(prefix="video", source_name=input_path.name)
        writer, video_path = self._create_video_writer(run_dir, width, height, fps, input_path.name)
        report_path = run_dir / "report.json"
        events_csv_path = run_dir / "events.csv"
        preview_dir = run_dir / "preview_frames"
        preview_dir.mkdir(parents=True, exist_ok=True)

        tracker = TemporalComplianceTracker(window_size=temporal_window, threshold=compliance_threshold)
        box_tracker = SimpleBoxTracker()
        identity_resolver = WorkerIdentityResolver(
            max_gap_frames=max(int(round(fps * 3.0)), temporal_window * 2),
        )
        presence_smoother = PPEPresenceSmoother(persistence_frames=temporal_window)
        timeline: list[dict[str, Any]] = []
        open_events: dict[str, dict[str, Any]] = {}
        final_events: list[dict[str, Any]] = []
        track_rollups: dict[str, dict[str, Any]] = {}
        preview_frames: list[str] = []

        total_people = 0
        status_counts = Counter()
        adaptive_scores: list[float] = []
        entropies: list[float] = []
        frame_index = 0
        sample_stride = max(1, int(round(fps)))
        preview_stride = max(1, math.ceil(frame_count / MAX_BROWSER_PREVIEW_FRAMES)) if frame_count else 1
        peak_violations = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_result = self._analyze_frame(
                runtime=runtime,
                frame=frame,
                frame_index=frame_index,
                confidence_threshold=confidence_threshold,
                required_ppe=selected_ppe,
                compliance_threshold=compliance_threshold,
                tracker=tracker,
                box_tracker=box_tracker,
                identity_resolver=identity_resolver,
                presence_smoother=presence_smoother,
                use_tracking=True,
            )

            writer.write(frame_result["annotated_frame"])

            if frame_index % preview_stride == 0:
                preview_path = preview_dir / f"frame_{len(preview_frames):04d}.jpg"
                preview_frame = prepare_preview_frame(frame_result["annotated_frame"])
                cv2.imwrite(
                    str(preview_path),
                    preview_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY],
                )
                preview_frames.append(_as_relative_url(preview_path))

            people = frame_result["people"]
            total_people += len(people)
            frame_status = Counter(person["status_kind"] for person in people)
            status_counts.update(frame_status)
            peak_violations = max(peak_violations, frame_status.get("violation", 0))

            for person in people:
                adaptive_scores.append(person["adaptive_score"])
                entropies.append(person["entropy"])
                track_key = person["track_key"]

                rollup = track_rollups.setdefault(
                    track_key,
                    {
                        "track_key": track_key,
                        "track_label": person["track_label"],
                        "frames_seen": 0,
                        "status_counts": Counter(),
                        "score_sum": 0.0,
                        "source_tracks": set(),
                        "missing_frames": Counter(),
                        "current_missing_streaks": {},
                        "max_missing_streaks": Counter(),
                    },
                )
                rollup["frames_seen"] += 1
                rollup["status_counts"][person["status_kind"]] += 1
                rollup["score_sum"] += person["adaptive_score"]
                rollup["source_tracks"].add(person.get("raw_track_key", person["track_key"]))
                update_missing_evidence(rollup, selected_ppe, person["missing_now"])

                if person["status_kind"] == "violation":
                    event = open_events.setdefault(
                        track_key,
                        {
                            "track_key": track_key,
                            "track_label": person["track_label"],
                            "start_frame": frame_index,
                            "end_frame": frame_index,
                            "frame_count": 0,
                            "missing_frames": Counter(),
                            "current_missing_streaks": {},
                            "max_missing_streaks": Counter(),
                        },
                    )
                    event["end_frame"] = frame_index
                    event["frame_count"] += 1
                    update_missing_evidence(event, selected_ppe, person["missing_now"])
                elif track_key in open_events:
                    final_events.append(close_event(open_events.pop(track_key), fps, selected_ppe))

            active_keys = {person["track_key"] for person in people}
            stale_keys = [track_key for track_key in open_events if track_key not in active_keys]
            for track_key in stale_keys:
                final_events.append(close_event(open_events.pop(track_key), fps, selected_ppe))

            if frame_index % sample_stride == 0:
                timeline.append(
                    {
                        "time_seconds": _round(frame_index / fps),
                        "workers": len(people),
                        "compliant": frame_status.get("compliant", 0),
                        "at_risk": frame_status.get("warning", 0),
                        "violations": frame_status.get("violation", 0),
                        "mean_adaptive_score": _round(
                            np.mean([person["adaptive_score"] for person in people]) if people else 0.0
                        ),
                    }
                )

            tracker.prune(frame_index, max_idle_frames=temporal_window * 4)
            frame_index += 1

        capture.release()
        writer.release()

        for event in open_events.values():
            final_events.append(close_event(event, fps, selected_ppe))

        final_events.sort(key=lambda item: (item["start_time_seconds"], item["track_label"]))
        write_events_csv(events_csv_path, final_events)

        duration_seconds = _round(frame_index / fps if fps else 0.0)
        people_rollup = finalize_track_rollups(track_rollups)
        payload = {
            "mode": "video",
            "input_name": input_path.name,
            "selected_model": model_name or self.default_model(),
            "required_ppe": selected_ppe,
            "summary": {
                "frames_processed": frame_index,
                "source_frames": frame_count,
                "duration_seconds": duration_seconds,
                "fps": _round(fps, 2),
                "unique_workers_detected": len(people_rollup),
                "total_worker_observations": total_people,
                "average_workers_per_frame": _round(total_people / frame_index if frame_index else 0.0),
                "compliant_observations": int(status_counts.get("compliant", 0)),
                "at_risk_observations": int(status_counts.get("warning", 0)),
                "violation_observations": int(status_counts.get("violation", 0)),
                "observation_compliance_rate": _round(
                    status_counts.get("compliant", 0) / total_people if total_people else 0.0
                ),
                "peak_concurrent_violations": int(peak_violations),
                "mean_adaptive_score": _round(np.mean(adaptive_scores) if adaptive_scores else 0.0),
                "mean_entropy": _round(np.mean(entropies) if entropies else 0.0),
                "event_count": len(final_events),
            },
            "timeline": timeline,
            "events": final_events,
            "people": people_rollup,
            "annotated_media_url": _as_relative_url(video_path),
            "browser_preview": {
                "frame_urls": preview_frames,
                "fps": _round(fps / preview_stride if preview_stride else fps, 2),
                "sample_stride": int(preview_stride),
                "width": min(width, PREVIEW_MAX_WIDTH),
                "height": int(round(height * (min(width, PREVIEW_MAX_WIDTH) / width))) if width else height,
            },
            "report_url": _as_relative_url(report_path),
            "events_csv_url": _as_relative_url(events_csv_path),
        }

        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _detect(
        self,
        runtime: ModelRuntime,
        frame: np.ndarray,
        confidence_threshold: float,
        use_tracking: bool,
    ) -> list[Detection]:
        with runtime.lock:
            results = runtime.model.predict(
                frame,
                conf=confidence_threshold,
                imgsz=DEFAULT_IMAGE_SIZE,
                device=self._device,
                verbose=False,
            )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([None] * len(boxes))
        class_names = self._class_names(runtime.model)

        detections: list[Detection] = []
        for index, coords in enumerate(xyxy):
            class_id = int(classes[index])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_names[class_id],
                    confidence=float(confidences[index]),
                    box=_to_box(coords, frame.shape),
                    track_id=int(track_ids[index]) if track_ids[index] is not None else None,
                )
            )
        return detections

    def _analyze_frame(
        self,
        runtime: ModelRuntime,
        frame: np.ndarray,
        frame_index: int,
        confidence_threshold: float,
        required_ppe: list[str],
        compliance_threshold: float,
        tracker: TemporalComplianceTracker,
        box_tracker: SimpleBoxTracker | None,
        identity_resolver: WorkerIdentityResolver | None,
        presence_smoother: PPEPresenceSmoother | None,
        use_tracking: bool,
    ) -> dict[str, Any]:
        detections = self._detect(runtime, frame, confidence_threshold, use_tracking)
        persons = [item for item in detections if item.class_name.lower() == "person"]
        ppe_items = [item for item in detections if item.class_name.lower() != "person"]
        visible_ppe_items = [item for item in ppe_items if item.class_name in required_ppe]

        if not persons and visible_ppe_items:
            persons = [infer_person_detection(visible_ppe_items, frame.shape)]

        if box_tracker is not None:
            persons = box_tracker.update(persons, frame)

        if identity_resolver is not None:
            identities = identity_resolver.assign(persons, frame, frame_index)
        else:
            identities = []

        if identities:
            deduped: dict[str, dict[str, Any]] = {}
            for index, person in enumerate(persons):
                if index >= len(identities):
                    continue
                identity = identities[index]
                candidate_quality = (
                    0 if person.inferred else 1,
                    float(person.confidence),
                    _area(person.box),
                )
                current = deduped.get(identity["track_key"])
                if current is None or candidate_quality > current["quality"]:
                    deduped[identity["track_key"]] = {
                        "person": person,
                        "identity": identity,
                        "quality": candidate_quality,
                        "order": index,
                    }

            ordered = sorted(deduped.values(), key=lambda item: item["order"])
            persons = [item["person"] for item in ordered]
            identities = [item["identity"] for item in ordered]

        assigned_items: dict[int, dict[str, Detection]] = {index: {} for index in range(len(persons))}
        for item in sorted(visible_ppe_items, key=lambda det: det.confidence, reverse=True):
            best_score = 0.0
            best_person_index: int | None = None
            for person_index, person in enumerate(persons):
                score = assignment_score(person.box, item.box, item.class_name, frame.shape)
                if score > best_score:
                    best_score = score
                    best_person_index = person_index

            if best_person_index is None or best_score < 0.16:
                continue

            existing = assigned_items[best_person_index].get(item.class_name)
            if existing is None or item.confidence > existing.confidence:
                assigned_items[best_person_index][item.class_name] = item

        people: list[dict[str, Any]] = []
        detected_class_counter = Counter(item.class_name for item in visible_ppe_items)

        for person_index, person in enumerate(persons, start=1):
            person_items = assigned_items[person_index - 1]
            identity = identities[person_index - 1] if person_index - 1 < len(identities) else None
            raw_track_key = str(person.track_id) if person.track_id is not None else f"frame-{frame_index}-person-{person_index}"
            track_key = identity["track_key"] if identity is not None else str(raw_track_key)
            track_label = (
                identity["track_label"]
                if identity is not None
                else (f"ID {person.track_id}" if person.track_id is not None else f"P{person_index}")
            )
            stable_track_id = identity["track_id"] if identity is not None else person.track_id

            if presence_smoother is not None:
                presence = presence_smoother.resolve(track_key, person_items, required_ppe, frame_index)
                score_map = presence["effective_scores"]
                missing_now = presence["missing_now"]
                supported_by_memory = presence["supported_by_memory"]
            else:
                score_map = {name: 1.0 if name in person_items else 0.0 for name in required_ppe}
                missing_now = [name for name in required_ppe if name not in person_items]
                supported_by_memory = []

            frame_score = float(np.mean(list(score_map.values()))) if score_map else 1.0
            temporal = tracker.update(track_key, frame_score, frame_index=frame_index)
            status = classify_status(missing_now, temporal["adaptive_score"], compliance_threshold)

            people.append(
                {
                    "person_index": person_index,
                    "track_key": track_key,
                    "track_label": track_label,
                    "track_id": stable_track_id,
                    "raw_track_key": raw_track_key,
                    "raw_track_id": person.track_id,
                    "inferred_person_box": person.inferred,
                    "box": list(person.box),
                    "status_label": status["label"],
                    "status_kind": status["kind"],
                    "found_now": sorted(person_items.keys()),
                    "missing_now": missing_now,
                    "supported_by_memory": supported_by_memory,
                    "confidence_by_ppe": {
                        name: _round(person_items[name].confidence) if name in person_items else 0.0
                        for name in required_ppe
                    },
                    "frame_score": temporal["frame_score"],
                    "adaptive_score": temporal["adaptive_score"],
                    "entropy": temporal["entropy"],
                    "trend": temporal["trend"],
                    "window_length": temporal["window_length"],
                    "adaptive_decay": temporal["decay"],
                }
            )

        summary = {
            "workers": len(people),
            "required_ppe": required_ppe,
            "compliant": sum(person["status_kind"] == "compliant" for person in people),
            "at_risk": sum(person["status_kind"] == "warning" for person in people),
            "violations": sum(person["status_kind"] == "violation" for person in people),
            "detected_ppe_counts": dict(sorted(detected_class_counter.items())),
            "mean_frame_score": _round(np.mean([person["frame_score"] for person in people]) if people else 0.0),
            "mean_adaptive_score": _round(np.mean([person["adaptive_score"] for person in people]) if people else 0.0),
            "mean_entropy": _round(np.mean([person["entropy"] for person in people]) if people else 0.0),
        }

        annotated = render_frame(frame.copy(), people, visible_ppe_items)
        return {
            "annotated_frame": annotated,
            "summary": summary,
            "people": people,
            "detections": [item.to_dict() for item in detections],
        }
