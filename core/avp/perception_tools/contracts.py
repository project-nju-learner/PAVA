from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import numpy as np


BoundaryKind = Literal[
    "video_start",
    "video_end",
    "interval_start",
    "interval_end",
    "adaptive_visual_change",
    "forced_max_duration",
]


@dataclass(frozen=True)
class DynamicPerceptionConfig:
    """Parameters for query-independent global and local visual segmentation."""

    global_sample_interval_sec: float = 1.0
    local_sample_interval_sec: float = 0.5
    context_radius: int = 2
    boundary_percentile: float = 82.0
    local_boundary_percentile: float = 75.0
    minimum_boundary_score: float = 0.35
    minimum_event_sec: float = 6.0
    maximum_event_sec: float = 45.0
    local_minimum_event_sec: float = 2.0
    local_branch_factor: int = 3
    semantic_weight: float = 0.75
    shot_weight: float = 0.25
    representative_frames: int = 3

    def __post_init__(self) -> None:
        positive = {
            "global_sample_interval_sec": self.global_sample_interval_sec,
            "local_sample_interval_sec": self.local_sample_interval_sec,
            "minimum_event_sec": self.minimum_event_sec,
            "maximum_event_sec": self.maximum_event_sec,
            "local_minimum_event_sec": self.local_minimum_event_sec,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.local_sample_interval_sec >= self.global_sample_interval_sec:
            raise ValueError("local sampling must be denser than global sampling")
        if self.maximum_event_sec < self.minimum_event_sec:
            raise ValueError("maximum_event_sec must be at least minimum_event_sec")
        if self.context_radius < 1:
            raise ValueError("context_radius must be at least 1")
        for name, value in (
            ("boundary_percentile", self.boundary_percentile),
            ("local_boundary_percentile", self.local_boundary_percentile),
        ):
            if not 0.0 <= float(value) <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")
        if not 0.0 <= float(self.minimum_boundary_score) <= 1.0:
            raise ValueError("minimum_boundary_score must be in [0, 1]")
        if self.semantic_weight < 0.0 or self.shot_weight < 0.0:
            raise ValueError("boundary weights cannot be negative")
        if not math.isclose(
            self.semantic_weight + self.shot_weight, 1.0, abs_tol=1e-6
        ):
            raise ValueError("semantic_weight + shot_weight must equal 1")
        if self.local_branch_factor < 2:
            raise ValueError("local_branch_factor must be at least 2")
        if self.representative_frames < 1:
            raise ValueError("representative_frames must be at least 1")


@dataclass(frozen=True)
class FrameSample:
    """One caller-owned RGB frame and its normalized visual embedding input."""

    timestamp_sec: float
    image_rgb: np.ndarray
    embedding: np.ndarray

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_sec)
        image = np.asarray(self.image_rgb)
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_sec must be finite and non-negative")
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            raise ValueError("image_rgb must have shape (height, width, at least 3)")
        if image.dtype != np.uint8:
            raise ValueError("image_rgb must use uint8 RGB values")
        if embedding.ndim != 1 or not embedding.size:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        if not np.isfinite(embedding).all() or float(np.linalg.norm(embedding)) <= 1e-12:
            raise ValueError("embedding must contain finite, non-zero values")
        object.__setattr__(self, "timestamp_sec", timestamp)
        object.__setattr__(self, "image_rgb", image)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True)
class BoundaryScore:
    timestamp_sec: float
    kind: BoundaryKind
    combined_score: float | None
    semantic_score: float | None
    shot_score: float | None
    locator_start_sec: float
    locator_end_sec: float

    @property
    def forced(self) -> bool:
        return self.kind == "forced_max_duration"

    def to_dict(self) -> dict[str, Any]:
        def optional_score(value: float | None) -> float | None:
            return None if value is None else round(float(value), 6)

        return {
            "timestamp_sec": round(float(self.timestamp_sec), 6),
            "kind": self.kind,
            "combined_score": optional_score(self.combined_score),
            "semantic_score": optional_score(self.semantic_score),
            "shot_score": optional_score(self.shot_score),
            "forced": self.forced,
            "locator": {
                "start_sec": round(float(self.locator_start_sec), 6),
                "end_sec": round(float(self.locator_end_sec), 6),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundaryScore":
        locator = payload.get("locator") or {}
        return cls(
            timestamp_sec=float(payload["timestamp_sec"]),
            kind=payload["kind"],
            combined_score=(
                None
                if payload.get("combined_score") is None
                else float(payload["combined_score"])
            ),
            semantic_score=(
                None
                if payload.get("semantic_score") is None
                else float(payload["semantic_score"])
            ),
            shot_score=(
                None
                if payload.get("shot_score") is None
                else float(payload["shot_score"])
            ),
            locator_start_sec=float(locator["start_sec"]),
            locator_end_sec=float(locator["end_sec"]),
        )


@dataclass(frozen=True)
class PerceptionEvent:
    event_index: int
    start_sec: float
    end_sec: float
    representative_timestamps: tuple[float, ...]
    sample_count: int
    embedding: tuple[float, ...]
    start_boundary: BoundaryScore
    end_boundary: BoundaryScore

    def __post_init__(self) -> None:
        if self.event_index < 0:
            raise ValueError("event_index must be non-negative")
        if self.sample_count < 1:
            raise ValueError("each event must contain at least one sampled frame")
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if (
            embedding.ndim != 1
            or not embedding.size
            or not np.isfinite(embedding).all()
            or float(np.linalg.norm(embedding)) <= 1e-12
        ):
            raise ValueError("event embedding must be finite and non-zero")
        if not 0.0 <= float(self.start_sec) < float(self.end_sec):
            raise ValueError("event must satisfy 0 <= start_sec < end_sec")
        if tuple(sorted(self.representative_timestamps)) != self.representative_timestamps:
            raise ValueError("representative_timestamps must be ordered")
        if any(
            timestamp < self.start_sec or timestamp >= self.end_sec
            for timestamp in self.representative_timestamps
        ):
            raise ValueError("representative timestamp lies outside its event")

    @property
    def event_id(self) -> str:
        return f"event_{self.event_index:06d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "start_sec": round(float(self.start_sec), 6),
            "end_sec": round(float(self.end_sec), 6),
            "representative_timestamps": [
                round(float(timestamp), 6)
                for timestamp in self.representative_timestamps
            ],
            "sample_count": self.sample_count,
            "embedding": [round(float(value), 8) for value in self.embedding],
            "start_boundary": self.start_boundary.to_dict(),
            "end_boundary": self.end_boundary.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PerceptionEvent":
        return cls(
            event_index=int(payload["event_index"]),
            start_sec=float(payload["start_sec"]),
            end_sec=float(payload["end_sec"]),
            representative_timestamps=tuple(
                float(value) for value in payload["representative_timestamps"]
            ),
            sample_count=int(payload["sample_count"]),
            embedding=tuple(float(value) for value in payload["embedding"]),
            start_boundary=BoundaryScore.from_dict(payload["start_boundary"]),
            end_boundary=BoundaryScore.from_dict(payload["end_boundary"]),
        )


@dataclass(frozen=True)
class PerceptionMap:
    """Ordered, query-independent navigation sidecar; never Evidence."""

    sampling_fps: float
    sample_timestamps: tuple[float, ...]
    events: tuple[PerceptionEvent, ...]
    coverage_start_sec: float
    coverage_end_sec: float
    coverage_complete_at_declared_cadence: bool
    coverage_violations: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = "perception_map_v2"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.sampling_fps)) or self.sampling_fps <= 0.0:
            raise ValueError("sampling_fps must be finite and positive")
        if tuple(sorted(self.sample_timestamps)) != self.sample_timestamps:
            raise ValueError("sample_timestamps must be ordered")
        for expected_index, event in enumerate(self.events):
            if event.event_index != expected_index:
                raise ValueError("events must have contiguous chronological indexes")
            if expected_index and not math.isclose(
                self.events[expected_index - 1].end_sec,
                event.start_sec,
                abs_tol=1e-6,
            ):
                raise ValueError("events must be contiguous and ordered")
        if not 0.0 <= self.coverage_start_sec <= self.coverage_end_sec:
            raise ValueError("coverage range is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sampling_fps": round(float(self.sampling_fps), 6),
            "sample_timestamps": [
                round(float(timestamp), 6) for timestamp in self.sample_timestamps
            ],
            "events": [event.to_dict() for event in self.events],
            "coverage": {
                "start_sec": round(float(self.coverage_start_sec), 6),
                "end_sec": round(float(self.coverage_end_sec), 6),
                "complete_at_declared_cadence": bool(
                    self.coverage_complete_at_declared_cadence
                ),
                "violations": list(self.coverage_violations),
            },
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PerceptionMap":
        if payload.get("schema_version") != "perception_map_v2":
            raise ValueError("event assist requires perception_map_v2")
        coverage = payload.get("coverage") or {}
        return cls(
            sampling_fps=float(payload["sampling_fps"]),
            sample_timestamps=tuple(
                float(value) for value in payload["sample_timestamps"]
            ),
            events=tuple(
                PerceptionEvent.from_dict(item) for item in payload["events"]
            ),
            coverage_start_sec=float(coverage["start_sec"]),
            coverage_end_sec=float(coverage["end_sec"]),
            coverage_complete_at_declared_cadence=bool(
                coverage["complete_at_declared_cadence"]
            ),
            coverage_violations=tuple(
                str(value) for value in coverage.get("violations", [])
            ),
            errors=tuple(str(value) for value in payload.get("errors", [])),
            schema_version=str(payload["schema_version"]),
        )


@dataclass(frozen=True)
class EventMatch:
    event_index: int
    start_sec: float
    end_sec: float
    event_score: float
    best_frame_score: float
    best_timestamp_sec: float
    representative_timestamps: tuple[float, ...]

    @property
    def event_id(self) -> str:
        return f"event_{self.event_index:06d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "start_sec": round(float(self.start_sec), 6),
            "end_sec": round(float(self.end_sec), 6),
            "event_score": round(float(self.event_score), 6),
            "best_frame_score": round(float(self.best_frame_score), 6),
            "best_timestamp_sec": round(float(self.best_timestamp_sec), 6),
            "representative_timestamps": [
                round(float(timestamp), 6)
                for timestamp in self.representative_timestamps
            ],
        }
