from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class QueryFramePick:
    frame_index: int
    timestamp_sec: float
    event_index: int
    query_index: int
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 6),
            "event_index": self.event_index,
            "query_index": self.query_index,
            "score": round(self.score, 6),
        }


def select_query_frames(
    *,
    timestamps: Sequence[float],
    frame_embeddings: np.ndarray,
    event_indexes: Sequence[int],
    query_embeddings: np.ndarray,
    frame_budget: int = 32,
    minimum_time_gap_sec: float = 2.0,
) -> tuple[QueryFramePick, ...]:
    """Score every sampled frame, then allocate a diverse budget across queries."""

    times = np.asarray(timestamps, dtype=np.float64)
    frames = _normalized_rows(frame_embeddings, "frame_embeddings")
    queries = _normalized_rows(query_embeddings, "query_embeddings")
    events = np.asarray(event_indexes, dtype=np.int64)
    if len(times) != len(frames) or len(events) != len(frames):
        raise ValueError("timestamps, frame_embeddings, and event_indexes must align")
    if not len(frames):
        raise ValueError("at least one frame is required")
    if frames.shape[1] != queries.shape[1]:
        raise ValueError("frame and query embedding dimensions must match")
    if tuple(sorted(float(value) for value in times)) != tuple(
        float(value) for value in times
    ):
        raise ValueError("timestamps must be chronological")
    if not np.isfinite(times).all() or bool(np.any(times < 0.0)):
        raise ValueError("timestamps must be finite and non-negative")
    if frame_budget < 1:
        raise ValueError("frame_budget must be positive")
    if not math.isfinite(float(minimum_time_gap_sec)) or minimum_time_gap_sec < 0.0:
        raise ValueError("minimum_time_gap_sec must be finite and non-negative")
    budget = min(int(frame_budget), len(frames))
    scores = frames @ queries.T
    query_count = len(queries)
    quotas = [budget // query_count] * query_count
    for index in range(budget % query_count):
        quotas[index] += 1

    ranked = [
        sorted(
            range(len(frames)),
            key=lambda index: (-float(scores[index, query_index]), index),
        )
        for query_index in range(query_count)
    ]
    selected: list[QueryFramePick] = []
    selected_indexes: set[int] = set()
    selected_query_events: set[tuple[int, int]] = set()

    def eligible(frame_index: int, query_index: int) -> bool:
        if frame_index in selected_indexes:
            return False
        event_index = int(events[frame_index])
        if (query_index, event_index) in selected_query_events:
            return False
        timestamp = float(times[frame_index])
        return all(
            abs(timestamp - pick.timestamp_sec) >= minimum_time_gap_sec
            for pick in selected
        )

    def take(frame_index: int, query_index: int) -> None:
        event_index = int(events[frame_index])
        selected_indexes.add(frame_index)
        selected_query_events.add((query_index, event_index))
        selected.append(
            QueryFramePick(
                frame_index=frame_index,
                timestamp_sec=float(times[frame_index]),
                event_index=event_index,
                query_index=query_index,
                score=float(scores[frame_index, query_index]),
            )
        )

    for query_index, quota in enumerate(quotas):
        used_events: set[int] = set()
        for frame_index in ranked[query_index]:
            event_index = int(events[frame_index])
            if event_index in used_events or not eligible(frame_index, query_index):
                continue
            take(frame_index, query_index)
            used_events.add(event_index)
            if len(used_events) >= quota:
                break

    fill_candidates = sorted(
        (
            (float(scores[frame_index, query_index]), frame_index, query_index)
            for frame_index in range(len(frames))
            for query_index in range(query_count)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for _, frame_index, query_index in fill_candidates:
        if len(selected) >= budget:
            break
        if eligible(frame_index, query_index):
            take(frame_index, query_index)

    return tuple(sorted(selected, key=lambda pick: pick.timestamp_sec))


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool(np.any(norms <= 1e-12)):
        raise ValueError(f"{name} must not contain zero vectors")
    return (array / norms).astype(np.float32)
