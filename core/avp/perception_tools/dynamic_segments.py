from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .contracts import (
    BoundaryScore,
    DynamicPerceptionConfig,
    FrameSample,
    PerceptionEvent,
    PerceptionMap,
)


def image_histogram(image_rgb: np.ndarray) -> np.ndarray:
    """Return a compact per-channel RGB histogram using only NumPy."""

    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        raise ValueError("image_rgb must have shape (height, width, at least 3)")
    array = array[:, :, :3]
    if array.shape[0] > 64:
        rows = np.linspace(0, array.shape[0] - 1, 64, dtype=np.int64)
        array = array[rows]
    if array.shape[1] > 64:
        columns = np.linspace(0, array.shape[1] - 1, 64, dtype=np.int64)
        array = array[:, columns]
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.max()) <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0.0, 255.0).astype(np.uint8, copy=False)
    channels: list[np.ndarray] = []
    for channel in range(3):
        histogram, _ = np.histogram(array[:, :, channel], bins=16, range=(0, 256))
        row = histogram.astype(np.float32)
        row /= max(1.0, float(row.sum()))
        channels.append(row)
    return np.concatenate(channels)


def semantic_change(embeddings: np.ndarray, radius: int) -> np.ndarray:
    embeddings = _normalized_rows(embeddings)
    count = len(embeddings)
    scores = np.zeros(count, dtype=np.float32)
    radius = max(1, int(radius))
    for index in range(1, count):
        left = embeddings[max(0, index - radius) : index].mean(axis=0)
        right = embeddings[index : min(count, index + radius)].mean(axis=0)
        left /= max(1e-12, float(np.linalg.norm(left)))
        right /= max(1e-12, float(np.linalg.norm(right)))
        scores[index] = max(0.0, 1.0 - float(left @ right))
    return scores


def shot_change(histograms: np.ndarray) -> np.ndarray:
    histograms = np.asarray(histograms, dtype=np.float32)
    if histograms.ndim != 2 or not len(histograms):
        raise ValueError("histograms must be a non-empty two-dimensional array")
    scores = np.zeros(len(histograms), dtype=np.float32)
    if len(histograms) > 1:
        scores[1:] = (
            0.5 * np.abs(histograms[1:] - histograms[:-1]).sum(axis=1) / 3.0
        )
    return scores


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty one-dimensional array")
    if len(values) <= 1:
        return np.zeros_like(values)
    scale = float(np.percentile(values[1:], 95.0))
    if scale <= 1e-12:
        scale = float(np.max(values[1:]))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def build_global_perception(
    duration_sec: float,
    frames: Sequence[FrameSample],
    config: DynamicPerceptionConfig | None = None,
) -> PerceptionMap:
    """Build a chronological navigation map from caller-supplied sampled frames."""

    config = config or DynamicPerceptionConfig()
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    ordered = _ordered_frames(frames, 0.0, duration)
    times, embeddings, histograms = _feature_arrays(ordered)
    semantic_scores, shot_scores, combined_scores = _change_scores(
        embeddings, histograms, config
    )
    threshold = _threshold(
        combined_scores,
        percentile=config.boundary_percentile,
        minimum=config.minimum_boundary_score,
    )
    selected, forced, errors = _select_global_boundaries(
        times,
        combined_scores,
        threshold,
        duration,
        config.minimum_event_sec,
        config.maximum_event_sec,
    )
    events = _build_ordered_events(
        0.0,
        duration,
        ordered,
        selected,
        forced,
        semantic_scores,
        shot_scores,
        combined_scores,
        config,
        global_scope=True,
    )
    coverage_start, coverage_end, coverage_complete, coverage_violations = _coverage(
        times, 0.0, duration, config.global_sample_interval_sec
    )
    return PerceptionMap(
        sampling_fps=1.0 / config.global_sample_interval_sec,
        sample_timestamps=tuple(float(item) for item in times),
        events=events,
        coverage_start_sec=coverage_start,
        coverage_end_sec=coverage_end,
        coverage_complete_at_declared_cadence=coverage_complete,
        coverage_violations=coverage_violations,
        errors=tuple(errors),
    )


def refine_interval(
    start_sec: float,
    end_sec: float,
    frames: Sequence[FrameSample],
    config: DynamicPerceptionConfig | None = None,
) -> tuple[PerceptionEvent, ...]:
    """Split one interval more densely without reading video or writing state."""

    config = config or DynamicPerceptionConfig()
    start = float(start_sec)
    end = float(end_sec)
    if not math.isfinite(start) or not math.isfinite(end) or not 0.0 <= start < end:
        raise ValueError("interval must satisfy 0 <= start_sec < end_sec")
    ordered = _ordered_frames(frames, start, end)
    times, embeddings, histograms = _feature_arrays(ordered)
    _, _, coverage_complete, coverage_violations = _coverage(
        times, start, end, config.local_sample_interval_sec
    )
    if not coverage_complete:
        raise ValueError(
            "local refinement requires complete declared cadence: "
            + ", ".join(coverage_violations)
        )
    semantic_scores, shot_scores, combined_scores = _change_scores(
        embeddings, histograms, config
    )
    threshold = _threshold(
        combined_scores,
        percentile=config.local_boundary_percentile,
        minimum=config.minimum_boundary_score,
    )
    candidates = [
        index
        for index in range(1, len(times))
        if float(combined_scores[index]) >= threshold
        and float(times[index]) - start >= config.local_minimum_event_sec
        and end - float(times[index]) >= config.local_minimum_event_sec
    ]
    selected: list[int] = []
    for index in sorted(
        candidates,
        key=lambda item: (-float(combined_scores[item]), float(times[item])),
    ):
        if all(
            abs(float(times[index] - times[other]))
            >= config.local_minimum_event_sec
            for other in selected
        ):
            selected.append(index)
            if len(selected) >= config.local_branch_factor - 1:
                break
    selected.sort()
    return _build_ordered_events(
        start,
        end,
        ordered,
        selected,
        set(),
        semantic_scores,
        shot_scores,
        combined_scores,
        config,
        global_scope=False,
    )


def _ordered_frames(
    frames: Sequence[FrameSample], start_sec: float, end_sec: float
) -> tuple[FrameSample, ...]:
    if not frames:
        raise ValueError("at least one frame sample is required")
    ordered = tuple(sorted(frames, key=lambda frame: frame.timestamp_sec))
    previous: float | None = None
    embedding_size: int | None = None
    for frame in ordered:
        if frame.timestamp_sec < start_sec or frame.timestamp_sec >= end_sec:
            raise ValueError("frame timestamp lies outside the requested range")
        if previous is not None and math.isclose(
            frame.timestamp_sec, previous, abs_tol=1e-9
        ):
            raise ValueError("frame timestamps must be unique")
        if embedding_size is None:
            embedding_size = int(frame.embedding.size)
        elif int(frame.embedding.size) != embedding_size:
            raise ValueError("all frame embeddings must have the same dimension")
        previous = frame.timestamp_sec
    return ordered


def _feature_arrays(
    frames: Sequence[FrameSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([frame.timestamp_sec for frame in frames], dtype=np.float64)
    embeddings = _normalized_rows(
        np.stack([frame.embedding for frame in frames]).astype(np.float32)
    )
    histograms = np.stack([image_histogram(frame.image_rgb) for frame in frames])
    return times, embeddings, histograms


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not array.shape[1]:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool(np.any(norms <= 1e-12)):
        raise ValueError("embeddings must not contain zero vectors")
    return (array / norms).astype(np.float32)


def _change_scores(
    embeddings: np.ndarray,
    histograms: np.ndarray,
    config: DynamicPerceptionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic_scores = robust_normalize(
        semantic_change(embeddings, config.context_radius)
    )
    shot_scores = robust_normalize(shot_change(histograms))
    combined_scores = (
        config.semantic_weight * semantic_scores
        + config.shot_weight * shot_scores
    ).astype(np.float32)
    return semantic_scores, shot_scores, combined_scores


def _threshold(values: np.ndarray, percentile: float, minimum: float) -> float:
    adaptive = float(np.percentile(values[1:], percentile)) if len(values) > 1 else 1.0
    return max(float(minimum), adaptive)


def _select_global_boundaries(
    times: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    duration_sec: float,
    minimum_event_sec: float,
    maximum_event_sec: float,
) -> tuple[list[int], set[int], list[str]]:
    candidates = [
        index
        for index in range(1, len(times))
        if float(scores[index]) >= threshold
        and float(times[index]) >= minimum_event_sec
        and duration_sec - float(times[index]) >= minimum_event_sec
    ]
    selected: list[int] = []
    for index in sorted(
        candidates, key=lambda item: (-float(scores[item]), float(times[item]))
    ):
        if all(
            abs(float(times[index] - times[other])) >= minimum_event_sec
            for other in selected
        ):
            selected.append(index)
    selected.sort()
    forced: set[int] = set()
    errors: list[str] = []
    while True:
        changed = False
        boundary_times = [0.0] + [float(times[index]) for index in selected] + [duration_sec]
        for left, right in zip(boundary_times, boundary_times[1:]):
            if right - left <= maximum_event_sec + 1e-6:
                continue
            low = left + minimum_event_sec
            high = min(left + maximum_event_sec, right - minimum_event_sec)
            eligible = [
                index
                for index, second in enumerate(times)
                if low <= float(second) <= high and index not in selected
            ]
            if not eligible:
                errors.append(
                    "unable_to_enforce_maximum_event_duration_due_to_sparse_samples"
                )
                return selected, forced, errors
            best = max(
                eligible,
                key=lambda item: (float(scores[item]), float(times[item])),
            )
            selected.append(best)
            selected.sort()
            forced.add(best)
            changed = True
            break
        if not changed:
            return selected, forced, errors


def _build_ordered_events(
    start_sec: float,
    end_sec: float,
    frames: Sequence[FrameSample],
    selected_indexes: Sequence[int],
    forced_indexes: set[int],
    semantic_scores: np.ndarray,
    shot_scores: np.ndarray,
    combined_scores: np.ndarray,
    config: DynamicPerceptionConfig,
    *,
    global_scope: bool,
) -> tuple[PerceptionEvent, ...]:
    times = np.asarray([frame.timestamp_sec for frame in frames], dtype=np.float64)
    boundary_times = [start_sec]
    boundary_times.extend(float(times[index]) for index in selected_indexes)
    boundary_times.append(end_sec)
    index_by_time = {float(times[index]): index for index in selected_indexes}

    def boundary(timestamp: float, *, is_start: bool, is_end: bool) -> BoundaryScore:
        index = index_by_time.get(float(timestamp))
        if index is None:
            if is_start:
                kind = "video_start" if global_scope else "interval_start"
            elif is_end:
                kind = "video_end" if global_scope else "interval_end"
            else:
                raise RuntimeError("internal boundary is missing its source sample")
            return BoundaryScore(
                timestamp,
                kind,
                None,
                None,
                None,
                timestamp,
                timestamp,
            )
        kind = (
            "forced_max_duration"
            if index in forced_indexes
            else "adaptive_visual_change"
        )
        locator_start = (
            float(times[max(0, index - 1)])
            if kind == "adaptive_visual_change"
            else timestamp
        )
        return BoundaryScore(
            timestamp,
            kind,
            float(combined_scores[index]),
            float(semantic_scores[index]),
            float(shot_scores[index]),
            locator_start,
            timestamp,
        )

    events: list[PerceptionEvent] = []
    for event_index, (left, right) in enumerate(zip(boundary_times, boundary_times[1:])):
        in_event = [
            float(timestamp)
            for timestamp in times
            if left <= float(timestamp) < right
        ]
        if not in_event:
            raise ValueError("a generated event contains no sampled frame")
        member_indexes = [
            index
            for index, timestamp in enumerate(times)
            if left <= float(timestamp) < right
        ]
        member_embeddings = np.stack(
            [
                frame.embedding
                for frame in frames
                if left <= frame.timestamp_sec < right
            ]
        ).astype(np.float32)
        member_embeddings = _normalized_rows(member_embeddings)
        member_change_scores = np.asarray(
            [combined_scores[index] for index in member_indexes],
            dtype=np.float32,
        )
        representatives = _representatives(
            in_event,
            member_embeddings,
            member_change_scores,
            event_duration=float(right - left),
            config=config,
        )
        event_embedding = member_embeddings.mean(axis=0)
        event_embedding /= max(1e-12, float(np.linalg.norm(event_embedding)))
        events.append(
            PerceptionEvent(
                event_index=event_index,
                start_sec=float(left),
                end_sec=float(right),
                representative_timestamps=representatives,
                sample_count=len(in_event),
                embedding=tuple(float(value) for value in event_embedding),
                start_boundary=boundary(
                    float(left),
                    is_start=event_index == 0,
                    is_end=False,
                ),
                end_boundary=boundary(
                    float(right),
                    is_start=False,
                    is_end=event_index == len(boundary_times) - 2,
                ),
            )
        )
    return tuple(events)


def _representatives(
    timestamps: Sequence[float],
    embeddings: np.ndarray,
    change_scores: np.ndarray,
    *,
    event_duration: float,
    config: DynamicPerceptionConfig,
) -> tuple[float, ...]:
    """Choose one stable medoid plus at most two diverse local representatives."""

    if not timestamps:
        return ()
    vectors = _normalized_rows(embeddings)
    if len(vectors) != len(timestamps) or len(change_scores) != len(timestamps):
        raise ValueError("representative inputs must have matching lengths")

    maximum = min(int(config.representative_frames), len(timestamps), 3)
    if maximum <= 1:
        count = 1
    else:
        duration_fraction = min(
            1.0,
            max(0.0, float(event_duration)) / float(config.maximum_event_sec),
        )
        duration_count = 1 + int(duration_fraction > (1.0 / 3.0)) + int(
            duration_fraction > (2.0 / 3.0)
        )
        internal_change = float(np.max(change_scores)) if len(change_scores) else 0.0
        high_change = (1.0 + float(config.minimum_boundary_score)) / 2.0
        change_count = 1 + int(internal_change >= config.minimum_boundary_score)
        change_count += int(internal_change >= high_change)
        count = min(maximum, max(duration_count, change_count))

    mean = vectors.mean(axis=0)
    mean /= max(1e-12, float(np.linalg.norm(mean)))
    similarity_to_mean = vectors @ mean
    selected = [
        min(
            range(len(timestamps)),
            key=lambda index: (
                -float(similarity_to_mean[index]),
                float(timestamps[index]),
            ),
        )
    ]
    while len(selected) < count:
        remaining = [index for index in range(len(timestamps)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(1.0 - float(vectors[index] @ vectors[other]) for other in selected),
                -float(timestamps[index]),
            ),
        )
        selected.append(next_index)
    selected.sort(key=lambda index: float(timestamps[index]))
    return tuple(float(timestamps[index]) for index in selected)


def _coverage(
    times: np.ndarray,
    requested_start: float,
    requested_end: float,
    expected_interval: float,
) -> tuple[float, float, bool, tuple[str, ...]]:
    tolerance = 0.001
    limit = expected_interval + tolerance
    leading_gap = float(times[0]) - requested_start
    trailing_gap = requested_end - float(times[-1])
    violations: list[str] = []
    if leading_gap > limit:
        violations.append("leading_gap_exceeds_declared_cadence")
    if trailing_gap > limit:
        violations.append("trailing_gap_exceeds_declared_cadence")
    for index, gap in enumerate(np.diff(times)):
        if float(gap) > limit:
            violations.append(f"internal_gap_after_sample_{index}")
    actual_start = requested_start if leading_gap <= limit else float(times[0])
    actual_end = requested_end if trailing_gap <= limit else float(times[-1])
    return actual_start, actual_end, not violations, tuple(violations)
