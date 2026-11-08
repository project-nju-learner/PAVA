"""Standalone perception utilities. This package is intentionally not wired into AVP."""

from .contracts import (
    BoundaryScore,
    DynamicPerceptionConfig,
    EventMatch,
    FrameSample,
    PerceptionEvent,
    PerceptionMap,
)
from .dynamic_segments import (
    build_global_perception,
    image_histogram,
    refine_interval,
    robust_normalize,
    semantic_change,
    shot_change,
)
from .openclip_backend import OpenClipBackend
from .event_ranking import rank_events
from .event_assist import (
    EventAssistRuntime,
    event_refs_for_ranges,
    select_ranked_event_batch,
)

__all__ = [
    "BoundaryScore",
    "DynamicPerceptionConfig",
    "EventAssistRuntime",
    "EventMatch",
    "FrameSample",
    "OpenClipBackend",
    "PerceptionEvent",
    "PerceptionMap",
    "build_global_perception",
    "event_refs_for_ranges",
    "image_histogram",
    "rank_events",
    "refine_interval",
    "robust_normalize",
    "semantic_change",
    "select_ranked_event_batch",
    "shot_change",
]
