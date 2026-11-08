"""Run an isolated recursive Event-tree perception experiment.

The script consumes frozen flat perception maps, builds a transient
query-independent hierarchy, and lets node-scoped visual observations request
direct-child refinement. It never calls Planner, Verifier, or Answer.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from avp.perception_tools.contracts import PerceptionMap
from avp.perception_tools.hierarchical_event_tree import (
    MIN_NODE_FRAMES,
    HierarchicalEventTree,
    RefinementChoice,
    ScheduledTreeFrame,
    build_transient_event_tree,
    normalize_refinement_requests,
    schedule_node_frames,
    select_embedding_fallback,
)
from experiments.pollution_audit import write_json
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    DashScopeFlashTransport,
)
from scripts.run_perception_shadow import (
    decode_scheduled_frames,
    load_inference_input,
)
from scripts.run_query_driven_frame_shadow import (
    MODEL,
    answer_from_facts,
    call_json,
    encode_labeled_jpeg,
    option_block,
    response_format,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodeObservationFailure(RuntimeError):
    """Carry a failed provider trace back across the worker boundary."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        super().__init__(str(failure.get("error") or "node_observation_failed"))
        self.failure = dict(failure)


MAX_INITIAL_OVERVIEW_FRAMES = 200
MAX_GLOBAL_COVERAGE_PACKETS = 8

FRAME_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])F\d+(?![A-Za-z0-9_])")


def normalize_observation_payload(
    payload: Mapping[str, Any],
    *,
    frame_ids: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize only the provider's known string-fact schema deviation."""
    normalized = dict(payload)
    facts = normalized.get("facts")
    if not isinstance(facts, list):
        return normalized, []

    allowed = set(frame_ids)
    normalized_facts: list[Any] = []
    warnings: list[str] = []
    recovered_string_count = 0
    for index, raw in enumerate(facts):
        if not isinstance(raw, str):
            normalized_facts.append(raw)
            continue
        text = raw.strip()
        cited: list[str] = []
        for match in FRAME_ID_PATTERN.finditer(text):
            frame_id = match.group(0)
            if frame_id in allowed and frame_id not in cited:
                cited.append(frame_id)
        if not text or not cited or len(cited) > 4:
            warnings.append(
                f"fact_{index}:string_fact_missing_valid_frame_binding"
            )
            continue
        normalized_facts.append(
            {
                "frame_ids": cited,
                "fact": text,
                "_normalized_from_string_fact": True,
            }
        )
        recovered_string_count += 1

    if (
        facts
        and all(isinstance(raw, str) for raw in facts)
        and recovered_string_count == 0
    ):
        raise ValueError("node_observation_string_facts_not_recoverable")
    normalized["facts"] = normalized_facts
    return normalized, warnings


def _trace_is_empty_provider_observation(trace: Mapping[str, Any]) -> bool:
    raw = trace.get("raw_response")
    if not isinstance(raw, str):
        return False
    try:
        return json.loads(raw) == []
    except json.JSONDecodeError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--perception-map", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--query-embeddings")
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument(
        "--initial-frame-budget",
        type=int,
        default=MAX_INITIAL_OVERVIEW_FRAMES,
    )
    parser.add_argument("--full-coverage-1fps", action="store_true")
    parser.add_argument(
        "--max-frames-per-packet",
        type=int,
        default=MAX_INITIAL_OVERVIEW_FRAMES,
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--retry-empty-packets",
        type=int,
        choices=(0, 1),
        default=0,
        help="Retry only literal empty-array coverage responses, per packet.",
    )
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.branch_factor <= 4:
        raise ValueError("branch_factor_must_be_between_two_and_four")
    if not 4 <= args.initial_frame_budget <= MAX_INITIAL_OVERVIEW_FRAMES:
        raise ValueError("initial_frame_budget_must_be_between_four_and_200")
    if not 4 <= args.max_frames_per_packet <= MAX_INITIAL_OVERVIEW_FRAMES:
        raise ValueError("max_frames_per_packet_must_be_between_four_and_200")
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max_workers_must_be_between_one_and_eight")
    if args.max_edge < 64:
        raise ValueError("max_edge_must_be_at_least_64")
    if (
        not args.dry_run and not args.full_coverage_1fps and not args.query_embeddings
    ):
        raise ValueError("live_run_requires_query_embeddings")


def load_map_artifact(path: str | Path) -> tuple[str, str, float, PerceptionMap]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("perception_map_artifact_must_be_an_object")
    if payload.get("ground_truth_loaded") is not False:
        raise ValueError("perception_map_ground_truth_loaded_must_be_false")
    video_id = str(payload.get("video_id") or "").strip()
    video_path = str(payload.get("video_path") or "").strip()
    duration_sec = float(payload.get("duration_sec"))
    if not video_id or not Path(video_path).is_file():
        raise ValueError("perception_map_video_identity_is_invalid")
    perception_map = PerceptionMap.from_dict(payload["perception_map"])
    if not math.isclose(
        perception_map.coverage_end_sec, duration_sec, abs_tol=0.01
    ):
        raise ValueError("perception_map_duration_does_not_match_video")
    return video_id, video_path, duration_sec, perception_map


def frame_budget(perception_map: PerceptionMap) -> int:
    return sum(
        len(event.representative_timestamps) for event in perception_map.events
    )


def schedule_global_overview_frames(
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    *,
    max_frames: int = MAX_INITIAL_OVERVIEW_FRAMES,
) -> tuple[ScheduledTreeFrame, ...]:
    """Return one chronological root package from existing representatives."""
    if not 4 <= max_frames <= MAX_INITIAL_OVERVIEW_FRAMES:
        raise ValueError("global_overview_frame_count_outside_contract")
    candidates = []
    for event in perception_map.events:
        child_id = next(
            child_id
            for child_id in tree.root.child_ids
            if event.event_index in tree.node(child_id).source_event_indexes
        )
        for timestamp in event.representative_timestamps:
            candidates.append(
                ScheduledTreeFrame(
                    node_id=tree.root_id,
                    child_id=child_id,
                    source_event_id=event.event_id,
                    timestamp_sec=float(timestamp),
                )
            )
    candidates.sort(key=lambda frame: frame.timestamp_sec)
    if len(candidates) <= max_frames:
        return tuple(candidates)
    indexes = [
        round(index * (len(candidates) - 1) / (max_frames - 1))
        for index in range(max_frames)
    ]
    if len(set(indexes)) != max_frames:
        raise RuntimeError("global_overview_sampling_must_be_distinct")
    return tuple(candidates[index] for index in indexes)


def initial_schedule(
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    *,
    initial_frame_budget: int,
) -> tuple[dict[str, Any], ...]:
    frames = schedule_global_overview_frames(
        tree,
        perception_map,
        max_frames=min(initial_frame_budget, frame_budget(perception_map)),
    )
    return (
        {
            "node_id": tree.root_id,
            "start_sec": round(tree.root.start_sec, 6),
            "end_sec": round(tree.root.end_sec, 6),
            "source_event_count": len(tree.root.source_event_indexes),
            "frames": [frame.to_dict() for frame in frames],
        },
    )


def _sample_records(
    perception_map: PerceptionMap,
) -> tuple[tuple[float, int, str], ...]:
    if not math.isclose(perception_map.sampling_fps, 1.0, abs_tol=1e-6):
        raise ValueError("full_coverage_requires_one_fps_perception_map")
    if (
        not perception_map.coverage_complete_at_declared_cadence
        or perception_map.coverage_violations
        or perception_map.errors
    ):
        raise ValueError("full_coverage_requires_complete_perception_map")
    records = []
    event_index = 0
    for raw_timestamp in perception_map.sample_timestamps:
        timestamp = float(raw_timestamp)
        while (
            event_index + 1 < len(perception_map.events)
            and timestamp >= perception_map.events[event_index].end_sec
        ):
            event_index += 1
        event = perception_map.events[event_index]
        if not event.start_sec <= timestamp < event.end_sec:
            raise ValueError("sample_timestamp_outside_source_event")
        records.append((timestamp, event.event_index, event.event_id))
    if len({round(record[0], 6) for record in records}) != len(records):
        raise ValueError("full_coverage_sample_timestamps_must_be_distinct")
    return tuple(records)


def schedule_full_coverage_packets(
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    *,
    max_frames_per_packet: int = MAX_INITIAL_OVERVIEW_FRAMES,
) -> tuple[dict[str, Any], ...]:
    """Partition every 1 fps sample on Event-tree boundaries."""
    if not 4 <= max_frames_per_packet <= MAX_INITIAL_OVERVIEW_FRAMES:
        raise ValueError("max_frames_per_packet_outside_contract")
    records = _sample_records(perception_map)
    if len(records) < MIN_NODE_FRAMES:
        raise ValueError("full_coverage_has_fewer_than_four_frames")

    records_by_event: dict[int, list[tuple[float, int, str]]] = {}
    for record in records:
        records_by_event.setdefault(record[1], []).append(record)

    raw_packets: list[dict[str, Any]] = []

    def collect(node_id: str) -> None:
        node = tree.node(node_id)
        node_records = [
            record
            for event_index in node.source_event_indexes
            for record in records_by_event.get(event_index, [])
        ]
        node_records.sort(key=lambda record: record[0])
        if len(node_records) <= max_frames_per_packet:
            raw_packets.append(
                {
                    "node_ids": [node.node_id],
                    "source_event_indexes": list(node.source_event_indexes),
                    "records": node_records,
                }
            )
            return
        if node.is_leaf:
            raise ValueError("single_event_exceeds_packet_frame_limit")
        for child_id in node.child_ids:
            collect(child_id)

    for child_id in tree.root.child_ids:
        collect(child_id)

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(raw_packets):
        packet = raw_packets[index]
        if (
            len(packet["records"]) < MIN_NODE_FRAMES
            and index + 1 < len(raw_packets)
            and len(packet["records"]) + len(raw_packets[index + 1]["records"])
            <= max_frames_per_packet
        ):
            following = raw_packets[index + 1]
            packet = {
                "node_ids": packet["node_ids"] + following["node_ids"],
                "source_event_indexes": (
                    packet["source_event_indexes"]
                    + following["source_event_indexes"]
                ),
                "records": packet["records"] + following["records"],
            }
            index += 1
        elif (
            len(packet["records"]) < MIN_NODE_FRAMES
            and merged
            and len(merged[-1]["records"]) + len(packet["records"])
            <= max_frames_per_packet
        ):
            previous = merged.pop()
            packet = {
                "node_ids": previous["node_ids"] + packet["node_ids"],
                "source_event_indexes": (
                    previous["source_event_indexes"]
                    + packet["source_event_indexes"]
                ),
                "records": previous["records"] + packet["records"],
            }
        merged.append(packet)
        index += 1

    if any(
        not MIN_NODE_FRAMES <= len(packet["records"]) <= max_frames_per_packet
        for packet in merged
    ):
        raise ValueError("full_coverage_packet_size_outside_contract")
    if len(merged) > MAX_GLOBAL_COVERAGE_PACKETS:
        raise ValueError("global_coverage_packet_count_exceeds_eight")

    packets = []
    all_timestamps = []
    all_event_ids = []
    for ordinal, packet in enumerate(merged, 1):
        node_ids = tuple(packet["node_ids"])
        event_indexes = tuple(packet["source_event_indexes"])
        event_ids = tuple(
            perception_map.events[event_index].event_id
            for event_index in event_indexes
            if records_by_event.get(event_index)
        )
        frames = []
        for timestamp, _event_index, event_id in packet["records"]:
            root_child_id = next(
                child_id
                for child_id in tree.root.child_ids
                if _event_index in tree.node(child_id).source_event_indexes
            )
            frames.append(
                ScheduledTreeFrame(
                    node_id=node_ids[0] if len(node_ids) == 1 else tree.root_id,
                    child_id=root_child_id,
                    source_event_id=event_id,
                    timestamp_sec=timestamp,
                )
            )
        packet_id = f"P{ordinal:02d}"
        packets.append(
            {
                "packet_id": packet_id,
                "ordinal": ordinal,
                "node_ids": node_ids,
                "start_sec": frames[0].timestamp_sec,
                "end_sec": frames[-1].timestamp_sec,
                "event_ids": event_ids,
                "frames": tuple(frames),
            }
        )
        all_timestamps.extend(frame.timestamp_sec for frame in frames)
        all_event_ids.extend(event_ids)

    expected_timestamps = [record[0] for record in records]
    if all_timestamps != expected_timestamps:
        raise RuntimeError("full_coverage_packet_timeline_is_not_exact")
    expected_event_ids = [
        event.event_id
        for event in perception_map.events
        if records_by_event.get(event.event_index)
    ]
    if all_event_ids != expected_event_ids:
        raise RuntimeError("full_coverage_packet_events_are_not_exact")
    return tuple(packets)


def dry_run_record(
    item: Mapping[str, Any],
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    *,
    initial_frame_budget: int,
) -> dict[str, Any]:
    budget = frame_budget(perception_map)
    initial = initial_schedule(
        tree,
        perception_map,
        initial_frame_budget=initial_frame_budget,
    )
    return {
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "ground_truth_loaded": False,
        "tree": tree.to_dict(),
        "initial_frontier": list(initial),
        "frame_budget": budget,
        "initial_frame_count": sum(len(node["frames"]) for node in initial),
        "maximum_request_count": 1
        + max(
            0,
            budget - sum(len(node["frames"]) for node in initial),
        )
        // MIN_NODE_FRAMES,
        "maximum_concurrent_requests": 8,
    }



def coverage_observation_format(
    frame_ids: Sequence[str], event_ids: Sequence[str]
) -> dict[str, Any]:
    return response_format(
        "hierarchical_event_coverage_packet_v1",
        {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": len(event_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "OBSERVED",
                                    "NO_RELEVANT_FACT",
                                    "UNCERTAIN",
                                ],
                            },
                            "facts": {
                                "type": "array",
                                "maxItems": 2,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "frame_ids": {
                                            "type": "array",
                                            "minItems": 1,
                                            "maxItems": 4,
                                            "items": {
                                                "type": "string",
                                                "minLength": 1,
                                            },
                                        },
                                        "fact": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                    "required": ["frame_ids", "fact"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["event_id", "status", "facts"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["events"],
            "additionalProperties": False,
        },
    )


def _coverage_manifests(
    packet: Mapping[str, Any],
    perception_map: PerceptionMap,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    frames = packet["frames"]
    frame_manifest = []
    frames_by_event: dict[str, list[dict[str, Any]]] = {
        event_id: [] for event_id in packet["event_ids"]
    }
    frame_ids = []
    for index, frame in enumerate(frames, 1):
        frame_id = f"{packet['packet_id']}F{index:03d}"
        frame_ids.append(frame_id)
        item = frame.to_dict()
        item["frame_id"] = frame_id
        frame_manifest.append(item)
        frames_by_event[frame.source_event_id].append(
            {
                "frame_id": frame_id,
                "timestamp_sec": round(frame.timestamp_sec, 6),
            }
        )
    events_by_id = {event.event_id: event for event in perception_map.events}
    event_manifest = []
    for event_id in packet["event_ids"]:
        event = events_by_id[event_id]
        event_manifest.append(
            {
                "event_id": event_id,
                "start_sec": round(event.start_sec, 6),
                "end_sec": round(event.end_sec, 6),
                "frames": frames_by_event[event_id],
            }
        )
    return frame_manifest, event_manifest, frame_ids


def validate_coverage_payload(
    payload: Mapping[str, Any],
    *,
    event_ids: Sequence[str],
    frame_manifest: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("coverage_packet_events_must_be_an_array")
    expected = tuple(event_ids)
    by_event: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError(f"coverage_event_{index}:not_an_object")
        event_id = str(raw.get("event_id") or "")
        if event_id not in expected:
            raise ValueError(f"coverage_event_{index}:unknown_event_id")
        if event_id in by_event:
            raise ValueError(f"coverage_event_{index}:duplicate_event_id")
        status = str(raw.get("status") or "")
        if status not in {"OBSERVED", "NO_RELEVANT_FACT", "UNCERTAIN"}:
            raise ValueError(f"coverage_event_{index}:invalid_status")
        facts = raw.get("facts")
        if not isinstance(facts, list):
            raise ValueError(f"coverage_event_{index}:invalid_facts")
        facts = facts[:2]
        if status == "OBSERVED" and not facts:
            raise ValueError(f"coverage_event_{index}:observed_without_fact")
        if status == "NO_RELEVANT_FACT" and facts:
            raise ValueError(f"coverage_event_{index}:no_relevant_fact_has_fact")
        allowed_frames = {
            str(frame["frame_id"]) for frame in frame_manifest
        }
        normalized_facts = []
        for fact_index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                raise ValueError(
                    f"coverage_event_{index}:fact_{fact_index}:not_an_object"
                )
            cited = fact.get("frame_ids")
            text = str(fact.get("fact") or "").strip()
            if (
                not isinstance(cited, list)
                or not 1 <= len(cited) <= 4
                or not text
                or any(str(frame_id) not in allowed_frames for frame_id in cited)
            ):
                raise ValueError(
                    f"coverage_event_{index}:fact_{fact_index}:invalid_frame_binding"
                )
            normalized_facts.append(
                {
                    "frame_ids": [str(frame_id) for frame_id in cited],
                    "fact": text,
                }
            )
        by_event[event_id] = {
            "event_id": event_id,
            "status": status,
            "facts": normalized_facts,
        }
    if set(by_event) != set(expected):
        missing = [event_id for event_id in expected if event_id not in by_event]
        raise ValueError("coverage_packet_missing_events:" + ",".join(missing))
    return tuple(by_event[event_id] for event_id in expected)


def observe_coverage_packet(
    *,
    request_index: int,
    base_url: str,
    item: Mapping[str, Any],
    perception_map: PerceptionMap,
    video_path: str,
    packet: Mapping[str, Any],
    max_edge: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    frames = packet["frames"]
    frame_manifest, event_manifest, frame_ids = _coverage_manifests(
        packet, perception_map
    )
    images, decoder = decode_scheduled_frames(
        video_path,
        [frame.timestamp_sec for frame in frames],
        max_edge=max_edge,
    )
    jpegs = [
        encode_labeled_jpeg(image, frame_id)
        for image, frame_id in zip(images, frame_ids)
    ]
    frame_uris = [
        "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        for jpeg in jpegs
    ]
    local_trace: list[dict[str, Any]] = []
    transport = DashScopeFlashTransport(base_url=base_url, generation_overrides={})
    try:
        parsed = call_json(
            transport,
            stage="HIERARCHICAL_EVENT_COVERAGE_OBSERVE",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a chronological visual observer, not an answerer. "
                        "Return one JSON object with the top-level key events; never "
                        "return a bare array. Inspect the entire frame package. Return "
                        "every listed Event "
                        "exactly once and in chronological order, even when it has no "
                        "answer-relevant fact. Synthesize at most two concise, directly "
                        "visible facts per Event instead of enumerating frames from the "
                        "start. Cite only frame IDs belonging to that Event. Use "
                        "OBSERVED when at least one grounded fact is visible, "
                        "NO_RELEVANT_FACT with an empty facts array when nothing "
                        "question-relevant is visible, and UNCERTAIN when the frames do "
                        "not support a reliable statement. Do not infer unseen gaps, "
                        "invent timestamps, or select an answer option."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Question: {item['question']}\n"
                                f"Official options:\n{option_block(item)}\n"
                                f"Coverage packet: {packet['packet_id']} "
                                f"[{packet['start_sec']:.6f}, "
                                f"{packet['end_sec']:.6f}]\n"
                                "Event-to-frame manifest:\n"
                                + json.dumps(
                                    event_manifest,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            ),
                        },
                        {"type": "video", "video": frame_uris, "fps": 1.0},
                    ],
                },
            ],
            schema=coverage_observation_format(
                frame_ids, packet["event_ids"]
            ),
            enable_thinking=False,
            temperature=temperature,
            seed=seed,
            max_tokens=4096,
            request_trace=local_trace,
            submitted_frame_count=len(frames),
            bare_array_key="events",
        )
        normalized_events = validate_coverage_payload(
            parsed,
            event_ids=packet["event_ids"],
            frame_manifest=frame_manifest,
        )
    except Exception as exc:
        trace = (
            dict(local_trace[0])
            if local_trace
            else {
                "stage": "HIERARCHICAL_EVENT_COVERAGE_OBSERVE",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        trace["request_index"] = request_index
        if trace.get("status") != "FAILED":
            trace["provider_status"] = trace.get("status")
        trace["status"] = "FAILED"
        trace["error_type"] = type(exc).__name__
        trace["error"] = str(exc)
        raise NodeObservationFailure(
            {
                "request_index": request_index,
                "packet_id": packet["packet_id"],
                "node_ids": list(packet["node_ids"]),
                "submitted_frames": frame_manifest,
                "decoder": decoder,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trace": trace,
            }
        ) from exc
    finally:
        transport.close()
    trace = local_trace[0]
    trace["request_index"] = request_index
    return {
        "request_index": request_index,
        "packet_id": packet["packet_id"],
        "ordinal": packet["ordinal"],
        "node_ids": list(packet["node_ids"]),
        "start_sec": packet["start_sec"],
        "end_sec": packet["end_sec"],
        "expected_event_ids": list(packet["event_ids"]),
        "submitted_frames": frame_manifest,
        "decoder": decoder,
        "event_observations": list(normalized_events),
        "trace": trace,
    }


def ground_coverage_response(
    response: Mapping[str, Any],
    perception_map: PerceptionMap,
    *,
    fact_start_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_manifest = {
        str(frame["frame_id"]): frame for frame in response["submitted_frames"]
    }
    events_by_id = {event.event_id: event for event in perception_map.events}
    grounded_facts = []
    grounded_events = []
    for raw_event in response["event_observations"]:
        event_id = str(raw_event["event_id"])
        event = events_by_id[event_id]
        event_fact_ids = []
        for raw_fact in raw_event["facts"]:
            locators = [
                frame_manifest[str(frame_id)]
                for frame_id in raw_fact["frame_ids"]
            ]
            fact_id = f"E{fact_start_index + len(grounded_facts):04d}"
            grounded_facts.append(
                {
                    "fact_id": fact_id,
                    "fact": raw_fact["fact"],
                    "packet_id": response["packet_id"],
                    "event_id": event_id,
                    "frame_locators": locators,
                    "earliest_timestamp_sec": min(
                        float(locator["timestamp_sec"]) for locator in locators
                    ),
                }
            )
            event_fact_ids.append(fact_id)
        grounded_events.append(
            {
                "packet_id": response["packet_id"],
                "event_id": event_id,
                "start_sec": round(event.start_sec, 6),
                "end_sec": round(event.end_sec, 6),
                "status": raw_event["status"],
                "fact_ids": event_fact_ids,
            }
        )
    return grounded_facts, grounded_events

def observation_format(
    frame_ids: Sequence[str], refinement_target_ids: Sequence[str]
) -> dict[str, Any]:
    return response_format(
        "hierarchical_event_node_observation_v1",
        {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {
                                    "type": "string",
                                    "enum": list(frame_ids),
                                },
                            },
                            "fact": {"type": "string", "minLength": 1},
                        },
                        "required": ["frame_ids", "fact"],
                        "additionalProperties": False,
                    },
                },
                "refinement_requests": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_node_id": {
                                "type": "string",
                                "enum": list(refinement_target_ids),
                            },
                            "requested_frame_count": {
                                "type": "integer",
                                "minimum": 4,
                                "maximum": 16,
                            },
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "target_node_id",
                            "requested_frame_count",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["facts", "refinement_requests"],
            "additionalProperties": False,
        },
    )


def _frame_manifest(
    frames: Sequence[ScheduledTreeFrame],
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = []
    frame_ids = []
    for index, frame in enumerate(frames, 1):
        frame_id = f"F{index:02d}"
        frame_ids.append(frame_id)
        item = frame.to_dict()
        item["frame_id"] = frame_id
        manifest.append(item)
    return manifest, frame_ids


def observe_node(
    *,
    request_index: int,
    base_url: str,
    item: Mapping[str, Any],
    tree: HierarchicalEventTree,
    video_path: str,
    choice: RefinementChoice,
    frames: Sequence[ScheduledTreeFrame],
    max_edge: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    node = tree.node(choice.target_node_id)
    manifest, frame_ids = _frame_manifest(frames)
    images, decoder = decode_scheduled_frames(
        video_path,
        [frame.timestamp_sec for frame in frames],
        max_edge=max_edge,
    )
    jpegs = [
        encode_labeled_jpeg(image, frame_id)
        for image, frame_id in zip(images, frame_ids)
    ]
    frame_uris = [
        "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        for jpeg in jpegs
    ]
    allowed_targets = list(node.child_ids) if not node.is_leaf else [node.node_id]
    local_trace: list[dict[str, Any]] = []
    transport = DashScopeFlashTransport(base_url=base_url, generation_overrides={})
    normalization_warnings: list[str] = []
    try:
        parsed = call_json(
            transport,
            stage="HIERARCHICAL_EVENT_OBSERVE",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a node-scoped visual observer, not an answerer. "
                        "Report only facts directly visible in this frame package. "
                        "Cite printed frame IDs exactly. Do not infer unseen gaps, "
                        "timestamps, global counts, or the final option. Request "
                        "refinement only when a listed direct child or this leaf "
                        "could reveal answer-discriminating visible information."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Question: {item['question']}\n"
                                f"Official options:\n{option_block(item)}\n"
                                f"Current node: {node.node_id} "
                                f"[{node.start_sec:.6f}, {node.end_sec:.6f})\n"
                                f"Node type: {'leaf' if node.is_leaf else 'internal'}\n"
                                "Frame provenance manifest:\n"
                                + json.dumps(manifest, ensure_ascii=False, indent=2)
                                + "\nAllowed refinement targets:\n"
                                + json.dumps(allowed_targets, ensure_ascii=False)
                            ),
                        },
                        {"type": "video", "video": frame_uris, "fps": 1.0},
                    ],
                },
            ],
            schema=observation_format(frame_ids, allowed_targets),
            enable_thinking=False,
            temperature=temperature,
            seed=seed,
            max_tokens=1536,
            request_trace=local_trace,
            submitted_frame_count=len(frames),
        )
        parsed, normalization_warnings = normalize_observation_payload(
            parsed,
            frame_ids=frame_ids,
        )
    except Exception as exc:
        trace = (
            dict(local_trace[0])
            if local_trace
            else {
                "request_index": request_index,
                "stage": "HIERARCHICAL_EVENT_OBSERVE",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        trace["request_index"] = request_index
        if _trace_is_empty_provider_observation(trace):
            parsed = {"facts": [], "refinement_requests": []}
            normalization_warnings = ["empty_provider_observation"]
            trace["provider_parse_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            trace["status"] = "COMPLETED"
            trace.pop("error_type", None)
            trace.pop("error", None)
            local_trace[:] = [trace]
        else:
            if trace.get("status") != "FAILED":
                trace["provider_status"] = trace.get("status")
            trace["status"] = "FAILED"
            trace["error_type"] = type(exc).__name__
            trace["error"] = str(exc)
            raise NodeObservationFailure(
                {
                    "request_index": request_index,
                    "node_id": node.node_id,
                    "node_start_sec": node.start_sec,
                    "node_end_sec": node.end_sec,
                    "submitted_frames": manifest,
                    "decoder": decoder,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trace": trace,
                }
            ) from exc
    finally:
        transport.close()
    trace = local_trace[0]
    trace["request_index"] = request_index
    if normalization_warnings:
        trace["normalization_warnings"] = list(normalization_warnings)
    return {
        "request_index": request_index,
        "node_id": node.node_id,
        "node_start_sec": node.start_sec,
        "node_end_sec": node.end_sec,
        "node_is_leaf": node.is_leaf,
        "choice_reason": choice.reason,
        "submitted_frames": manifest,
        "decoder": decoder,
        "parsed": parsed,
        "trace": trace,
        "normalization_warnings": normalization_warnings,
    }


def grounded_facts(
    response: Mapping[str, Any],
    *,
    fact_start_index: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    parsed = response["parsed"]
    manifest = {
        str(frame["frame_id"]): frame for frame in response["submitted_frames"]
    }
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        raise ValueError("node_observation_facts_must_be_an_array")
    grounded = []
    errors = []
    for offset, raw in enumerate(facts):
        if not isinstance(raw, dict):
            errors.append(f"fact_{offset}:not_an_object")
            continue
        cited = raw.get("frame_ids")
        text = str(raw.get("fact") or "").strip()
        if (
            not isinstance(cited, list)
            or not cited
            or not text
            or any(str(frame_id) not in manifest for frame_id in cited)
        ):
            errors.append(f"fact_{offset}:invalid_frame_binding")
            continue
        locators = [manifest[str(frame_id)] for frame_id in cited]
        fact = {
            "fact_id": f"E{fact_start_index + len(grounded):04d}",
            "fact": text,
            "node_id": response["node_id"],
            "frame_locators": locators,
            "earliest_timestamp_sec": min(
                float(locator["timestamp_sec"]) for locator in locators
            ),
        }
        if raw.get("_normalized_from_string_fact") is True:
            fact["normalized_from_string_fact"] = True
        grounded.append(fact)
    return grounded, errors


def _deduplicate_choices(
    choices: Sequence[RefinementChoice],
) -> list[RefinementChoice]:
    result = []
    seen = set()
    for choice in choices:
        if choice.target_node_id not in seen:
            seen.add(choice.target_node_id)
            result.append(choice)
    return result



def full_coverage_dry_run_record(
    item: Mapping[str, Any],
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    *,
    max_frames_per_packet: int,
    empty_packet_retry_limit: int,
) -> dict[str, Any]:
    packets = schedule_full_coverage_packets(
        tree,
        perception_map,
        max_frames_per_packet=max_frames_per_packet,
    )
    return {
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "ground_truth_loaded": False,
        "tree": tree.to_dict(),
        "coverage_mode": "full_1fps_event_aligned_packets",
        "coverage_complete": True,
        "frame_budget": len(perception_map.sample_timestamps),
        "initial_frame_count": sum(len(packet["frames"]) for packet in packets),
        "empty_packet_retry_limit": empty_packet_retry_limit,
        "maximum_request_count": (
            len(packets) * (1 + empty_packet_retry_limit) + 1
        ),
        "maximum_concurrent_requests": min(
            MAX_GLOBAL_COVERAGE_PACKETS, len(packets)
        ),
        "initial_frontier": [
            {
                "packet_id": packet["packet_id"],
                "ordinal": packet["ordinal"],
                "node_ids": list(packet["node_ids"]),
                "start_sec": round(packet["start_sec"], 6),
                "end_sec": round(packet["end_sec"], 6),
                "event_ids": list(packet["event_ids"]),
                "frame_count": len(packet["frames"]),
                "frames": [frame.to_dict() for frame in packet["frames"]],
            }
            for packet in packets
        ],
    }


def run_full_coverage_question(
    *,
    base_url: str,
    item: Mapping[str, Any],
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    video_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    packets = schedule_full_coverage_packets(
        tree,
        perception_map,
        max_frames_per_packet=args.max_frames_per_packet,
    )
    responses = []
    initial_failures = []
    futures = {}
    with ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(packets))
    ) as executor:
        for packet in packets:
            future = executor.submit(
                observe_coverage_packet,
                request_index=packet["ordinal"],
                base_url=base_url,
                item=item,
                perception_map=perception_map,
                video_path=video_path,
                packet=packet,
                max_edge=args.max_edge,
                temperature=args.temperature,
                seed=args.seed,
            )
            futures[future] = packet
        for future in as_completed(futures):
            packet = futures[future]
            try:
                response = dict(future.result())
                response["attempt"] = 1
                responses.append(response)
            except NodeObservationFailure as exc:
                initial_failures.append(exc.failure)
            except Exception as exc:
                initial_failures.append(
                    {
                        "request_index": packet["ordinal"],
                        "packet_id": packet["packet_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    packets_by_id = {packet["packet_id"]: packet for packet in packets}
    attempt_failures = []
    terminal_failures = []
    retry_candidates = []
    for raw_failure in sorted(
        initial_failures,
        key=lambda failure: int(failure.get("request_index", 0)),
    ):
        failure = dict(raw_failure)
        failure["attempt"] = 1
        retry_eligible = (
            args.retry_empty_packets > 0
            and _trace_is_empty_provider_observation(failure.get("trace", {}))
            and failure.get("packet_id") in packets_by_id
        )
        failure["empty_packet_retry_eligible"] = retry_eligible
        attempt_failures.append(failure)
        if retry_eligible:
            retry_candidates.append((failure, packets_by_id[failure["packet_id"]]))
        else:
            terminal_failures.append(failure)

    retry_count = 0
    retry_futures = {}
    if retry_candidates:
        with ThreadPoolExecutor(
            max_workers=min(args.max_workers, len(retry_candidates))
        ) as executor:
            for retry_ordinal, (failure, packet) in enumerate(
                retry_candidates,
                start=1,
            ):
                request_index = len(packets) + retry_ordinal
                future = executor.submit(
                    observe_coverage_packet,
                    request_index=request_index,
                    base_url=base_url,
                    item=item,
                    perception_map=perception_map,
                    video_path=video_path,
                    packet=packet,
                    max_edge=args.max_edge,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                retry_futures[future] = (failure, packet, request_index)
                retry_count += 1
            for future in as_completed(retry_futures):
                original_failure, packet, request_index = retry_futures[future]
                try:
                    response = dict(future.result())
                    response["attempt"] = 2
                    response["retry_of_request_index"] = original_failure[
                        "request_index"
                    ]
                    responses.append(response)
                except NodeObservationFailure as exc:
                    retry_failure = dict(exc.failure)
                    retry_failure["attempt"] = 2
                    retry_failure["retry_of_request_index"] = original_failure[
                        "request_index"
                    ]
                    retry_failure["empty_packet_retry_eligible"] = (
                        _trace_is_empty_provider_observation(
                            retry_failure.get("trace", {})
                        )
                    )
                    attempt_failures.append(retry_failure)
                    terminal_failures.append(retry_failure)
                except Exception as exc:
                    retry_failure = {
                        "request_index": request_index,
                        "packet_id": packet["packet_id"],
                        "attempt": 2,
                        "retry_of_request_index": original_failure[
                            "request_index"
                        ],
                        "empty_packet_retry_eligible": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    attempt_failures.append(retry_failure)
                    terminal_failures.append(retry_failure)

    responses.sort(key=lambda response: response["ordinal"])
    attempt_failures.sort(
        key=lambda failure: int(failure.get("request_index", 0))
    )
    terminal_failures.sort(
        key=lambda failure: int(failure.get("request_index", 0))
    )

    expected_frame_count = len(perception_map.sample_timestamps)
    submitted_frame_count = sum(
        len(response["submitted_frames"]) for response in responses
    ) + sum(
        len(failure.get("submitted_frames", []))
        for failure in terminal_failures
    )
    attempted_submitted_frame_count = expected_frame_count + sum(
        len(packets_by_id[failure["packet_id"]]["frames"])
        for failure, _packet in retry_candidates
    )
    expected_event_ids = [
        event.event_id
        for event in perception_map.events
        if any(
            event.start_sec <= timestamp < event.end_sec
            for timestamp in perception_map.sample_timestamps
        )
    ]

    facts = []
    event_observations = []
    for response in responses:
        packet_facts, packet_events = ground_coverage_response(
            response,
            perception_map,
            fact_start_index=len(facts) + 1,
        )
        facts.extend(packet_facts)
        event_observations.extend(packet_events)
    facts.sort(
        key=lambda fact: (fact["earliest_timestamp_sec"], fact["fact_id"])
    )
    event_observations.sort(
        key=lambda event: (
            event["start_sec"],
            event["event_id"],
        )
    )
    observed_event_ids = [event["event_id"] for event in event_observations]
    coverage_complete = (
        not terminal_failures
        and submitted_frame_count == expected_frame_count
        and observed_event_ids == expected_event_ids
        and len(responses) == len(packets)
    )

    answer = None
    answer_trace = []
    answer_attempted = False
    answer_request_index = len(packets) + retry_count + 1
    if coverage_complete and facts:
        answer_attempted = True
        transport = DashScopeFlashTransport(
            base_url=base_url, generation_overrides={}
        )
        try:
            answer = answer_from_facts(
                transport,
                item,
                facts,
                args,
                answer_trace,
            )
            if answer_trace:
                answer_trace[0]["request_index"] = answer_request_index
        except Exception as exc:
            trace = (
                dict(answer_trace[0])
                if answer_trace
                else {
                    "stage": "ANSWER",
                    "status": "FAILED",
                }
            )
            trace["request_index"] = answer_request_index
            if trace.get("status") != "FAILED":
                trace["provider_status"] = trace.get("status")
            trace["status"] = "FAILED"
            trace["error_type"] = type(exc).__name__
            trace["error"] = str(exc)
            terminal_failures.append(
                {
                    "request_index": answer_request_index,
                    "stage": "ANSWER",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trace": trace,
                }
            )
        finally:
            transport.close()

    if not coverage_complete:
        status = "INCOMPLETE_COVERAGE"
        stop_reason = "coverage_packet_failed_or_incomplete"
    elif not facts:
        status = "INCOMPLETE_COVERAGE"
        stop_reason = "full_coverage_returned_no_grounded_facts"
    elif answer is None:
        status = "COMPLETED_WITH_FAILURES"
        stop_reason = "answer_failed"
    else:
        status = "COMPLETED"
        stop_reason = "full_1fps_coverage_answered"

    return {
        "schema_version": "hierarchical_event_tree_full_coverage_question_v1",
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "status": status,
        "question": item["question"],
        "options": list(item["options"]),
        "ground_truth_loaded": False,
        "coverage_mode": "full_1fps_event_aligned_packets",
        "coverage_complete": coverage_complete,
        "frame_budget": expected_frame_count,
        "submitted_frame_count": submitted_frame_count,
        "attempted_submitted_frame_count": attempted_submitted_frame_count,
        "packet_count": len(packets),
        "empty_packet_retry_limit": args.retry_empty_packets,
        "empty_packet_retry_count": retry_count,
        "request_count": len(packets) + retry_count + int(answer_attempted),
        "maximum_request_count": (
            len(packets) * (1 + args.retry_empty_packets) + 1
        ),
        "maximum_concurrent_requests": min(args.max_workers, len(packets)),
        "stop_reason": stop_reason,
        "expected_event_ids": expected_event_ids,
        "observed_event_ids": observed_event_ids,
        "event_observations": event_observations,
        "observations": facts,
        "answer_generated": answer is not None,
        "answer": answer,
        "failures": terminal_failures,
        "failed_attempts": attempt_failures,
        "requests": responses,
        "answer_request": answer_trace[0] if answer_trace else None,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }

def run_question(
    *,
    base_url: str,
    item: Mapping[str, Any],
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    video_path: str,
    query_embedding: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    budget = frame_budget(perception_map)
    used_frames = 0
    observed: set[str] = set()
    dense_leaves: set[str] = set()
    facts: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    normalization_warnings: list[str] = []
    failures: list[dict[str, Any]] = []
    fallback_used = False
    round_index = 0
    choices = [
        RefinementChoice(
            tree.root_id,
            min(args.initial_frame_budget, budget),
            "initial_global_overview",
        )
    ]
    stop_reason = "frontier_exhausted"

    while choices:
        round_index += 1
        runnable: list[tuple[RefinementChoice, tuple[ScheduledTreeFrame, ...], bool]] = []
        for choice in _deduplicate_choices(choices):
            remaining = budget - used_frames
            if remaining < MIN_NODE_FRAMES:
                stop_reason = "frame_budget_exhausted"
                break
            node = tree.node(choice.target_node_id)
            dense_pass = node.is_leaf and choice.target_node_id in observed
            count = min(choice.requested_frame_count, remaining)
            if choice.reason == "initial_global_overview":
                frames = schedule_global_overview_frames(
                    tree,
                    perception_map,
                    max_frames=count,
                )
            else:
                if not node.is_leaf:
                    count = MIN_NODE_FRAMES
                frames = schedule_node_frames(
                    tree,
                    perception_map,
                    choice.target_node_id,
                    frame_count=count,
                )
            runnable.append((choice, frames, dense_pass))
            used_frames += len(frames)
        if not runnable:
            break

        futures = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for choice, frames, dense_pass in runnable:
                request_index = len(requests) + len(futures) + 1
                future = executor.submit(
                    observe_node,
                    request_index=request_index,
                    base_url=base_url,
                    item=item,
                    tree=tree,
                    video_path=video_path,
                    choice=choice,
                    frames=frames,
                    max_edge=args.max_edge,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                futures[future] = (choice, dense_pass)
            completed = []
            for future in as_completed(futures):
                choice, dense_pass = futures[future]
                try:
                    completed.append((future.result(), choice, dense_pass))
                except NodeObservationFailure as exc:
                    failures.append(exc.failure)
                except Exception as exc:
                    failures.append(
                        {
                            "node_id": choice.target_node_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        completed.sort(key=lambda value: value[0]["request_index"])
        next_choices: list[RefinementChoice] = []
        for response, choice, dense_pass in completed:
            requests.append(response)
            observed.add(choice.target_node_id)
            if dense_pass:
                dense_leaves.add(choice.target_node_id)
            new_facts, fact_errors = grounded_facts(
                response, fact_start_index=len(facts) + 1
            )
            facts.extend(new_facts)
            validation_errors.extend(
                f"{choice.target_node_id}:{error}" for error in fact_errors
            )
            normalization_warnings.extend(
                f"{choice.target_node_id}:{warning}"
                for warning in response.get("normalization_warnings", [])
            )
            raw_refinements = response["parsed"].get("refinement_requests")
            if not isinstance(raw_refinements, list):
                validation_errors.append(
                    f"{choice.target_node_id}:refinement_requests_not_an_array"
                )
                raw_refinements = []
            accepted, rejected = normalize_refinement_requests(
                tree,
                choice.target_node_id,
                raw_refinements,
                already_observed=observed,
                dense_leaf_ids=dense_leaves,
            )
            next_choices.extend(accepted)
            validation_errors.extend(
                f"{choice.target_node_id}:{error}" for error in rejected
            )

        choices = _deduplicate_choices(next_choices)
        if choices:
            continue
        if not fallback_used:
            fallback_used = True
            fallback = select_embedding_fallback(
                tree,
                query_embedding,
                observed_node_ids=observed,
                dense_leaf_ids=dense_leaves,
            )
            if fallback is not None:
                choices = [fallback]
                continue
            stop_reason = "no_embedding_fallback_candidate"
        else:
            stop_reason = "no_new_valid_refinement_after_fallback"

    facts.sort(
        key=lambda fact: (fact["earliest_timestamp_sec"], fact["fact_id"])
    )
    status = "COMPLETED_WITH_FAILURES" if failures else "COMPLETED"
    return {
        "schema_version": "hierarchical_event_tree_shadow_question_v1",
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "status": status,
        "question": item["question"],
        "options": list(item["options"]),
        "ground_truth_loaded": False,
        "answer_generated": False,
        "frame_budget": budget,
        "submitted_frame_count": used_frames,
        "request_count": len(requests) + len(failures),
        "maximum_request_count": 1
        + max(0, budget - min(args.initial_frame_budget, budget))
        // MIN_NODE_FRAMES,
        "maximum_concurrent_requests": args.max_workers,
        "round_count": round_index,
        "observed_node_ids": sorted(
            observed, key=lambda node_id: tree.node(node_id).start_sec
        ),
        "dense_leaf_ids": sorted(
            dense_leaves, key=lambda node_id: tree.node(node_id).start_sec
        ),
        "embedding_fallback_used": fallback_used,
        "stop_reason": stop_reason,
        "observations": facts,
        "validation_errors": validation_errors,
        "normalization_warnings": normalization_warnings,
        "failures": failures,
        "requests": requests,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def load_query_embeddings(path: str | Path) -> dict[str, list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("ground_truth_loaded") is not False:
        raise ValueError("query_embedding_ground_truth_loaded_must_be_false")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("query_embeddings_must_be_an_object")
    return {
        str(question_id): [float(value) for value in vector]
        for question_id, vector in embeddings.items()
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    inputs = [load_inference_input(path) for path in args.input_json]
    map_records = [load_map_artifact(path) for path in args.perception_map]
    maps = {video_id: record for video_id, *record in map_records}
    if len(maps) != len(map_records):
        raise ValueError("perception_map_video_ids_must_be_unique")
    output_root = Path(args.out).resolve()
    if output_root.exists():
        raise FileExistsError(f"output_already_exists:{output_root}")
    output_root.mkdir(parents=True)
    trees = {
        video_id: build_transient_event_tree(
            record[2], branch_factor=args.branch_factor
        )
        for video_id, record in maps.items()
    }
    for video_id, tree in trees.items():
        write_json(output_root / "trees" / f"{video_id}.json", tree.to_dict())

    if args.dry_run:
        samples = []
        for item in inputs:
            video_path, duration_sec, perception_map = maps[item["video_id"]]
            if (
                str(Path(video_path).resolve()) != item["video_path"]
                or not math.isclose(duration_sec, item["duration_sec"], abs_tol=0.01)
            ):
                raise ValueError("inference_input_does_not_match_perception_map")
            if args.full_coverage_1fps:
                record = full_coverage_dry_run_record(
                    item,
                    trees[item["video_id"]],
                    perception_map,
                    max_frames_per_packet=args.max_frames_per_packet,
                    empty_packet_retry_limit=args.retry_empty_packets,
                )
            else:
                record = dry_run_record(
                    item,
                    trees[item["video_id"]],
                    perception_map,
                    initial_frame_budget=args.initial_frame_budget,
                )
            write_json(
                output_root / "questions" / item["question_id"] / "dry_run.json",
                record,
            )
            samples.append(record)
        manifest = {
            "schema_version": "hierarchical_event_tree_shadow_dry_run_v1",
            "status": "COMPLETED",
            "ground_truth_loaded": False,
            "options_visible": True,
            "provider_request_count": 0,
            "samples": [
                {
                    "question_id": sample["question_id"],
                    "video_id": sample["video_id"],
                    "tree_depth": sample["tree"]["tree_depth"],
                    "root_child_count": len(
                        trees[sample["video_id"]].root.child_ids
                    ),
                    "initial_request_count": len(sample["initial_frontier"]),
                    "initial_frame_count": sample["initial_frame_count"],
                    "frame_budget": sample["frame_budget"],
                    "maximum_request_count": sample["maximum_request_count"],
                }
                for sample in samples
            ],
        }
        write_json(output_root / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL_required")
    if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        raise RuntimeError("DASHSCOPE_API_KEY_required")
    embeddings = (
        {}
        if args.full_coverage_1fps
        else load_query_embeddings(args.query_embeddings)
    )
    started_at = utc_now()
    started = time.monotonic()
    samples = []
    for item in inputs:
        video_path, duration_sec, perception_map = maps[item["video_id"]]
        if (
            str(Path(video_path).resolve()) != item["video_path"]
            or not math.isclose(duration_sec, item["duration_sec"], abs_tol=0.01)
        ):
            raise ValueError("inference_input_does_not_match_perception_map")
        if args.full_coverage_1fps:
            sample = run_full_coverage_question(
                base_url=base_url,
                item=item,
                tree=trees[item["video_id"]],
                perception_map=perception_map,
                video_path=video_path,
                args=args,
            )
        else:
            if item["question_id"] not in embeddings:
                raise ValueError(f"query_embedding_missing:{item['question_id']}")
            sample = run_question(
                base_url=base_url,
                item=item,
                tree=trees[item["video_id"]],
                perception_map=perception_map,
                video_path=video_path,
                query_embedding=embeddings[item["question_id"]],
                args=args,
            )
        write_json(
            output_root / "questions" / item["question_id"] / "result.json",
            sample,
        )
        samples.append(sample)
    status = (
        "COMPLETED"
        if all(sample["status"] == "COMPLETED" for sample in samples)
        else "COMPLETED_WITH_FAILURES"
    )
    manifest = {
        "schema_version": (
            "hierarchical_event_tree_full_coverage_run_v1"
            if args.full_coverage_1fps
            else "hierarchical_event_tree_shadow_run_v1"
        ),
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": MODEL,
        "ground_truth_loaded": False,
        "options_visible": True,
        "answer_generated": all(
            sample.get("answer_generated", False) for sample in samples
        ),
        "cache_reuse_expected": False,
        "automatic_retry": bool(args.retry_empty_packets),
        "retry_scope": (
            "literal_empty_coverage_packets_only"
            if args.retry_empty_packets
            else "none"
        ),
        "empty_packet_retry_limit": args.retry_empty_packets,
        "empty_packet_retry_count": sum(
            sample.get("empty_packet_retry_count", 0) for sample in samples
        ),
        "maximum_concurrent_requests": args.max_workers,
        "maximum_provider_requests": sum(
            sample["maximum_request_count"] for sample in samples
        ),
        "actual_provider_requests": sum(
            sample["request_count"] for sample in samples
        ),
        "samples": [
            {
                "question_id": sample["question_id"],
                "status": sample["status"],
                "submitted_frame_count": sample["submitted_frame_count"],
                "coverage_complete": sample.get("coverage_complete"),
                "packet_count": sample.get("packet_count"),
                "answer_generated": sample.get("answer_generated", False),
                "request_count": sample["request_count"],
                "empty_packet_retry_count": sample.get(
                    "empty_packet_retry_count", 0
                ),
                "observation_count": len(sample["observations"]),
                "stop_reason": sample["stop_reason"],
            }
            for sample in samples
        ],
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
