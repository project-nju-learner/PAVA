"""Run isolated Event-scoped 1 fps observation and global aggregation.

This shadow runner does not enter the AVP Controller, Evidence Audit,
Verifier, or Answer.  It keeps the production path unchanged while testing
whether complete 1 fps coverage becomes more reliable when every source Event
is observed independently and only grounded Event facts are aggregated.
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
import time
from typing import Any, Mapping, Sequence

from avp.perception_tools.contracts import PerceptionEvent, PerceptionMap
from avp.perception_tools.hierarchical_event_tree import (
    ScheduledTreeFrame,
    build_transient_event_tree,
)
from experiments.pollution_audit import write_json
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    DashScopeFlashTransport,
)
from scripts.run_hierarchical_event_tree_shadow import (
    NodeObservationFailure,
    load_map_artifact,
)
from scripts.run_perception_shadow import (
    decode_scheduled_frames,
    load_inference_input,
)
from scripts.run_query_driven_frame_shadow import (
    MODEL,
    call_json,
    encode_labeled_jpeg,
    option_block,
    response_format,
)


MIN_PROVIDER_FRAMES = 4
MAX_PROVIDER_FRAMES = 200
MAX_WORKERS = 8
EVENT_OBSERVE_MAX_TOKENS = 4096
AGGREGATE_MAX_TOKENS = 8192


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--perception-map", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--max-frames-per-request",
        type=int,
        default=MAX_PROVIDER_FRAMES,
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not MIN_PROVIDER_FRAMES <= args.max_frames_per_request <= MAX_PROVIDER_FRAMES:
        raise ValueError("max_frames_per_request_must_be_between_4_and_200")
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ValueError("max_workers_must_be_between_one_and_eight")
    if args.max_edge < 64:
        raise ValueError("max_edge_must_be_at_least_64")


def _source_timestamps(
    perception_map: PerceptionMap,
    event: PerceptionEvent,
) -> tuple[float, ...]:
    return tuple(
        float(timestamp)
        for timestamp in perception_map.sample_timestamps
        if event.start_sec <= float(timestamp) < event.end_sec
    )


def _at_least_four_event_timestamps(
    event: PerceptionEvent,
    source_timestamps: Sequence[float],
) -> tuple[float, ...]:
    selected = list(float(value) for value in source_timestamps)
    used = {round(value, 9) for value in selected}
    if len(selected) >= MIN_PROVIDER_FRAMES:
        return tuple(sorted(selected))
    duration = float(event.end_sec - event.start_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"event_duration_must_be_positive:{event.event_id}")
    for denominator in (4, 8, 16, 32, 64, 128):
        for index in range(denominator):
            value = float(
                event.start_sec + (index + 0.5) * duration / denominator
            )
            key = round(value, 9)
            if key in used:
                continue
            used.add(key)
            selected.append(value)
            if len(selected) >= MIN_PROVIDER_FRAMES:
                return tuple(sorted(selected))
    raise ValueError(f"event_cannot_supply_four_distinct_frames:{event.event_id}")


def _balanced_chunks(
    values: Sequence[float],
    *,
    maximum_size: int,
) -> tuple[tuple[float, ...], ...]:
    if len(values) <= maximum_size:
        return (tuple(values),)
    count = int(math.ceil(len(values) / maximum_size))
    bounds = [round(index * len(values) / count) for index in range(count + 1)]
    chunks = tuple(
        tuple(values[bounds[index] : bounds[index + 1]])
        for index in range(count)
    )
    if (
        not chunks
        or any(
            not MIN_PROVIDER_FRAMES <= len(chunk) <= maximum_size
            for chunk in chunks
        )
        or tuple(value for chunk in chunks for value in chunk) != tuple(values)
    ):
        raise RuntimeError("event_packet_partition_contract_failed")
    return chunks


def schedule_event_packets(
    perception_map: PerceptionMap,
    *,
    max_frames_per_request: int = MAX_PROVIDER_FRAMES,
) -> tuple[dict[str, Any], ...]:
    """Schedule complete 1 fps coverage as one Event-scoped request at a time."""
    if not math.isclose(perception_map.sampling_fps, 1.0, abs_tol=1e-6):
        raise ValueError("event_observe_requires_one_fps_perception_map")
    if (
        not perception_map.coverage_complete_at_declared_cadence
        or perception_map.coverage_violations
        or perception_map.errors
    ):
        raise ValueError("event_observe_requires_complete_perception_map")
    if not MIN_PROVIDER_FRAMES <= max_frames_per_request <= MAX_PROVIDER_FRAMES:
        raise ValueError("max_frames_per_request_outside_contract")

    packets: list[dict[str, Any]] = []
    source_sample_keys = {
        round(float(timestamp), 9)
        for timestamp in perception_map.sample_timestamps
    }
    ordinal = 0
    for event in perception_map.events:
        source = _source_timestamps(perception_map, event)
        event_source_keys = {round(timestamp, 9) for timestamp in source}
        scheduled = _at_least_four_event_timestamps(event, source)
        chunks = _balanced_chunks(
            scheduled,
            maximum_size=max_frames_per_request,
        )
        for segment_index, chunk in enumerate(chunks, 1):
            ordinal += 1
            packet_id = f"EV{event.event_index:06d}S{segment_index:02d}"
            frames = tuple(
                ScheduledTreeFrame(
                    node_id=event.event_id,
                    child_id=event.event_id,
                    source_event_id=event.event_id,
                    timestamp_sec=timestamp,
                )
                for timestamp in chunk
            )
            packets.append(
                {
                    "packet_id": packet_id,
                    "ordinal": ordinal,
                    "event_id": event.event_id,
                    "event_index": event.event_index,
                    "event_start_sec": float(event.start_sec),
                    "event_end_sec": float(event.end_sec),
                    "segment_index": segment_index,
                    "segment_count": len(chunks),
                    "recovery_of_packet_id": None,
                    "frames": frames,
                    "_source_sample_timestamp_keys": tuple(
                        sorted(event_source_keys)
                    ),
                    "source_sample_frame_count": sum(
                        round(timestamp, 9) in event_source_keys
                        for timestamp in chunk
                    ),
                    "supplemental_frame_count": sum(
                        round(timestamp, 9) not in event_source_keys
                        for timestamp in chunk
                    ),
                }
            )

    scheduled_source = [
        round(frame.timestamp_sec, 9)
        for packet in packets
        for frame in packet["frames"]
        if round(frame.timestamp_sec, 9) in source_sample_keys
    ]
    expected_source = [
        round(float(timestamp), 9)
        for timestamp in perception_map.sample_timestamps
    ]
    if scheduled_source != expected_source:
        raise RuntimeError("event_packets_must_preserve_every_one_fps_sample")
    if any(
        len({frame.source_event_id for frame in packet["frames"]}) != 1
        for packet in packets
    ):
        raise RuntimeError("event_packet_must_not_mix_source_events")
    return tuple(packets)


def event_observation_format() -> dict[str, Any]:
    return response_format(
        "event_scoped_observation_v1",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "OBSERVED",
                        "NO_RELEVANT_FACT",
                        "INDETERMINATE",
                    ],
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_ids": {
                                "type": "array",
                                "minItems": 1,
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
            "required": ["status", "facts"],
            "additionalProperties": False,
        },
    )


def _frame_manifest(packet: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = []
    frame_ids = []
    for index, frame in enumerate(packet["frames"], 1):
        frame_id = f"{packet['packet_id']}F{index:03d}"
        frame_ids.append(frame_id)
        item = frame.to_dict()
        item["frame_id"] = frame_id
        manifest.append(item)
    return manifest, frame_ids


def validate_event_observation(
    payload: Mapping[str, Any],
    *,
    event_id: str,
    frame_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status = str(payload.get("status") or "")
    if status not in {"OBSERVED", "NO_RELEVANT_FACT", "INDETERMINATE"}:
        raise ValueError("event_observation_invalid_status")
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("event_observation_facts_must_be_an_array")
    if status == "OBSERVED" and not raw_facts:
        raise ValueError("event_observation_observed_without_fact")
    if status == "NO_RELEVANT_FACT" and raw_facts:
        raise ValueError("event_observation_no_relevant_fact_has_fact")

    by_frame_id = {
        str(frame["frame_id"]): frame
        for frame in frame_manifest
    }
    if any(
        str(frame.get("source_event_id") or "") != event_id
        for frame in frame_manifest
    ):
        raise ValueError("submitted_frame_wrong_event_binding")
    facts = []
    for index, raw_fact in enumerate(raw_facts):
        if not isinstance(raw_fact, dict):
            raise ValueError(f"event_fact_{index}:not_an_object")
        cited = raw_fact.get("frame_ids")
        text = str(raw_fact.get("fact") or "").strip()
        if not isinstance(cited, list) or not cited or not text:
            raise ValueError(f"event_fact_{index}:invalid")
        normalized_ids = [str(frame_id) for frame_id in cited]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"event_fact_{index}:duplicate_frame_id")
        if any(frame_id not in by_frame_id for frame_id in normalized_ids):
            raise ValueError(f"event_fact_{index}:unknown_frame_id")
        if any(
            str(by_frame_id[frame_id].get("source_event_id") or "") != event_id
            for frame_id in normalized_ids
        ):
            raise ValueError(f"event_fact_{index}:wrong_event_binding")
        facts.append({"frame_ids": normalized_ids, "fact": text})
    return {"status": status, "facts": facts}


def observe_event_packet(
    *,
    request_index: int,
    base_url: str,
    item: Mapping[str, Any],
    video_path: str,
    packet: Mapping[str, Any],
    max_edge: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    frame_manifest, frame_ids = _frame_manifest(packet)
    images, decoder = decode_scheduled_frames(
        video_path,
        [frame.timestamp_sec for frame in packet["frames"]],
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
    trace_records: list[dict[str, Any]] = []
    transport = DashScopeFlashTransport(
        base_url=base_url,
        generation_overrides={},
    )
    try:
        parsed = call_json(
            transport,
            stage="EVENT_1FPS_OBSERVE",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a chronological visual observer, not an answerer. "
                        "The request contains frames from exactly one Event. Return "
                        "one JSON object with status and facts; never return a bare "
                        "array. Preserve every distinct visible fact that is either "
                        "directly relevant to the question/options or useful for "
                        "global theme, counting, chronology, and later retrieval. "
                        "Do not write a frame-by-frame diary and do not impose an "
                        "arbitrary fact-count limit. Cite only the supplied frame "
                        "IDs. Use OBSERVED when facts are grounded, "
                        "NO_RELEVANT_FACT with facts=[] only after inspecting all "
                        "frames and finding no useful fact, and INDETERMINATE when "
                        "the frames cannot support a reliable observation. Do not "
                        "infer unseen gaps or select an answer option."
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
                                f"Event: {packet['event_id']} "
                                f"[{packet['event_start_sec']:.6f}, "
                                f"{packet['event_end_sec']:.6f})\n"
                                f"Segment: {packet['segment_index']}/"
                                f"{packet['segment_count']}\n"
                                "Frame manifest:\n"
                                + json.dumps(
                                    frame_manifest,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            ),
                        },
                        {
                            "type": "video",
                            "video": frame_uris,
                            "fps": 1.0,
                        },
                    ],
                },
            ],
            schema=event_observation_format(),
            enable_thinking=False,
            temperature=temperature,
            seed=seed,
            max_tokens=EVENT_OBSERVE_MAX_TOKENS,
            request_trace=trace_records,
            submitted_frame_count=len(frame_ids),
        )
        normalized = validate_event_observation(
            parsed,
            event_id=str(packet["event_id"]),
            frame_manifest=frame_manifest,
        )
    except Exception as exc:
        trace = (
            dict(trace_records[0])
            if trace_records
            else {
                "stage": "EVENT_1FPS_OBSERVE",
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
                "event_id": packet["event_id"],
                "submitted_frames": frame_manifest,
                "decoder": decoder,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trace": trace,
            }
        ) from exc
    finally:
        transport.close()

    trace = trace_records[0]
    trace["request_index"] = request_index
    return {
        "request_index": request_index,
        "packet_id": packet["packet_id"],
        "event_id": packet["event_id"],
        "segment_index": packet["segment_index"],
        "segment_count": packet["segment_count"],
        "recovery_of_packet_id": packet.get("recovery_of_packet_id"),
        "submitted_frames": frame_manifest,
        "decoder": decoder,
        "observation": normalized,
        "trace": trace,
    }


def _output_failure_is_recoverable(failure: Mapping[str, Any]) -> bool:
    error = str(failure.get("error") or "").lower()
    if any(
        marker in error
        for marker in (
            "unknown_frame_id",
            "wrong_event_binding",
            "submitted_frame_wrong_event_binding",
        )
    ):
        return False
    trace = failure.get("trace")
    if not isinstance(trace, Mapping) or "raw_response" not in trace:
        return False
    return str(failure.get("error_type") or "") in {
        "JSONDecodeError",
        "ValueError",
    }


def recovery_packets(packet: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    frames = tuple(packet["frames"])
    if len(frames) < 2 * MIN_PROVIDER_FRAMES:
        retry = dict(packet)
        retry["packet_id"] = f"{packet['packet_id']}R1"
        retry["recovery_of_packet_id"] = packet["packet_id"]
        return (retry,)

    midpoint = len(frames) // 2
    chunks = (frames[:midpoint], frames[midpoint:])
    recovered = []
    for index, chunk in enumerate(chunks, 1):
        if len(chunk) < MIN_PROVIDER_FRAMES:
            raise RuntimeError("recovery_split_below_provider_minimum")
        child = dict(packet)
        child["packet_id"] = f"{packet['packet_id']}R{index}"
        child["recovery_of_packet_id"] = packet["packet_id"]
        child["frames"] = chunk
        source_keys = set(packet["_source_sample_timestamp_keys"])
        child["source_sample_frame_count"] = sum(
            round(frame.timestamp_sec, 9) in source_keys
            for frame in chunk
        )
        child["supplemental_frame_count"] = (
            len(chunk) - child["source_sample_frame_count"]
        )
        recovered.append(child)
    return tuple(recovered)


def _observe_batch(
    packets: Sequence[Mapping[str, Any]],
    *,
    request_index_start: int,
    base_url: str,
    item: Mapping[str, Any],
    video_path: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    responses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    futures = {}
    with ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(packets))
    ) as executor:
        for offset, packet in enumerate(packets):
            request_index = request_index_start + offset
            future = executor.submit(
                observe_event_packet,
                request_index=request_index,
                base_url=base_url,
                item=item,
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
                responses.append(dict(future.result()))
            except NodeObservationFailure as exc:
                failures.append(dict(exc.failure))
            except Exception as exc:
                failures.append(
                    {
                        "packet_id": packet["packet_id"],
                        "event_id": packet["event_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    return responses, failures


def _ground_and_merge(
    responses: Sequence[Mapping[str, Any]],
    perception_map: PerceptionMap,
    *,
    complete_event_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    fragment_statuses: dict[str, list[str]] = {}
    for response in responses:
        event_id = str(response["event_id"])
        frame_manifest = {
            str(frame["frame_id"]): frame
            for frame in response["submitted_frames"]
        }
        fragment_statuses.setdefault(event_id, []).append(
            str(response["observation"]["status"])
        )
        for raw_fact in response["observation"]["facts"]:
            locators = [
                frame_manifest[str(frame_id)]
                for frame_id in raw_fact["frame_ids"]
            ]
            if any(
                str(locator.get("source_event_id") or "") != event_id
                for locator in locators
            ):
                raise ValueError("grounding_detected_wrong_event_binding")
            candidates.append(
                {
                    "fact": str(raw_fact["fact"]),
                    "event_id": event_id,
                    "frame_locators": locators,
                    "earliest_timestamp_sec": min(
                        float(locator["timestamp_sec"])
                        for locator in locators
                    ),
                }
            )

    unique = {}
    for fact in candidates:
        locator_key = tuple(
            (
                str(locator["source_event_id"]),
                round(float(locator["timestamp_sec"]), 6),
            )
            for locator in fact["frame_locators"]
        )
        key = (fact["fact"], locator_key)
        unique.setdefault(key, fact)
    facts = sorted(
        unique.values(),
        key=lambda fact: (
            fact["earliest_timestamp_sec"],
            fact["event_id"],
            fact["fact"],
        ),
    )
    for index, fact in enumerate(facts, 1):
        fact["fact_id"] = f"EVF{index:04d}"

    fact_ids_by_event: dict[str, list[str]] = {}
    for fact in facts:
        fact_ids_by_event.setdefault(fact["event_id"], []).append(fact["fact_id"])
    event_records = []
    for event in perception_map.events:
        statuses = fragment_statuses.get(event.event_id, [])
        fact_ids = fact_ids_by_event.get(event.event_id, [])
        coverage_complete = event.event_id in complete_event_ids
        if not coverage_complete:
            observation_status = "INDETERMINATE"
        elif fact_ids:
            observation_status = "OBSERVED"
        elif statuses and all(
            status == "NO_RELEVANT_FACT" for status in statuses
        ):
            observation_status = "NO_RELEVANT_FACT"
        else:
            observation_status = "INDETERMINATE"
        event_records.append(
            {
                "event_id": event.event_id,
                "start_sec": round(event.start_sec, 6),
                "end_sec": round(event.end_sec, 6),
                "coverage_complete": coverage_complete,
                "observation_status": observation_status,
                "fragment_statuses": statuses,
                "fact_ids": fact_ids,
            }
        )
    return facts, event_records


def aggregation_format() -> dict[str, Any]:
    referenced_finding = {
        "type": "object",
        "properties": {
            "fact_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "finding": {"type": "string", "minLength": 1},
        },
        "required": ["fact_ids", "finding"],
        "additionalProperties": False,
    }
    return response_format(
        "event_global_work_map_v1",
        {
            "type": "object",
            "properties": {
                "overview": {"type": "string", "minLength": 1},
                "chronological_findings": {
                    "type": "array",
                    "items": referenced_finding,
                },
                "cross_event_patterns": {
                    "type": "array",
                    "items": referenced_finding,
                },
                "unresolved_questions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": [
                "overview",
                "chronological_findings",
                "cross_event_patterns",
                "unresolved_questions",
            ],
            "additionalProperties": False,
        },
    )


def aggregate_event_facts(
    *,
    request_index: int,
    base_url: str,
    item: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    event_observations: Sequence[Mapping[str, Any]],
    temperature: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_records: list[dict[str, Any]] = []
    transport = DashScopeFlashTransport(
        base_url=base_url,
        generation_overrides={},
    )
    try:
        parsed = call_json(
            transport,
            stage="EVENT_GLOBAL_AGGREGATE",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Build a question-scoped global work map from grounded "
                        "Event facts. Do not select, rank, or recommend an answer "
                        "option. Do not create new visual facts. Every finding "
                        "must cite the fact IDs that support it. Preserve useful "
                        "chronology, counts, repeated patterns, and explicit "
                        "remaining gaps. The work map is a Planner aid, not "
                        "Evidence. Events marked INDETERMINATE or with incomplete "
                        "coverage are explicit gaps; never infer what their "
                        "unobserved frames contain."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {item['question']}\n"
                        f"Official options:\n{option_block(item)}\n"
                        "Event coverage and observation status:\n"
                        + json.dumps(
                            list(event_observations),
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                        "Grounded Event facts:\n"
                        + json.dumps(
                            list(facts),
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                },
            ],
            schema=aggregation_format(),
            enable_thinking=False,
            temperature=temperature,
            seed=seed,
            max_tokens=AGGREGATE_MAX_TOKENS,
            request_trace=trace_records,
        )
        allowed = {str(fact["fact_id"]) for fact in facts}
        for section in ("chronological_findings", "cross_event_patterns"):
            entries = parsed.get(section)
            if not isinstance(entries, list):
                raise ValueError(f"aggregation_{section}_must_be_an_array")
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(f"aggregation_{section}_{index}:invalid")
                cited = entry.get("fact_ids")
                finding = str(entry.get("finding") or "").strip()
                if (
                    not isinstance(cited, list)
                    or not cited
                    or not finding
                    or any(str(fact_id) not in allowed for fact_id in cited)
                ):
                    raise ValueError(
                        f"aggregation_{section}_{index}:invalid_fact_binding"
                    )
        if not str(parsed.get("overview") or "").strip():
            raise ValueError("aggregation_overview_required")
        unresolved = parsed.get("unresolved_questions")
        if not isinstance(unresolved, list):
            raise ValueError("aggregation_unresolved_questions_must_be_an_array")
    except Exception as exc:
        trace = (
            dict(trace_records[0])
            if trace_records
            else {
                "stage": "EVENT_GLOBAL_AGGREGATE",
                "status": "FAILED",
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
                "stage": "EVENT_GLOBAL_AGGREGATE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trace": trace,
            }
        ) from exc
    finally:
        transport.close()
    trace = trace_records[0]
    trace["request_index"] = request_index
    return parsed, trace


def run_event_observe_question(
    *,
    base_url: str,
    item: Mapping[str, Any],
    perception_map: PerceptionMap,
    video_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    packets = schedule_event_packets(
        perception_map,
        max_frames_per_request=args.max_frames_per_request,
    )
    initial_responses, initial_failures = _observe_batch(
        packets,
        request_index_start=1,
        base_url=base_url,
        item=item,
        video_path=video_path,
        args=args,
    )
    for response in initial_responses:
        response["attempt"] = 1
    packet_by_id = {packet["packet_id"]: packet for packet in packets}
    logical_success = {
        response["packet_id"]
        for response in initial_responses
    }
    failed_attempts = []
    terminal_failures = []
    recovery_plan: dict[str, tuple[dict[str, Any], ...]] = {}
    recovery_jobs = []
    for raw_failure in initial_failures:
        failure = dict(raw_failure)
        failure["attempt"] = 1
        parent_id = str(failure.get("packet_id") or "")
        packet = packet_by_id.get(parent_id)
        eligible = packet is not None and _output_failure_is_recoverable(failure)
        failure["output_recovery_eligible"] = eligible
        if eligible:
            children = recovery_packets(packet)
            recovery_plan[parent_id] = children
            failure["recovery_action"] = (
                "split_once" if len(children) == 2 else "retry_same_payload_once"
            )
            failure["recovery_packet_ids"] = [
                child["packet_id"] for child in children
            ]
            recovery_jobs.extend(children)
        else:
            failure["recovery_action"] = "none"
            terminal_failures.append(failure)
        failed_attempts.append(failure)

    recovery_responses: list[dict[str, Any]] = []
    recovery_failures: list[dict[str, Any]] = []
    if recovery_jobs:
        recovery_responses, recovery_failures = _observe_batch(
            recovery_jobs,
            request_index_start=len(packets) + 1,
            base_url=base_url,
            item=item,
            video_path=video_path,
            args=args,
        )
        for response in recovery_responses:
            response["attempt"] = 2
        for failure in recovery_failures:
            failure["attempt"] = 2
            failure["output_recovery_eligible"] = False
            failure["recovery_action"] = "none"
            failed_attempts.append(failure)

    recovered_response_ids = {
        response["packet_id"] for response in recovery_responses
    }
    failed_recovery_ids = {
        str(failure.get("packet_id") or "")
        for failure in recovery_failures
    }
    for parent_id, children in recovery_plan.items():
        child_ids = {child["packet_id"] for child in children}
        if (
            child_ids <= recovered_response_ids
            and not child_ids & failed_recovery_ids
        ):
            logical_success.add(parent_id)
        else:
            terminal_failures.extend(
                failure
                for failure in recovery_failures
                if str(failure.get("packet_id") or "") in child_ids
            )
            if not child_ids & failed_recovery_ids:
                terminal_failures.append(
                    {
                        "packet_id": parent_id,
                        "event_id": packet_by_id[parent_id]["event_id"],
                        "error_type": "RuntimeError",
                        "error": "recovery_did_not_cover_all_child_packets",
                    }
                )

    responses = initial_responses + recovery_responses
    responses.sort(key=lambda response: int(response["request_index"]))
    failed_attempts.sort(
        key=lambda failure: int(failure.get("request_index", 0))
    )
    terminal_failures.sort(
        key=lambda failure: int(failure.get("request_index", 0))
    )
    event_packet_ids: dict[str, set[str]] = {}
    for packet in packets:
        event_packet_ids.setdefault(packet["event_id"], set()).add(
            packet["packet_id"]
        )
    complete_event_ids = {
        event_id
        for event_id, packet_ids in event_packet_ids.items()
        if packet_ids <= logical_success
    }
    coverage_complete = (
        len(logical_success) == len(packets)
        and len(complete_event_ids) == len(perception_map.events)
        and not terminal_failures
    )
    facts, event_observations = _ground_and_merge(
        responses,
        perception_map,
        complete_event_ids=complete_event_ids,
    )

    aggregation = None
    aggregation_trace = None
    aggregation_failure = None
    aggregation_attempted = bool(facts)
    if aggregation_attempted:
        try:
            aggregation, aggregation_trace = aggregate_event_facts(
                request_index=len(packets) + len(recovery_jobs) + 1,
                base_url=base_url,
                item=item,
                facts=facts,
                event_observations=event_observations,
                temperature=args.temperature,
                seed=args.seed,
            )
        except NodeObservationFailure as exc:
            aggregation_failure = dict(exc.failure)
            terminal_failures.append(aggregation_failure)
        except Exception as exc:
            aggregation_failure = {
                "stage": "EVENT_GLOBAL_AGGREGATE",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            terminal_failures.append(aggregation_failure)

    if not coverage_complete:
        status = "INCOMPLETE_COVERAGE"
        stop_reason = "event_observation_failed_or_incomplete"
    elif aggregation_failure is not None:
        status = "COMPLETED_WITH_FAILURES"
        stop_reason = "global_aggregation_failed"
    elif not facts:
        status = "COMPLETED"
        stop_reason = "coverage_complete_no_relevant_fact"
    else:
        status = "COMPLETED"
        stop_reason = "event_observation_and_aggregation_complete"

    attempted_frame_count = sum(
        len(response.get("submitted_frames", []))
        for response in responses
    ) + sum(
        len(failure.get("submitted_frames", []))
        for failure in failed_attempts
    )
    return {
        "schema_version": "event_1fps_observe_shadow_question_v1",
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "status": status,
        "question": item["question"],
        "options": list(item["options"]),
        "ground_truth_loaded": False,
        "coverage_mode": "one_fps_event_scoped_observation",
        "coverage_complete": coverage_complete,
        "source_sample_frame_count": len(perception_map.sample_timestamps),
        "scheduled_frame_count": sum(
            len(packet["frames"]) for packet in packets
        ),
        "supplemental_frame_count": sum(
            packet["supplemental_frame_count"] for packet in packets
        ),
        "attempted_submitted_frame_count": attempted_frame_count,
        "event_count": len(perception_map.events),
        "packet_count": len(packets),
        "recovery_request_count": len(recovery_jobs),
        "request_count": (
            len(packets)
            + len(recovery_jobs)
            + int(aggregation_attempted)
        ),
        "maximum_request_count": len(packets) * 3 + 1,
        "maximum_concurrent_requests": min(args.max_workers, len(packets)),
        "stop_reason": stop_reason,
        "event_observations": event_observations,
        "observations": facts,
        "global_work_map": aggregation,
        "global_aggregation_request": aggregation_trace,
        "failures": terminal_failures,
        "failed_attempts": failed_attempts,
        "requests": responses,
        "packet_plan": [
            {
                **{
                    key: value
                    for key, value in packet.items()
                    if key != "frames" and not key.startswith("_")
                },
                "frame_count": len(packet["frames"]),
                "frames": [
                    frame.to_dict() for frame in packet["frames"]
                ],
            }
            for packet in packets
        ],
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def dry_run_record(
    item: Mapping[str, Any],
    perception_map: PerceptionMap,
    *,
    max_frames_per_request: int,
) -> dict[str, Any]:
    packets = schedule_event_packets(
        perception_map,
        max_frames_per_request=max_frames_per_request,
    )
    return {
        "schema_version": "event_1fps_observe_shadow_dry_run_v1",
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "ground_truth_loaded": False,
        "event_count": len(perception_map.events),
        "packet_count": len(packets),
        "source_sample_frame_count": len(perception_map.sample_timestamps),
        "scheduled_frame_count": sum(
            len(packet["frames"]) for packet in packets
        ),
        "supplemental_frame_count": sum(
            packet["supplemental_frame_count"] for packet in packets
        ),
        "maximum_request_count": len(packets) * 3 + 1,
        "maximum_concurrent_requests": min(MAX_WORKERS, len(packets)),
        "packet_plan": [
            {
                "packet_id": packet["packet_id"],
                "event_id": packet["event_id"],
                "segment_index": packet["segment_index"],
                "segment_count": packet["segment_count"],
                "frame_count": len(packet["frames"]),
                "source_sample_frame_count": packet[
                    "source_sample_frame_count"
                ],
                "supplemental_frame_count": packet[
                    "supplemental_frame_count"
                ],
                "frames": [
                    frame.to_dict() for frame in packet["frames"]
                ],
            }
            for packet in packets
        ],
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
    for video_id, record in maps.items():
        tree = build_transient_event_tree(record[2])
        write_json(
            output_root / "trees" / f"{video_id}.json",
            tree.to_dict(),
        )

    if args.dry_run:
        samples = []
        for item in inputs:
            video_path, duration_sec, perception_map = maps[item["video_id"]]
            if (
                str(Path(video_path).resolve()) != item["video_path"]
                or not math.isclose(
                    duration_sec,
                    item["duration_sec"],
                    abs_tol=0.01,
                )
            ):
                raise ValueError("inference_input_does_not_match_perception_map")
            sample = dry_run_record(
                item,
                perception_map,
                max_frames_per_request=args.max_frames_per_request,
            )
            write_json(
                output_root
                / "questions"
                / item["question_id"]
                / "dry_run.json",
                sample,
            )
            samples.append(sample)
        manifest = {
            "schema_version": "event_1fps_observe_shadow_dry_run_v1",
            "status": "COMPLETED",
            "ground_truth_loaded": False,
            "options_visible": True,
            "provider_request_count": 0,
            "samples": samples,
        }
        write_json(output_root / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL_required")
    if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        raise RuntimeError("DASHSCOPE_API_KEY_required")

    started_at = utc_now()
    started = time.monotonic()
    samples = []
    for item in inputs:
        video_path, duration_sec, perception_map = maps[item["video_id"]]
        if (
            str(Path(video_path).resolve()) != item["video_path"]
            or not math.isclose(
                duration_sec,
                item["duration_sec"],
                abs_tol=0.01,
            )
        ):
            raise ValueError("inference_input_does_not_match_perception_map")
        sample = run_event_observe_question(
            base_url=base_url,
            item=item,
            perception_map=perception_map,
            video_path=video_path,
            args=args,
        )
        write_json(
            output_root
            / "questions"
            / item["question_id"]
            / "result.json",
            sample,
        )
        samples.append(sample)

    status = (
        "COMPLETED"
        if all(sample["status"] == "COMPLETED" for sample in samples)
        else "COMPLETED_WITH_FAILURES"
    )
    manifest = {
        "schema_version": "event_1fps_observe_shadow_run_v1",
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": MODEL,
        "ground_truth_loaded": False,
        "options_visible": True,
        "answer_generated": False,
        "cache_reuse_expected": False,
        "automatic_retry": True,
        "retry_scope": "output_format_failure_one_recovery_layer_only",
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
                "coverage_complete": sample["coverage_complete"],
                "event_count": sample["event_count"],
                "packet_count": sample["packet_count"],
                "recovery_request_count": sample["recovery_request_count"],
                "observation_count": len(sample["observations"]),
                "global_work_map_generated": (
                    sample["global_work_map"] is not None
                ),
                "request_count": sample["request_count"],
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
