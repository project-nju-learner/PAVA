"""Run Event-scoped 1 fps Observe -> Evidence -> Evidence Audit in shadow mode.

The runner deliberately stops before Planner, Hypothesis Verifier, and Answer.
Every Event is covered in timeline order.  Event boundaries only schedule
standard AVP observations; they do not create a parallel fact schema.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

from avp.video_utils import set_metadata_source
from avp.main import (
    Blackboard,
    Evidence,
    Observer,
    PlanSpec,
    SpatialTokenRate,
    WatchConfig,
)
from experiments.pollution_audit import (
    write_json,
    write_qa_only_metadata_source,
)
from experiments.semantic_delta_pilot.avp_integration import (
    OpenAIJSONBackend,
    RepairController,
)
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    make_dashscope_flash_client,
)
from scripts.run_event_1fps_observe_shadow import schedule_event_packets
from scripts.run_hierarchical_event_tree_shadow import load_map_artifact
from scripts.run_perception_shadow import load_inference_input


MODEL = "qwen3.8-flash"
MAX_PROVIDER_FRAMES = 200


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 4 <= int(args.max_frames_per_request) <= MAX_PROVIDER_FRAMES:
        raise ValueError("max_frames_per_request_must_be_between_4_and_200")


def schedule_audited_packets(
    perception_map: Any,
    *,
    max_frames_per_request: int = MAX_PROVIDER_FRAMES,
) -> tuple[dict[str, Any], ...]:
    """Add contiguous source ranges to the existing Event/1 fps packet plan."""

    raw_packets = schedule_event_packets(
        perception_map,
        max_frames_per_request=max_frames_per_request,
    )
    by_event: dict[str, list[Mapping[str, Any]]] = {}
    for packet in raw_packets:
        by_event.setdefault(str(packet["event_id"]), []).append(packet)

    scheduled: list[dict[str, Any]] = []
    for event in perception_map.events:
        packets = sorted(
            by_event.get(event.event_id, []),
            key=lambda item: int(item["segment_index"]),
        )
        if not packets:
            raise RuntimeError(f"event_has_no_observation_packet:{event.event_id}")
        if [int(item["segment_index"]) for item in packets] != list(
            range(1, len(packets) + 1)
        ):
            raise RuntimeError(f"event_packet_order_invalid:{event.event_id}")

        boundaries = [float(event.start_sec)]
        for left, right in zip(packets, packets[1:]):
            left_last = float(left["frames"][-1].timestamp_sec)
            right_first = float(right["frames"][0].timestamp_sec)
            boundary = (left_last + right_first) / 2.0
            if not boundaries[-1] < boundary < float(event.end_sec):
                raise RuntimeError(
                    f"event_packet_boundary_invalid:{event.event_id}"
                )
            boundaries.append(boundary)
        boundaries.append(float(event.end_sec))

        for index, packet in enumerate(packets):
            record = dict(packet)
            record["observation_start_sec"] = boundaries[index]
            record["observation_end_sec"] = boundaries[index + 1]
            if record["observation_end_sec"] <= record["observation_start_sec"]:
                raise RuntimeError(
                    f"event_packet_scope_invalid:{event.event_id}"
                )
            scheduled.append(record)

    scheduled.sort(key=lambda item: int(item["ordinal"]))
    if len(scheduled) != len(raw_packets):
        raise RuntimeError("audited_packet_count_changed")
    return tuple(scheduled)


def build_question_only_plan(
    *,
    question: str,
    packet: Mapping[str, Any],
) -> PlanSpec:
    """Build one standard AVP plan without exposing answer options."""

    event_id = str(packet["event_id"])
    start = float(packet["observation_start_sec"])
    end = float(packet["observation_end_sec"])
    return PlanSpec(
        plan_version="event_1fps_audited_shadow_v1",
        query=str(question),
        watch=WatchConfig(
            load_mode="region",
            fps=1.0,
            spatial_token_rate=SpatialTokenRate.low,
            regions=[(start, end)],
        ),
        description=(
            f"Observe Event {event_id} over the complete submitted range. "
            "Report directly visible or audible facts relevant to the original "
            "question with grounded timestamps; do not choose an answer."
        ),
        completion_criteria=(
            "The complete submitted Event range was inspected and every "
            "question-relevant observation was returned as standard key_evidence."
        ),
    )


def build_shadow_blackboard(
    *,
    video_path: str,
    duration_sec: float,
    packet: Mapping[str, Any],
) -> Blackboard:
    """Create an empty observation context; notably, it has no options."""

    return Blackboard(
        video_path=str(video_path),
        duration_sec=float(duration_sec),
        meta={
            "shadow_mode": "event_1fps_audited",
            "preserve_observation_regions": True,
            "event_id": str(packet["event_id"]),
            "packet_id": str(packet["packet_id"]),
        },
    )


def _plan_record(plan: PlanSpec) -> dict[str, Any]:
    record = dataclasses.asdict(plan)
    record["watch"]["spatial_token_rate"] = plan.watch.spatial_token_rate.value
    return record


def _packet_record(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": str(packet["packet_id"]),
        "ordinal": int(packet["ordinal"]),
        "event_id": str(packet["event_id"]),
        "event_index": int(packet["event_index"]),
        "event_start_sec": float(packet["event_start_sec"]),
        "event_end_sec": float(packet["event_end_sec"]),
        "segment_index": int(packet["segment_index"]),
        "segment_count": int(packet["segment_count"]),
        "observation_start_sec": float(packet["observation_start_sec"]),
        "observation_end_sec": float(packet["observation_end_sec"]),
        "source_sample_frame_count": int(packet["source_sample_frame_count"]),
        "supplemental_frame_count": int(packet["supplemental_frame_count"]),
        "scheduled_timestamps": [
            float(frame.timestamp_sec) for frame in packet["frames"]
        ],
    }


def run_audited_question(
    *,
    item: Mapping[str, Any],
    perception_map: Any,
    video_path: str,
    duration_sec: float,
    observer: Observer,
    audit_runtime: RepairController,
    max_frames_per_request: int = MAX_PROVIDER_FRAMES,
) -> dict[str, Any]:
    """Run the standardized shadow for one question without answering it."""

    started = time.monotonic()
    packets = schedule_audited_packets(
        perception_map,
        max_frames_per_request=max_frames_per_request,
    )
    evidence_rounds: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for round_id, packet in enumerate(packets, 1):
        plan = build_question_only_plan(
            question=str(item["question"]),
            packet=packet,
        )
        blackboard = build_shadow_blackboard(
            video_path=video_path,
            duration_sec=duration_sec,
            packet=packet,
        )
        try:
            evidence = observer.observe(plan, blackboard)
            if not isinstance(evidence, Evidence):
                raise TypeError("observer_must_return_standard_evidence")
            evidence.round_id = round_id
        except Exception as exc:
            failures.append(
                {
                    "round_id": round_id,
                    "packet_id": str(packet["packet_id"]),
                    "event_id": str(packet["event_id"]),
                    "stage": "OBSERVE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        round_record = {
            "round_id": round_id,
            "packet": _packet_record(packet),
            "plan": _plan_record(plan),
            "evidence": dataclasses.asdict(evidence),
            "transport_receipt": (
                evidence.model_call.get("observation_receipt")
                if isinstance(evidence.model_call, dict)
                else None
            ),
            "audit_attempted": True,
            "audit_admission": None,
        }
        try:
            admission = audit_runtime.admit_observation(
                query=str(item["question"]),
                plan=plan,
                evidence=evidence,
                interaction_history=[],
                current_confidence=None,
                video_path=str(video_path),
            )
            round_record["audit_admission"] = admission
            blackboard.add_evidence(evidence)
        except Exception as exc:
            failures.append(
                {
                    "round_id": round_id,
                    "packet_id": str(packet["packet_id"]),
                    "event_id": str(packet["event_id"]),
                    "stage": "EVIDENCE_AUDIT_ADMISSION",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        evidence_rounds.append(round_record)

    observed_rounds = len(evidence_rounds)
    audited_rounds = sum(
        record["audit_admission"] is not None for record in evidence_rounds
    )
    coverage_complete = observed_rounds == len(packets)
    audit_complete = audited_rounds == observed_rounds == len(packets)
    return {
        "schema_version": "event_1fps_audited_shadow_question_v1",
        "question_id": str(item["question_id"]),
        "video_id": str(item["video_id"]),
        "question": str(item["question"]),
        "status": (
            "COMPLETED"
            if coverage_complete and audit_complete and not failures
            else "COMPLETED_WITH_FAILURES"
        ),
        "ground_truth_loaded": False,
        "options_visible_to_observer": False,
        "options_visible_to_evidence_audit": False,
        "answer_generated": False,
        "coverage_mode": "all_events_one_fps_standard_observe",
        "coverage_complete": coverage_complete,
        "audit_complete": audit_complete,
        "event_count": len(perception_map.events),
        "packet_count": len(packets),
        "source_sample_frame_count": len(perception_map.sample_timestamps),
        "evidence_round_count": observed_rounds,
        "audited_round_count": audited_rounds,
        "packet_plan": [_packet_record(packet) for packet in packets],
        "evidence_rounds": evidence_rounds,
        "audit_summary": audit_runtime.summary(),
        "failures": failures,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def dry_run_record(
    *,
    item: Mapping[str, Any],
    perception_map: Any,
    max_frames_per_request: int,
) -> dict[str, Any]:
    packets = schedule_audited_packets(
        perception_map,
        max_frames_per_request=max_frames_per_request,
    )
    return {
        "schema_version": "event_1fps_audited_shadow_dry_run_v1",
        "question_id": str(item["question_id"]),
        "video_id": str(item["video_id"]),
        "question": str(item["question"]),
        "ground_truth_loaded": False,
        "options_visible_to_observer": False,
        "options_visible_to_evidence_audit": False,
        "provider_request_count": 0,
        "event_count": len(perception_map.events),
        "packet_count": len(packets),
        "source_sample_frame_count": len(perception_map.sample_timestamps),
        "packet_plan": [
            {
                **_packet_record(packet),
                "plan": _plan_record(
                    build_question_only_plan(
                        question=str(item["question"]),
                        packet=packet,
                    )
                ),
            }
            for packet in packets
        ],
    }


def _request_counter(client: Any) -> Any:
    counter = {"value": 0}
    original = client.client.post_chat_completion

    def counted(payload: Any) -> str:
        counter["value"] += 1
        return original(payload)

    client.client.post_chat_completion = counted
    return lambda: int(counter["value"])


def run_model_question(
    *,
    item: Mapping[str, Any],
    perception_map: Any,
    video_path: str,
    duration_sec: float,
    sample_dir: Path,
    base_url: str,
    max_frames_per_request: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    metadata_path = sample_dir / "qa_only_video_metadata.json"
    write_qa_only_metadata_source(
        metadata_path,
        video_path=Path(video_path),
        video_id=str(item["video_id"]),
        duration_sec=float(duration_sec),
    )
    set_metadata_source(str(metadata_path))
    observer_client = make_dashscope_flash_client(
        base_url=base_url,
        model=MODEL,
        temperature=temperature,
        seed=seed,
    )
    audit_client = make_dashscope_flash_client(
        base_url=base_url,
        model=MODEL,
        temperature=temperature,
        seed=seed,
    )
    observer_requests = _request_counter(observer_client)
    audit_requests = _request_counter(audit_client)
    audit_backend = OpenAIJSONBackend(
        base_url=base_url,
        model=MODEL,
        trace_path=sample_dir / "native" / "semantic_delta" / "model_calls.jsonl",
        temperature=temperature,
        seed=seed,
    )
    runtime = RepairController(
        run_dir=sample_dir / "native",
        backend=audit_backend,
        evidence_audit_client=audit_client,
        raw_qwen_only=False,
        max_repair_observations=1,
    )
    try:
        result = run_audited_question(
            item=item,
            perception_map=perception_map,
            video_path=video_path,
            duration_sec=duration_sec,
            observer=Observer(observer_client),
            audit_runtime=runtime,
            max_frames_per_request=max_frames_per_request,
        )
        result["observer_provider_request_count"] = observer_requests()
        result["evidence_audit_provider_request_count"] = audit_requests()
        result["auxiliary_audit_backend_request_count"] = audit_backend.request_count
        result["provider_request_count"] = (
            result["observer_provider_request_count"]
            + result["evidence_audit_provider_request_count"]
            + result["auxiliary_audit_backend_request_count"]
        )
        return result
    finally:
        observer_client.cleanup_uploaded_files()
        audit_client.cleanup_uploaded_files()


def _validate_input_map(
    item: Mapping[str, Any],
    *,
    video_path: str,
    duration_sec: float,
) -> None:
    if str(Path(video_path).resolve()) != str(item["video_path"]):
        raise ValueError("inference_input_does_not_match_perception_map")
    if not math.isclose(
        float(duration_sec),
        float(item["duration_sec"]),
        abs_tol=0.01,
    ):
        raise ValueError("inference_input_does_not_match_perception_map")


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
    started_at = utc_now()
    samples: list[dict[str, Any]] = []

    base_url = ""
    if not args.dry_run:
        base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL_required")
        if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
            raise RuntimeError("DASHSCOPE_API_KEY_required")

    for item in inputs:
        video_path, duration_sec, perception_map = maps[item["video_id"]]
        _validate_input_map(
            item,
            video_path=video_path,
            duration_sec=duration_sec,
        )
        sample_dir = output_root / "questions" / str(item["question_id"])
        sample_dir.mkdir(parents=True, exist_ok=False)
        if args.dry_run:
            sample = dry_run_record(
                item=item,
                perception_map=perception_map,
                max_frames_per_request=args.max_frames_per_request,
            )
            write_json(sample_dir / "dry_run.json", sample)
        else:
            sample = run_model_question(
                item=item,
                perception_map=perception_map,
                video_path=video_path,
                duration_sec=duration_sec,
                sample_dir=sample_dir,
                base_url=base_url,
                max_frames_per_request=args.max_frames_per_request,
                temperature=args.temperature,
                seed=args.seed,
            )
            write_json(sample_dir / "result.json", sample)
        samples.append(sample)

    status = (
        "COMPLETED"
        if all(sample.get("status", "COMPLETED") == "COMPLETED" for sample in samples)
        else "COMPLETED_WITH_FAILURES"
    )
    manifest = {
        "schema_version": (
            "event_1fps_audited_shadow_dry_run_v1"
            if args.dry_run
            else "event_1fps_audited_shadow_run_v1"
        ),
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": None if args.dry_run else MODEL,
        "ground_truth_loaded": False,
        "options_visible_to_observer": False,
        "options_visible_to_evidence_audit": False,
        "answer_generated": False,
        "cache_reuse_expected": False,
        "automatic_retry": False,
        "provider_request_count": sum(
            int(sample.get("provider_request_count") or 0) for sample in samples
        ),
        "samples": [
            {
                "question_id": sample["question_id"],
                "status": sample.get("status", "DRY_RUN"),
                "event_count": sample["event_count"],
                "packet_count": sample["packet_count"],
                "evidence_round_count": sample.get("evidence_round_count", 0),
                "audited_round_count": sample.get("audited_round_count", 0),
            }
            for sample in samples
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
