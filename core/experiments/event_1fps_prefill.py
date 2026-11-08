"""Opt-in Event 1fps acquisition before the existing Controller decision loop."""
from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

from avp.main import Evidence, Observer, plan_from_dict
from experiments.run_event_1fps_verifier_answer_replay import whole_video_coverage
from experiments.semantic_delta_pilot.avp_integration import _validated_whole_video_coverage
from scripts.run_event_1fps_audited_shadow import run_audited_question, schedule_audited_packets
from scripts.run_hierarchical_event_tree_shadow import load_map_artifact


def prepare_event_prefill(controller, *, row, map_path):
    """Scan into the same Audit runtime, then return its last admitted round.

    No Verifier, Answer, replayed Audit, or alternate evidence store is used here.
    A failed scan remains a failed opt-in run; it never silently falls back.
    """
    runtime = controller.admission_runtime
    if runtime is None or controller.event_assist_runtime is not None:
        raise ValueError("event_prefill_requires_unassisted_repair_mode")
    if controller.bb.evidences or controller._load_latest_plan() is not None:
        raise ValueError("event_prefill_requires_fresh_controller")
    video_id, video_path, duration, perception_map = load_map_artifact(map_path)
    if (video_id != str(row["video_id"])
            or Path(video_path).resolve() != Path(controller.bb.video_path).resolve()
            or not math.isclose(duration, controller.bb.duration_sec, abs_tol=0.01)):
        raise ValueError("event_prefill_map_video_mismatch")
    packets = schedule_audited_packets(perception_map)
    cursor = 0.0
    for packet in packets:
        if abs(packet["observation_start_sec"] - cursor) > 0.001:
            raise ValueError("event_prefill_map_has_gap_or_overlap")
        cursor = packet["observation_end_sec"]
    if not packets or abs(cursor - duration) > 0.001:
        raise ValueError("event_prefill_map_incomplete")
    artifact = controller.store.root / "event_prefill.json"
    if artifact.exists():
        raise FileExistsError("event_prefill_artifact_already_exists")
    result = run_audited_question(
        item={key: row[key] for key in ("question", "question_id", "video_id")},
        perception_map=perception_map, video_path=video_path, duration_sec=duration,
        observer=Observer(controller.client), audit_runtime=runtime,
    )
    result["integration_stage"] = "before_first_hypothesis_verifier"
    result["handoff_complete"] = False
    result["perception_map_path"] = str(Path(map_path).resolve())
    controller.store.write_json(artifact, result)
    if result["status"] != "COMPLETED":
        raise RuntimeError("event_prefill_scan_or_audit_failed")
    coverage = whole_video_coverage(result, {"duration_sec": duration})
    units = []
    for record in result["evidence_rounds"]:
        receipt = record["transport_receipt"]
        if (not isinstance(receipt, dict)
                or receipt.get("producer") != "qwen_video_transport"
                or receipt.get("status") != "COMPLETED"
                or receipt.get("response_received") is not True
                or not receipt.get("source_units")):
            raise ValueError("event_prefill_completed_transport_receipt_required")
        units.extend(copy.deepcopy(receipt["source_units"]))
    coverage["source_unit_count"] = len(units)
    # Validation-only aggregation of original transport units. This is never
    # admitted as a new observation or used to invent source/claim identifiers.
    aggregate = Evidence(model_call={
        "observation_coverage": coverage,
        "full_video_coverage_reason": "User-selected Event 1fps initial coverage.",
        "observation_receipt": {
            "producer": "qwen_video_transport", "status": "COMPLETED",
            "response_received": True, "source_units": units,
        },
    })
    coverage = _validated_whole_video_coverage(aggregate, video_path=video_path)
    runtime._observation_coverage = coverage
    runtime._write_state("EVENT_PREFILL_COVERAGE", observation_coverage=coverage,
                         artifact_path=str(artifact))
    last = None
    for record in result["evidence_rounds"]:
        evidence = Evidence(**record["evidence"])
        plan = plan_from_dict(record["plan"])
        admission = record["audit_admission"]
        controller.bb.add_evidence(evidence)
        controller.store.write_json(controller.store.evidence_json(evidence.round_id),
                                    dataclasses.asdict(evidence))
        runtime.record_completed_observation_plan(plan, evidence, admission)
        controller.store.append_history({
            "event": "EVENT_PREFILL_AUDITED_ROUND", "round_id": evidence.round_id,
            "packet_id": record["packet"]["packet_id"], "admission": admission,
        })
        last = (plan, evidence, admission)
    result["handoff_complete"] = True
    result["validated_observation_coverage"] = coverage
    controller.store.write_json(artifact, result)
    return last
