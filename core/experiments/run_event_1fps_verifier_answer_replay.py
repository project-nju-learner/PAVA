"""Replay frozen Event-1fps audited Evidence through Verifier and Answer.

The replay is text-only: it never opens source media, reruns Observe, or loads
ground-truth labels.  One Hypothesis Verifier request and one forced Answer
request are made per supplied case.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from avp.main import Blackboard, Evidence, PlanSpec, SpatialTokenRate, WatchConfig
from avp.mcq_candidates import (
    apply_hypothesis_verifier_results,
    initialize_open_option_candidates,
)
from experiments.pollution_audit import format_mcq_query
from experiments.semantic_delta_pilot.avp_integration import (
    OpenAIJSONBackend,
    _response_text,
    _hypothesis_verifier_request,
    _parse_hypothesis_verifier_results,
)
from experiments.semantic_delta_pilot.admission_gate import (
    EVIDENCE_AUDIT_SCHEMA_VERSION,
    evidence_alignment_from_record,
)
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    make_dashscope_flash_client,
)
from experiments.semantic_delta_pilot.operator_gate import (
    OperatorGateValidationError,
    build_operator_gate_request,
    execute_operator_gate_response,
    failed_operator_gate_packet,
    is_input_too_large_error,
)


BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("PAVA_MODEL", "qwen3.8-flash")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-result", action="append", required=True)
    parser.add_argument("--qa-input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--operator-gate-mode",
        choices=("off", "shadow", "active"),
        default="off",
    )
    return parser


def load_case(
    shadow_path: Path,
    qa_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    shadow = read_json(shadow_path)
    qa = read_json(qa_path)
    if str(shadow.get("question_id")) != str(qa.get("question_id")):
        raise ValueError("shadow_and_qa_question_id_mismatch")
    if str(shadow.get("question")) != str(qa.get("question")):
        raise ValueError("shadow_and_qa_question_mismatch")
    if shadow.get("ground_truth_loaded") is not False:
        raise ValueError("shadow_ground_truth_must_be_unavailable")
    forbidden = {"answer", "ground_truth", "gold", "label"} & set(qa)
    if forbidden:
        raise ValueError(f"qa_input_contains_evaluation_fields:{sorted(forbidden)}")
    options = qa.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("qa_options_required")
    audit_summary = shadow.get("audit_summary")
    if not isinstance(audit_summary, Mapping):
        raise ValueError("shadow_audit_summary_required")
    raw_states = audit_summary.get("evidence_audit_states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError("shadow_evidence_audit_states_required")
    states = [dict(item) for item in raw_states if isinstance(item, Mapping)]
    if len(states) != len(raw_states):
        raise ValueError("shadow_evidence_state_must_be_object")
    return shadow, qa, states


def whole_video_coverage(
    shadow: Mapping[str, Any],
    qa: Mapping[str, Any],
) -> dict[str, Any]:
    if shadow.get("coverage_complete") is not True:
        raise ValueError("shadow_whole_video_coverage_required")
    if shadow.get("audit_complete") is not True:
        raise ValueError("shadow_complete_audit_required")
    if shadow.get("coverage_mode") != "all_events_one_fps_standard_observe":
        raise ValueError("shadow_one_fps_coverage_mode_required")
    packets = shadow.get("packet_plan")
    if not isinstance(packets, list) or not packets:
        raise ValueError("shadow_packet_plan_required")
    ordered = sorted(packets, key=lambda item: int(item["ordinal"]))
    previous_end = 0.0
    for index, packet in enumerate(ordered):
        start = float(packet["observation_start_sec"])
        end = float(packet["observation_end_sec"])
        if end <= start:
            raise ValueError("shadow_packet_scope_invalid")
        if index == 0:
            if abs(start) > 1e-3:
                raise ValueError("shadow_coverage_does_not_start_at_zero")
        elif abs(start - previous_end) > 1e-3:
            raise ValueError("shadow_coverage_has_timeline_gap")
        previous_end = end
    duration_sec = float(qa["duration_sec"])
    if abs(previous_end - duration_sec) > 0.02:
        raise ValueError("shadow_coverage_does_not_reach_video_end")
    if int(shadow.get("packet_count") or 0) != len(ordered):
        raise ValueError("shadow_packet_count_mismatch")
    if int(shadow.get("audited_round_count") or 0) != len(ordered):
        raise ValueError("shadow_audited_round_count_mismatch")
    return {
        "scope": "WHOLE_VIDEO",
        "sampling_fps": 1.0,
        "start_sec": 0.0,
        "end_sec": duration_sec,
        "coverage_complete": True,
        "source_unit_count": len(ordered),
    }


def normalized_evidence_states(
    states: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in states:
        item = dict(raw)
        item["schema_version"] = EVIDENCE_AUDIT_SCHEMA_VERSION
        item["source_alignment"] = evidence_alignment_from_record(item)
        item.pop("status", None)
        item.pop("verdict", None)
        normalized.append(item)
    return normalized


def replay_source_scopes(
    shadow: Mapping[str, Any],
    states: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Join claim sources to transport-owned scopes through round lineage."""

    raw_rounds = shadow.get("evidence_rounds")
    if not isinstance(raw_rounds, list):
        raise ValueError("shadow_rounds_required_for_source_scopes")
    scopes_by_round: dict[int, list[dict[str, Any]]] = {}
    for raw_round in raw_rounds:
        if not isinstance(raw_round, Mapping):
            continue
        try:
            round_id = int(raw_round["round_id"])
        except (KeyError, TypeError, ValueError):
            continue
        receipt = raw_round.get("transport_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("status") != "COMPLETED":
            continue
        producer = str(receipt.get("producer") or "").strip()
        request_id = str(receipt.get("request_id") or "").strip()
        units = receipt.get("source_units")
        if not producer or not request_id or not isinstance(units, list):
            continue
        scopes: list[dict[str, Any]] = []
        for unit in units:
            if not isinstance(unit, Mapping):
                continue
            try:
                start = float(unit["source_start_sec"])
                end = float(unit["source_end_sec"])
                fps = float(unit["submitted_fps"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or end <= start or fps <= 0:
                continue
            scopes.append(
                {
                    "producer": producer,
                    "transaction_id": request_id,
                    "media_identity": str(unit.get("source_video_path") or ""),
                    "sampling_fps": fps,
                    "start_sec": start,
                    "end_sec": end,
                }
            )
        if scopes:
            scopes_by_round[round_id] = scopes

    by_source: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        try:
            round_id = int(state.get("round_id"))
        except (TypeError, ValueError):
            continue
        for source_id in state.get("source_ids") or []:
            normalized_id = str(source_id).strip()
            if not normalized_id:
                continue
            for scope in scopes_by_round.get(round_id, []):
                item = {"source_id": normalized_id, **scope}
                if item not in by_source.setdefault(normalized_id, []):
                    by_source[normalized_id].append(item)
    return by_source


def latest_evidence_text(states: list[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for state in states:
        if state.get("latest") is False:
            continue
        status = evidence_alignment_from_record(state)
        claim_id = str(state.get("claim_id") or "unknown")
        claim = str(state.get("claim") or "").strip()
        source_ids = [str(value) for value in state.get("source_ids") or []]
        missing = [str(value) for value in state.get("missing_premises") or []]
        lines.append(
            f"[Evidence Audit {status}; claim_id={claim_id}; "
            f"source_ids={source_ids or ['none']}] {claim}"
        )
        if missing:
            lines.append(f"  unresolved_or_constraint: {' | '.join(missing)}")
    if not lines:
        raise ValueError("no_latest_evidence_states")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    if len(args.shadow_result) != len(args.qa_input):
        raise ValueError("shadow_result_and_qa_input_counts_must_match")
    case_paths = list(zip(args.shadow_result, args.qa_input))
    cases = [
        load_case(Path(shadow_path), Path(qa_path))
        for shadow_path, qa_path in case_paths
    ]
    coverages = [
        whole_video_coverage(shadow, qa)
        for shadow, qa, _ in cases
    ]
    gate_enabled = args.operator_gate_mode != "off"
    if gate_enabled:
        for shadow, _, states in cases:
            normalized = normalized_evidence_states(states)
            replay_source_scopes(shadow, normalized)
    planned_requests = len(cases) * (3 if gate_enabled else 2)
    if args.preflight:
        print(
            f"PREFLIGHT_OK cases={len(cases)} planned_requests={planned_requests} "
            f"operator_gate_mode={args.operator_gate_mode} media=0 gt_loaded=false"
        )
        return 0

    if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        raise RuntimeError("DASHSCOPE_API_KEY_required")
    output_root = Path(args.out).resolve()
    if output_root.exists():
        raise FileExistsError(f"output_already_exists:{output_root}")
    output_root.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": "event_1fps_verifier_answer_replay_v1",
        "status": "running",
        "started_at": utc_now(),
        "model": MODEL,
        "base_url": BASE_URL,
        "mode": "frozen_shadow_verifier_answer_text_replay",
        "planned_requests": planned_requests,
        "actual_requests": 0,
        "operator_gate_mode": args.operator_gate_mode,
        "operator_gate_requests": 0,
        "verifier_requests": 0,
        "answer_requests": 0,
        "planner_requests": 0,
        "observe_requests": 0,
        "evidence_audit_requests": 0,
        "challenge_requests": 0,
        "media_attachments": 0,
        "whole_video_coverage_declared": True,
        "cache_reuse_expected": False,
        "automatic_retry": False,
        "ground_truth_loaded": False,
        "samples": [],
    }
    write_json(output_root / "manifest.json", manifest)

    verifier_backend = OpenAIJSONBackend(
        base_url=BASE_URL,
        model=MODEL,
        trace_path=output_root / "verifier_model_calls.jsonl",
        temperature=args.temperature,
        seed=args.seed,
        max_requests=len(cases) * (2 if gate_enabled else 1),
    )
    answer_client = make_dashscope_flash_client(
        base_url=BASE_URL,
        model=MODEL,
        temperature=args.temperature,
        seed=args.seed,
    )
    original_answer_post = answer_client.client.post_chat_completion
    active_sample_dir: Path | None = None
    answer_request_count = 0
    operator_gate_request_count = 0
    verifier_request_count = 0

    def traced_answer_post(payload: Mapping[str, Any]) -> str:
        nonlocal answer_request_count
        if active_sample_dir is None:
            raise RuntimeError("active_sample_dir_missing")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("answer_messages_required")
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, list) and any(
                not isinstance(part, Mapping) or part.get("type") != "text"
                for part in content
            ):
                raise ValueError("answer_replay_must_be_text_only")
        if payload.get("enable_thinking") is not True:
            raise ValueError("answer_replay_must_enable_thinking")
        write_json(active_sample_dir / "answer_request.json", dict(payload))
        answer_request_count += 1
        manifest["answer_requests"] = answer_request_count
        manifest["actual_requests"] = verifier_backend.request_count + answer_request_count
        write_json(output_root / "manifest.json", manifest)
        response = original_answer_post(payload)
        (active_sample_dir / "answer_raw_response.txt").write_text(
            response,
            encoding="utf-8",
        )
        return response

    answer_client.client.post_chat_completion = traced_answer_post
    try:
        for (
            shadow,
            qa,
            evidence_states,
        ), observation_coverage in zip(cases, coverages):
            question_id = str(qa["question_id"])
            sample_dir = output_root / "samples" / question_id
            sample_dir.mkdir(parents=True)
            active_sample_dir = sample_dir
            started = time.monotonic()
            try:
                question = str(qa["question"])
                options = [str(value) for value in qa["options"]]
                base_candidates = initialize_open_option_candidates(options)
                evidence_states = normalized_evidence_states(evidence_states)
                operator_gate_packet: dict[str, Any] | None = None
                if gate_enabled:
                    source_scopes = replay_source_scopes(shadow, evidence_states)
                    gate_request = build_operator_gate_request(
                        question=question,
                        option_candidates=base_candidates,
                        evidence_states=evidence_states,
                        source_scopes=source_scopes,
                        sampling_coverage=observation_coverage,
                    )
                    try:
                        operator_gate_request_count += 1
                        gate_raw = verifier_backend.call("operator_gate", gate_request)
                        operator_gate_packet = execute_operator_gate_response(
                            text=_response_text(gate_raw),
                            evidence_revision=1,
                            evidence_states=evidence_states,
                            source_scopes=source_scopes,
                            sampling_coverage=observation_coverage,
                        )
                    except OperatorGateValidationError as exc:
                        operator_gate_packet = failed_operator_gate_packet(
                            evidence_revision=1,
                            evidence_states=evidence_states,
                            source_scopes=source_scopes,
                            sampling_coverage=observation_coverage,
                            status="INVALID",
                            reason_code=exc.reason_code,
                        )
                    except Exception as exc:
                        too_large = is_input_too_large_error(exc)
                        operator_gate_packet = failed_operator_gate_packet(
                            evidence_revision=1,
                            evidence_states=evidence_states,
                            source_scopes=source_scopes,
                            sampling_coverage=observation_coverage,
                            status="INPUT_TOO_LARGE" if too_large else "UNAVAILABLE",
                            reason_code=(
                                "INPUT_TOO_LARGE" if too_large else "GATE_UNAVAILABLE"
                            ),
                        )
                        operator_gate_packet["error_type"] = type(exc).__name__
                    write_json(
                        sample_dir / "operator_gate_packet.json",
                        operator_gate_packet,
                    )
                    manifest["operator_gate_requests"] = operator_gate_request_count
                verifier_request = _hypothesis_verifier_request(
                    question=question,
                    option_candidates=base_candidates,
                    evidence_states=evidence_states,
                    observation_coverage=observation_coverage,
                    operator_gate_packet=(
                        operator_gate_packet
                        if args.operator_gate_mode == "active"
                        else None
                    ),
                    allow_legacy_exact_count=args.operator_gate_mode != "active",
                )
                verifier_raw = verifier_backend.call(
                    "hypothesis_verifier",
                    verifier_request,
                )
                verifier_request_count += 1
                manifest["verifier_requests"] = verifier_request_count
                manifest["actual_requests"] = (
                    verifier_backend.request_count + answer_request_count
                )
                write_json(output_root / "manifest.json", manifest)
                verifier_text = _response_text(verifier_raw)
                verifier_results = _parse_hypothesis_verifier_results(
                    verifier_text,
                    option_candidates=base_candidates,
                    evidence_states=evidence_states,
                    question=question,
                    allow_legacy_exact_count=args.operator_gate_mode != "active",
                )
                option_candidates = apply_hypothesis_verifier_results(
                    base_candidates,
                    verifier_results,
                )
                write_json(
                    sample_dir / "verifier_result.json",
                    {
                        "question_id": question_id,
                        "observation_coverage": observation_coverage,
                        "results": verifier_results,
                        "option_candidates": option_candidates,
                        "operator_gate_mode": args.operator_gate_mode,
                        "operator_gate_packet": operator_gate_packet,
                    },
                )

                evidence_text = latest_evidence_text(evidence_states)
                blackboard = Blackboard(
                    video_path=f"frozen-shadow://{question_id}"
                )
                blackboard.meta["options"] = options
                blackboard.meta["option_candidates"] = option_candidates
                blackboard.meta["force_best_effort"] = True
                blackboard.meta["budget_forced_context"] = {
                    "source": "event_1fps_shadow_replay",
                    "verifier_results": verifier_results,
                }
                blackboard.evidences = [
                    Evidence(
                        detailed_response=evidence_text,
                        reasoning="",
                        key_evidence=[],
                        round_id=1,
                    )
                ]
                plan = PlanSpec(
                    plan_version="event_1fps_shadow_replay_v1",
                    query=format_mcq_query(question, options),
                    watch=WatchConfig(
                        load_mode="uniform",
                        fps=0.1,
                        spatial_token_rate=SpatialTokenRate.low,
                    ),
                    description="Synthesize only from frozen audited Evidence.",
                )
                answer = answer_client.synthesize_final_answer(plan, blackboard)
                selected = str(answer.get("selected_option") or "").strip().upper()
                valid = {
                    chr(ord("A") + index)
                    for index in range(len(options))
                }
                status = "completed" if selected in valid else "failed"
                record: dict[str, Any] = {
                    "status": status,
                    "question_id": question_id,
                    "selected_option": selected,
                    "selected_option_text": answer.get("selected_option_text"),
                    "answer": answer,
                    "verifier_results": verifier_results,
                    "option_candidates": option_candidates,
                    "answer_mode": "shadow_replay_forced",
                    "operator_gate_mode": args.operator_gate_mode,
                    "operator_gate_status": (
                        operator_gate_packet.get("status")
                        if operator_gate_packet is not None
                        else None
                    ),
                    "source_shadow_schema": shadow.get("schema_version"),
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "ground_truth_loaded": False,
                    "media_attachments": 0,
                }
                if status == "failed":
                    record["error"] = "answer_did_not_select_listed_option"
            except Exception as exc:
                record = {
                    "status": "failed",
                    "question_id": question_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "ground_truth_loaded": False,
                    "media_attachments": 0,
                }
            write_json(sample_dir / "result.json", record)
            manifest["samples"].append(record)
            write_json(output_root / "manifest.json", manifest)
    finally:
        answer_client.cleanup_uploaded_files()

    failed = sum(item["status"] != "completed" for item in manifest["samples"])
    manifest["status"] = "completed" if failed == 0 else "completed_with_failures"
    manifest["finished_at"] = utc_now()
    manifest["actual_requests"] = verifier_backend.request_count + answer_request_count
    manifest["verifier_requests"] = verifier_request_count
    manifest["answer_requests"] = answer_request_count
    manifest["operator_gate_requests"] = operator_gate_request_count
    manifest["sample_status_counts"] = {
        "completed": len(manifest["samples"]) - failed,
        "failed": failed,
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
