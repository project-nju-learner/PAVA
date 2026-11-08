from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from experiments.semantic_delta_pilot.admission_gate import (
    AdmissionAlignment,
    AuditRecord,
    DeltaPacket,
    EVIDENCE_AUDIT_SCHEMA_VERSION,
    SourceRecord,
    build_admission_request,
    derive_admission_alignment,
    evidence_alignment_from_record,
    freeze_audit_record,
    parse_admission_response,
)
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    MAX_VIDEO_FRAMES,
    MIN_VIDEO_FRAMES,
    SOURCE_SCOPE_TIME_TOLERANCE_SEC,
)
from experiments.semantic_delta_pilot.operator_gate import (
    OPERATOR_GATE_MODES,
    OperatorGateValidationError,
    build_operator_gate_request,
    execute_operator_gate_response,
    failed_operator_gate_packet,
    is_input_too_large_error,
)


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeltaSelection(StrictFrozenModel):
    delta_claim: str
    source_span_ids: tuple[str, ...]
    witness_request: str

    @model_validator(mode="after")
    def validate_all_or_none(self) -> "DeltaSelection":
        populated = bool(self.delta_claim.strip())
        if populated:
            if not self.source_span_ids or not self.witness_request.strip():
                raise ValueError("candidate_delta_requires_sources_and_witness")
            if len(set(self.source_span_ids)) != len(self.source_span_ids):
                raise ValueError("selector_source_ids_must_be_unique")
        elif self.source_span_ids or self.witness_request.strip():
            raise ValueError("empty_delta_must_not_carry_payload")
        return self


class JSONBackend(Protocol):
    request_count: int

    def call(self, stage: str, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _estimate_request_input_tokens(messages: list[Mapping[str, Any]]) -> int:
    """Cheap preflight estimate; the provider reports the exact count afterward."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized.encode("utf-8")) / 4))


@dataclass(frozen=True)
class SourceReceipt:
    record: SourceRecord
    transaction_id: str
    producer: str
    media_identity: str
    sampling_fps: float | None
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class OpenCheck:
    status: Literal["ACTIONABLE", "IRRELEVANT", "SATURATED"]
    diagnosis: str
    evidence_revision: int
    executed: bool = False


_OBSERVATION_NEED_EXECUTION_FIELDS = {
    "check",
    "completion_criteria",
    "end_sec",
    "fps",
    "load_mode",
    "method",
    "observation_request",
    "plan",
    "query",
    "regions",
    "sampling_fps",
    "scope",
    "start_sec",
    "strategy",
    "timestamps",
}


def _extract_observation_need_candidate(
    text: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Read one optional semantic need without weakening the result parser."""

    parsed = _json_object_with_minor_repair(text)
    if parsed is None:
        try:
            candidate = json.loads(text)
        except (TypeError, ValueError):
            candidate = None
        parsed = candidate if isinstance(candidate, Mapping) else None
    if parsed is None:
        return None, None
    if "observation_needs" in parsed:
        return None, "multiple_needs"
    if "observation_need" not in parsed:
        return None, None
    raw = parsed.get("observation_need")
    if raw is None:
        return None, None
    if isinstance(raw, (list, tuple)):
        return None, "multiple_needs"
    if not isinstance(raw, Mapping):
        return None, "malformed_output"
    return raw, None


def _validate_observation_need_candidate(
    raw: Mapping[str, Any],
    *,
    option_candidates: list[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Keep a Verifier need semantic; execution details belong to Planner."""

    keys = {str(key).strip() for key in raw}
    if keys & _OBSERVATION_NEED_EXECUTION_FIELDS:
        return None, "contains_execution_plan"
    if keys - {
        "missing_proposition",
        "affected_options",
        "distinguishes_options",
    }:
        return None, "malformed_output"
    proposition = str(raw.get("missing_proposition") or "").strip()
    raw_options = raw.get("affected_options")
    legacy_options = raw.get("distinguishes_options")
    if raw_options is None:
        raw_options = legacy_options
    elif legacy_options is not None:
        if not isinstance(legacy_options, (list, tuple)) or not isinstance(
            raw_options, (list, tuple)
        ):
            return None, "malformed_output"
        normalized_current = [
            str(item).strip().upper() for item in raw_options if str(item).strip()
        ]
        normalized_legacy = [
            str(item).strip().upper() for item in legacy_options if str(item).strip()
        ]
        if normalized_current != normalized_legacy:
            return None, "malformed_output"
    if not proposition or not isinstance(raw_options, (list, tuple)):
        return None, "malformed_output"
    allowed_options = {
        str(item.get("option") or "").strip().upper()
        for item in option_candidates
        if str(item.get("option") or "").strip()
    }
    options = list(
        dict.fromkeys(str(item).strip().upper() for item in raw_options if str(item).strip())
    )
    if (
        not options
        or len(options) > len(allowed_options)
        or any(option not in allowed_options for option in options)
    ):
        return None, "no_option_discriminant"
    if re.search(
        r"(?i)\b(?:ground[ -]?truth|official\s+answer|answer\s+key|internet|"
        r"web\s+search|outside\s+knowledge|external\s+source|model\s+confidence)\b",
        proposition,
    ):
        return None, "not_observable"
    return {
        "missing_proposition": proposition,
        "affected_options": options,
    }, None


def _extract_need_resolution_candidate(
    text: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Read the atomic resolution attached to one active SEARCH need."""

    parsed = _json_object_with_minor_repair(text)
    if parsed is None:
        try:
            candidate = json.loads(text)
        except (TypeError, ValueError):
            candidate = None
        parsed = candidate if isinstance(candidate, Mapping) else None
    if parsed is None or "need_resolution" not in parsed:
        return None, None
    raw = parsed.get("need_resolution")
    if not isinstance(raw, Mapping):
        return None, "malformed_need_resolution"
    return raw, None


def _validate_need_resolution_candidate(
    raw: Mapping[str, Any],
    *,
    active_need: Mapping[str, Any],
    option_candidates: list[Mapping[str, Any]],
    evidence_states: list[Mapping[str, Any]],
    results: list[Mapping[str, Any]],
    reused_claim_ids: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the finding only; results is the sole owner of option verdicts."""
    if not {"status", "finding", "claim_ids"}.issubset(raw):
        return None, "malformed_need_resolution"
    status = str(raw.get("status") or "").strip().upper()
    finding = str(raw.get("finding") or "").strip()
    ids = raw.get("claim_ids")
    if (status not in {"RESOLVED", "UNRESOLVED"} or not finding
            or not isinstance(ids, (list, tuple))
            or any(not isinstance(cid, str) or not cid.strip() for cid in ids)):
        return None, "malformed_need_resolution"
    claim_ids = list(dict.fromkeys(cid.strip() for cid in ids))
    if status == "RESOLVED" and not claim_ids:
        return None, "need_resolution_missing_evidence"
    eligible = {state.get("claim_id") for state in evidence_states
                if state.get("latest") is not False
                and evidence_alignment_from_record(state) == "ALIGNED"}
    if any(cid not in eligible for cid in claim_ids):
        return None, "need_resolution_ineligible_evidence"
    if reused_claim_ids is not None:
        if not reused_claim_ids or any(cid not in eligible for cid in reused_claim_ids):
            return None, "need_resolution_ineligible_reuse"
        if status == "RESOLVED" and not set(claim_ids).intersection(reused_claim_ids):
            return None, "need_resolution_missing_reused_evidence"
    # Historical extra metadata is ignored and never updates option states.
    return {"status": status, "finding": finding, "claim_ids": claim_ids}, None


@dataclass(frozen=True)
class PendingDelta:
    delta_id: str
    packet: DeltaPacket
    selection: DeltaSelection
    alignment: AdmissionAlignment
    permission: str
    revision: int
    source_root_ids: tuple[str, ...]
    observed_root_ids: tuple[str, ...]
    observed_transaction_ids: tuple[str, ...]
    open_check: OpenCheck | None = None


def _validated_whole_video_coverage(
    evidence: Any,
    *,
    video_path: str,
) -> dict[str, Any] | None:
    """Validate transport-owned whole-video sampling metadata, if present."""

    model_call = evidence.model_call if isinstance(evidence.model_call, dict) else {}
    raw = model_call.get("observation_coverage")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("observation_coverage_must_be_an_object")
    if str(raw.get("scope") or "") != "WHOLE_VIDEO":
        raise ValueError("observation_coverage_scope_must_be_whole_video")
    if raw.get("coverage_complete") is not True:
        raise ValueError("observation_coverage_must_be_complete")
    sampling_fps = float(raw["sampling_fps"])
    start_sec = float(raw["start_sec"])
    end_sec = float(raw["end_sec"])
    source_unit_count = int(raw["source_unit_count"])
    if (
        sampling_fps <= 0.1
        or abs(start_sec) > SOURCE_SCOPE_TIME_TOLERANCE_SEC
        or end_sec <= start_sec
    ):
        raise ValueError("invalid_whole_video_coverage_values")

    reason = str(model_call.get("full_video_coverage_reason") or "").strip()
    receipt = model_call.get("observation_receipt")
    if not reason or not isinstance(receipt, Mapping):
        raise ValueError("whole_video_coverage_authority_required")
    if (
        receipt.get("producer") != "qwen_video_transport"
        or receipt.get("status") != "COMPLETED"
        or receipt.get("response_received") is not True
    ):
        raise ValueError("whole_video_coverage_receipt_not_completed")
    units = receipt.get("source_units")
    if not isinstance(units, list) or len(units) != source_unit_count:
        raise ValueError("whole_video_coverage_unit_count_mismatch")

    expected_video = Path(video_path).resolve(strict=True)
    ordered: list[tuple[float, float, float]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            raise ValueError("whole_video_coverage_unit_must_be_an_object")
        source_video = Path(str(unit.get("source_video_path") or "")).resolve(
            strict=True
        )
        submitted_media = Path(
            str(unit.get("submitted_media_path") or "")
        ).resolve(strict=True)
        unit_start = float(unit["source_start_sec"])
        unit_end = float(unit["source_end_sec"])
        unit_fps = float(unit["fps"])
        submitted_start = float(unit["submitted_start_sec"])
        submitted_end = float(unit["submitted_end_sec"])
        submitted_fps = float(unit["submitted_fps"])
        transport = unit.get("transport_evidence")
        frame_count = (
            int(transport.get("frame_count") or 0)
            if isinstance(transport, Mapping)
            else 0
        )
        if (
            source_video != expected_video
            or submitted_media != expected_video
            or unit_end <= unit_start
            or abs(unit_fps - sampling_fps) > 1e-12
            or abs(submitted_start - unit_start) > SOURCE_SCOPE_TIME_TOLERANCE_SEC
            or abs(submitted_end - unit_end) > SOURCE_SCOPE_TIME_TOLERANCE_SEC
            or submitted_fps + 1e-12 < unit_fps
            or not isinstance(transport, Mapping)
            or transport.get("kind")
            != "direct_bounded_ffmpeg_base64_jpeg_frames"
            or not (MIN_VIDEO_FRAMES <= frame_count <= MAX_VIDEO_FRAMES)
        ):
            raise ValueError("whole_video_coverage_unit_not_authoritative")
        ordered.append((unit_start, unit_end, unit_fps))

    ordered.sort()
    cursor = start_sec
    for unit_start, unit_end, _unit_fps in ordered:
        if abs(unit_start - cursor) > SOURCE_SCOPE_TIME_TOLERANCE_SEC:
            raise ValueError("whole_video_coverage_has_gap_or_overlap")
        cursor = unit_end
    if abs(cursor - end_sec) > SOURCE_SCOPE_TIME_TOLERANCE_SEC:
        raise ValueError("whole_video_coverage_end_mismatch")
    return {
        "scope": "WHOLE_VIDEO",
        "sampling_fps": sampling_fps,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "coverage_complete": True,
        "source_unit_count": source_unit_count,
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _strict_json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("model_json_must_be_an_object")
    return value


class OpenAIJSONBackend:
    """One-shot JSON calls with append-only traces and no retries."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        trace_path: Path,
        temperature: float = 0.0,
        seed: int = 20260829,
        max_requests: int | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.trace_path = trace_path
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_requests = int(max_requests) if max_requests is not None else None
        if self.max_requests is not None and self.max_requests <= 0:
            raise ValueError("max_requests_must_be_positive")
        self.request_count = 0

    def call(self, stage: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.max_requests is not None and self.request_count >= self.max_requests:
            raise RuntimeError("admission_model_request_budget_exceeded")
        self.request_count += 1
        request_id = f"audit_request_{self.request_count:02d}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": copy.deepcopy(list(request["messages"])),
            "temperature": float(request.get("temperature", self.temperature)),
            "seed": self.seed,
            "max_tokens": int(request.get("max_tokens") or 1024),
            "stream": False,
        }
        enable_thinking = bool(request.get("enable_thinking", False))
        is_dashscope = "aliyuncs.com" in self.base_url.lower()
        if is_dashscope:
            payload["enable_thinking"] = enable_thinking
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if "response_format" in request:
            payload["response_format"] = copy.deepcopy(request["response_format"])
        _append_jsonl(
            self.trace_path,
            {
                "request_id": request_id,
                "stage": stage,
                "event": "REQUEST_FROZEN",
                "payload": payload,
                "estimated_input_tokens": request.get("estimated_input_tokens"),
            },
        )
        headers = {"Content-Type": "application/json"}
        api_key = str(os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        http_request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(http_request, timeout=1800) as response:
            envelope_text = response.read().decode("utf-8", errors="replace")
        envelope = json.loads(envelope_text)
        envelope_mapping = envelope if isinstance(envelope, dict) else {}
        choices = envelope_mapping.get("choices")
        first_choice = (
            choices[0]
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else {}
        )
        message = first_choice.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        finish_reason = first_choice.get("finish_reason")
        usage = envelope_mapping.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        completion_details = usage.get("completion_tokens_details")
        completion_details = (
            completion_details if isinstance(completion_details, dict) else {}
        )
        _append_jsonl(
            self.trace_path,
            {
                "request_id": request_id,
                "stage": stage,
                "event": "RESPONSE_ENVELOPE_SHAPE_FROZEN",
                "response_model": (
                    envelope_mapping.get("model")
                    if isinstance(envelope_mapping.get("model"), str)
                    else None
                ),
                "choice_count": len(choices) if isinstance(choices, list) else None,
                "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
                "message_keys": sorted(str(key) for key in message),
                "content_type": type(content).__name__,
                "content_length": len(content) if isinstance(content, str) else None,
                "reasoning_content_type": type(reasoning_content).__name__,
                "reasoning_content_length": (
                    len(reasoning_content) if isinstance(reasoning_content, str) else None
                ),
                "actual_prompt_tokens": usage.get("prompt_tokens"),
                "actual_completion_tokens": usage.get("completion_tokens"),
                "actual_reasoning_tokens": completion_details.get("reasoning_tokens"),
                "tool_calls_count": (
                    len(tool_calls) if isinstance(tool_calls, list) else 0
                    if tool_calls is None
                    else None
                ),
            },
        )
        if finish_reason == "length":
            _append_jsonl(
                self.trace_path,
                {"request_id": request_id, "stage": stage,
                 "event": "RESPONSE_FROZEN", "raw_content": content,
                 "parsed": None, "finish_reason": finish_reason},
            )
            raise RuntimeError("model_response_truncated")
        if not isinstance(content, str):
            raise TypeError("model_response_content_must_be_text")
        if not content.strip():
            raise ValueError("model_response_content_must_be_nonempty_text")
        response_mode = str(request.get("response_mode") or "json")
        parsed = (
            {"raw_text": content}
            if response_mode == "text"
            else _strict_json_object(content)
        )
        _append_jsonl(
            self.trace_path,
            {
                "request_id": request_id,
                "stage": stage,
                "event": "RESPONSE_FROZEN",
                "raw_content": content,
                "parsed": parsed,
            },
        )
        return parsed


def selector_request(
    *,
    question: str,
    candidate_text: str,
    candidate_reasoning: str,
    interaction_history: list[dict[str, Any]],
    current_confidence: float | None,
    source_records: tuple[SourceRecord, ...],
) -> dict[str, Any]:
    context = {
        "question": question,
        "candidate_text": candidate_text,
        "candidate_reasoning": candidate_reasoning,
        "interaction_history": interaction_history,
        "current_confidence": current_confidence,
        "available_source_records": [record.model_dump() for record in source_records],
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Identify at most one answer-critical semantic step added by the "
                    "candidate beyond the exact source records. Context may help choose "
                    "which step to inspect, but it is not evidence. Do not judge whether "
                    "the step is supported. Select only a video-observable factual claim "
                    "or relation, never answer correctness, option preference, relevance, "
                    "or whether something is the main story. A delta is answer-critical only "
                    "if resolving it could change the answer. Its witness_request must name the "
                    "smallest concrete unchecked video observation that would resolve the factual "
                    "gap; do not name or compare answer options here. Copy only complete source_id "
                    "values exactly as provided; never use "
                    "a locator or bracketed raw time span as a source ID. A non-empty delta_claim "
                    "MUST include at least one source_span_id and a non-empty witness_request. "
                    "Otherwise return an empty delta_claim, empty source_span_ids, and empty "
                    "witness_request."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, sort_keys=True),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "delta_selection",
                "strict": True,
                "schema": DeltaSelection.model_json_schema(),
            },
        },
    }


def extract_source_receipts(
    evidence: Any,
    *,
    round_id: int,
    video_path: str,
    preserve_response_sources: bool = False,
) -> tuple[SourceReceipt, ...]:
    model_call = evidence.model_call if isinstance(evidence.model_call, dict) else {}
    returned_clips = model_call.get("returned_clips")
    if isinstance(returned_clips, list):
        transaction_id = str(model_call.get("retrieval_call_id") or "").strip()
        producer = str(model_call.get("producer") or "").strip()
        if not transaction_id or producer != "longvidsearch_caption_observer":
            raise ValueError("caption_receipt_lineage_required")
        receipts: list[SourceReceipt] = []
        for clip in returned_clips:
            if not isinstance(clip, dict):
                raise ValueError("caption_receipt_must_be_an_object")
            clip_id = str(clip.get("clip_id") or "").strip()
            video_id = str(clip.get("video_id") or "").strip()
            caption = str(clip.get("caption") or "").strip()
            start = float(clip["nominal_start_sec"])
            end = float(clip["nominal_end_sec"])
            if not clip_id or not video_id or not caption or end <= start:
                raise ValueError("invalid_caption_receipt")
            record = SourceRecord(
                source_id=clip_id,
                provenance_root_id=f"longvidsearch:{video_id}:{clip_id}",
                source_kind="caption",
                exact_caption=caption,
                locator=f"nominal[{start:.0f},{end:.0f})",
            )
            receipts.append(
                SourceReceipt(
                    record=record,
                    transaction_id=transaction_id,
                    producer=producer,
                    media_identity=f"longvidsearch:{video_id}",
                    sampling_fps=None,
                    start_sec=start,
                    end_sec=end,
                )
            )
        return _deduplicate_receipts(receipts)

    frames = evidence.frames_used if isinstance(evidence.frames_used, list) else []
    normalized_frames: list[tuple[float, float, float | None]] = []
    for frame in frames:
        if not isinstance(frame, dict) or "start" not in frame or "end" not in frame:
            continue
        start, end = float(frame["start"]), float(frame["end"])
        raw_fps = frame.get("fps")
        fps = float(raw_fps) if raw_fps is not None else None
        if end > start and (fps is None or fps > 0):
            normalized_frames.append((start, end, fps))
    if not normalized_frames:
        raise ValueError("raw_observation_scope_required")

    authority_receipt = model_call.get("observation_receipt")
    if not isinstance(authority_receipt, dict):
        raise ValueError("authority_owned_raw_observation_receipt_required")
    producer = str(authority_receipt.get("producer") or "").strip()
    transaction_id = str(authority_receipt.get("request_id") or "").strip()
    if (
        producer != "qwen_video_transport"
        or not transaction_id
        or authority_receipt.get("status") != "COMPLETED"
        or authority_receipt.get("response_received") is not True
    ):
        raise ValueError("invalid_raw_observation_transport_receipt")
    source_units = authority_receipt.get("source_units")
    if not isinstance(source_units, list) or len(source_units) != len(normalized_frames):
        raise ValueError("raw_observation_unit_count_mismatch")

    expected_video = Path(video_path).resolve(strict=True)
    validated_units: list[dict[str, Any]] = []
    unmatched_frames = list(normalized_frames)
    for unit in source_units:
        if not isinstance(unit, dict):
            raise ValueError("raw_observation_unit_must_be_an_object")
        source_video = Path(str(unit.get("source_video_path") or "")).resolve(
            strict=True
        )
        submitted_media = Path(
            str(unit.get("submitted_media_path") or "")
        ).resolve(strict=True)
        if source_video != expected_video or not submitted_media.is_file():
            raise ValueError("raw_observation_media_identity_mismatch")

        start = float(unit["source_start_sec"])
        end = float(unit["source_end_sec"])
        raw_fps = unit.get("fps")
        fps = float(raw_fps) if raw_fps is not None else None
        submitted_start = float(unit["submitted_start_sec"])
        submitted_end = float(unit["submitted_end_sec"])
        submitted_raw_fps = unit.get("submitted_fps")
        submitted_fps = (
            float(submitted_raw_fps) if submitted_raw_fps is not None else None
        )
        if start < 0 or end <= start or submitted_start < 0 or submitted_end <= submitted_start:
            raise ValueError("invalid_raw_observation_transport_scope")
        transport = unit.get("transport_evidence")
        if not isinstance(transport, dict):
            raise ValueError("raw_video_transport_evidence_required")
        transport_kind = transport.get("kind")
        if (
            transport_kind != "direct_bounded_ffmpeg_base64_jpeg_frames"
            and fps != submitted_fps
        ):
            raise ValueError("raw_observation_transport_fps_mismatch")
        if transport_kind == "local_http_get":
            transport_valid = (
                int(transport.get("get_count") or 0) > 0
                and int(transport.get("bytes_served") or 0) > 0
            )
        elif transport_kind == "dashscope_temporary_oss_video":
            submitted_uri = str(unit.get("submitted_media_uri") or "")
            transport_valid = (
                submitted_uri.startswith("oss://")
                and transport.get("upload_completed") is True
                and transport.get("provider_response_received") is True
                and int(transport.get("frame_count") or 0) > 0
            )
        elif transport_kind == "direct_bounded_ffmpeg_base64_jpeg_frames":
            frame_count = int(transport.get("frame_count") or 0)
            transport_valid = (
                MIN_VIDEO_FRAMES <= frame_count <= MAX_VIDEO_FRAMES
                and submitted_media == source_video
                and submitted_fps is not None
                and fps is not None
                and submitted_fps + 1e-12 >= fps
                and abs(submitted_start - start)
                <= SOURCE_SCOPE_TIME_TOLERANCE_SEC
                and abs(submitted_end - end)
                <= SOURCE_SCOPE_TIME_TOLERANCE_SEC
            )
        else:
            transport_valid = False
        if not transport_valid:
            raise ValueError("raw_video_transport_was_not_completed")

        match_index = next(
            (
                index
                for index, frame in enumerate(unmatched_frames)
                if abs(frame[0] - start) <= SOURCE_SCOPE_TIME_TOLERANCE_SEC
                and abs(frame[1] - end) <= SOURCE_SCOPE_TIME_TOLERANCE_SEC
                and frame[2] == fps
            ),
            None,
        )
        if match_index is None:
            raise ValueError("raw_receipt_does_not_match_observer_scope")
        unmatched_frames.pop(match_index)
        effective_fps = submitted_fps if submitted_fps is not None else fps
        fps_token = (
            "provider_default" if effective_fps is None else f"{effective_fps:.6g}"
        )
        locator = f"raw_source[{start:.3f},{end:.3f})@fps={fps_token}"
        validated_units.append(
            {
                "start": start,
                "end": end,
                "fps": effective_fps,
                "locator": locator,
                "root": f"raw_video:{source_video}@{locator}",
                "statements": [],
            }
        )
    if unmatched_frames:
        raise ValueError("raw_receipt_does_not_cover_observer_scope")

    for item in evidence.key_evidence:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        try:
            start = float(item["timestamp_start"])
            end = float(item["timestamp_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            raise ValueError("invalid_observation_statement_range")
        matching_units = [
            unit
            for unit in validated_units
            if start >= math.floor(unit["start"]) - 1e-6
            and end <= math.ceil(unit["end"]) + 1e-6
        ]
        if not matching_units:
            raise ValueError("observation_outside_actual_scope")
        owner = min(
            matching_units,
            key=lambda unit: (unit["end"] - unit["start"], unit["locator"]),
        )
        actual_start = max(start, owner["start"])
        actual_end = min(end, owner["end"])
        if actual_end <= actual_start:
            raise ValueError("observation_statement_has_no_actual_scope_intersection")
        if abs(actual_start - start) <= 1e-6 and abs(actual_end - end) <= 1e-6:
            scope_label = f"raw[{actual_start:.3f},{actual_end:.3f})"
        else:
            scope_label = (
                f"observer_reported_raw[{start:.3f},{end:.3f});"
                f"actual_scope_intersection[{actual_start:.3f},{actual_end:.3f})"
            )
        owner["statements"].append(f"[{scope_label}] {description}")

    receipts: list[SourceReceipt] = []
    for index, unit in enumerate(validated_units, 1):
        if not unit["statements"] and not preserve_response_sources:
            continue
        record = SourceRecord(
            source_id=(
                f"raw_range_{unit['start']:.3f}_{unit['end']:.3f}_fps_"
                + ("provider_default" if unit["fps"] is None else f"{unit['fps']:.6g}")
            ),
            provenance_root_id=unit["root"],
            source_kind="observer_claim",
            exact_caption="\n".join(unit["statements"]) or "Source receipt only; full response retained separately, no atomic fact admitted.",
            locator=unit["locator"],
        )
        receipts.append(
            SourceReceipt(
                record=record,
                transaction_id=transaction_id,
                producer=producer,
                media_identity=f"raw_video:{expected_video}",
                sampling_fps=unit["fps"],
                start_sec=unit["start"],
                end_sec=unit["end"],
            )
        )
    if not receipts:
        raise ValueError("no_grounded_observation_statement")
    return _deduplicate_receipts(receipts)


def _deduplicate_receipts(receipts: list[SourceReceipt]) -> tuple[SourceReceipt, ...]:
    by_root: dict[str, SourceReceipt] = {}
    for receipt in receipts:
        root = receipt.record.provenance_root_id
        prior = by_root.get(root)
        if prior is not None and (
            prior.record.exact_caption != receipt.record.exact_caption
            or prior.record.locator != receipt.record.locator
        ):
            raise ValueError(f"conflicting_receipt_for_root:{root}")
        by_root[root] = prior or receipt
    return tuple(by_root[root] for root in sorted(by_root))


_TRANSPORT_ONLY_SOURCE_TEXT = (
    "Raw-video transport receipt only; semantic statements produced by the same "
    "observation transaction are not admitted as source."
)


def _transport_only_receipts(
    receipts: tuple[SourceReceipt, ...],
) -> tuple[SourceReceipt, ...]:
    return tuple(
        SourceReceipt(
            record=SourceRecord(
                source_id=receipt.record.source_id,
                provenance_root_id=receipt.record.provenance_root_id,
                source_kind="transport_receipt",
                exact_caption=_TRANSPORT_ONLY_SOURCE_TEXT,
                locator=receipt.record.locator,
            ),
            transaction_id=receipt.transaction_id,
            producer=receipt.producer,
            media_identity=receipt.media_identity,
            sampling_fps=receipt.sampling_fps,
            start_sec=receipt.start_sec,
            end_sec=receipt.end_sec,
        )
        for receipt in receipts
    )


def _adds_new_raw_observation(
    candidate: SourceReceipt,
    previous: tuple[SourceReceipt, ...],
) -> bool:
    """A replay is new only when it reads a disjoint range or denser frames."""

    for prior in previous:
        if prior.media_identity != candidate.media_identity:
            continue
        overlap = min(prior.end_sec, candidate.end_sec) - max(
            prior.start_sec,
            candidate.start_sec,
        )
        if overlap <= 1e-6:
            continue
        if (
            prior.sampling_fps is not None
            and candidate.sampling_fps is not None
            and candidate.sampling_fps > prior.sampling_fps
        ):
            continue
        return False
    return True


def _build_answer_binding_claim(
    *,
    final: Mapping[str, Any],
    options: list[str] | None,
    question: str | None = None,
) -> tuple[str | None, str | None]:
    """Validate the final answer shape and return only its selected text.

    This helper performs ordinary MCQ label/text binding.  Its return value is
    never an Evidence claim and must not be sent to the Admission Gate.
    """
    candidate_text = str(
        final.get("selected_option_text")
        or (final.get("reasoning") if not options else "")
        or ""
    ).strip()
    if options:
        label = str(final.get("selected_option") or "").strip().upper()
        if len(label) != 1 or not ("A" <= label <= "Z"):
            return None, "selected_option_label_required"
        index = ord(label) - ord("A")
        if index >= len(options):
            return None, "selected_option_out_of_range"
        exact_option = str(options[index]).strip()
        semantic_option = exact_option
        for prefix in (f"{label}. ", f"{label}) ", f"{label}: "):
            if semantic_option.startswith(prefix):
                semantic_option = semantic_option[len(prefix):].strip()
                break
        if not exact_option or candidate_text not in {exact_option, semantic_option}:
            return None, "selected_option_text_must_match_exact_option"
        if not str(question or "").strip():
            return None, "question_required_for_mcq_answer_binding"
        return semantic_option, None
    if not candidate_text:
        return None, "free_form_answer_text_required"
    return candidate_text, None


def _response_text(raw: Mapping[str, Any]) -> str:
    return str(raw.get("raw_text") or raw.get("content") or "").strip()


def _negative_observation_request(
    *,
    observation_goal: str,
    observed_scopes: list[dict[str, float]],
    detailed_response: str,
) -> dict[str, Any]:
    payload = {
        "observation_goal": observation_goal,
        "observed_scopes": observed_scopes,
        "observer_commentary": detailed_response,
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Perform one narrow semantic routing check. Decide only whether the "
                    "Observer commentary explicitly reports that it inspected the submitted "
                    "scope but did not find the requested observation target. Do not infer "
                    "absence merely because the target is not mentioned. Do not extract "
                    "positive evidence, judge answer options, or decide whether the negative "
                    "observation is visually true; a later Evidence Audit performs that check. "
                    "Return exactly one line: NEGATIVE_OBSERVATION: <one concise scoped "
                    "negative observation>, or NOT_USABLE."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        "response_mode": "text",
        "max_tokens": 160,
    }


def _parse_negative_observation(raw: Mapping[str, Any]) -> str | None:
    text = _response_text(raw)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if re.fullmatch(r"(?i)NOT[ _-]?USABLE[.!]?", first_line):
        return None
    match = re.fullmatch(
        r"(?is)NEGATIVE[ _-]?OBSERVATION\s*:\s*(.+)",
        first_line,
    )
    if match is None:
        return None
    observation = match.group(1).strip()
    return observation or None


def _observed_scope_payload(evidence: Any) -> list[dict[str, float]]:
    scopes: list[dict[str, float]] = []
    for frame in getattr(evidence, "frames_used", []) or []:
        if not isinstance(frame, Mapping):
            continue
        try:
            start = float(frame["start"])
            end = float(frame["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start >= 0 and end > start:
            scopes.append({"start_sec": start, "end_sec": end})
    return scopes


def _with_scoped_negative_observation(evidence: Any, observation: str) -> Any:
    clone = copy.deepcopy(evidence)
    clone.key_evidence = [
        {
            "timestamp_start": scope["start_sec"],
            "timestamp_end": scope["end_sec"],
            "description": observation,
            "scope_precision": "submitted_negative_observation",
        }
        for scope in _observed_scope_payload(evidence)
    ]
    model_call = copy.deepcopy(
        clone.model_call if isinstance(clone.model_call, dict) else {}
    )
    model_call["negative_observation_fallback"] = {
        "source": "detailed_response",
        "classification": "NEGATIVE_OBSERVATION",
    }
    clone.model_call = model_call
    return clone


def _json_object_with_minor_repair(text: str) -> Mapping[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    attempts = [candidate]
    if candidate.count("{") > candidate.count("}"):
        attempts.append(candidate + ("}" * (candidate.count("{") - candidate.count("}"))))
    for attempt in attempts:
        attempt = re.sub(r",\s*([}\]])", r"\1", attempt)
        try:
            value = json.loads(attempt)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _balanced_json_mapping_fragments(text: str) -> list[Mapping[str, Any]]:
    """Recover complete JSON objects without guessing missing semantic fields."""
    fragments: list[Mapping[str, Any]] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            fragments.append(value)
    return fragments


def _option_label(value: str, options: list[str]) -> str | None:
    candidate = str(value or "").strip()
    match = re.fullmatch(r"(?i)(?:option\s*)?([A-Z])(?:[.):\s].*)?", candidate)
    if match:
        index = ord(match.group(1).upper()) - ord("A")
        if 0 <= index < len(options):
            return chr(ord("A") + index)
    for index, option in enumerate(options):
        exact = str(option).strip()
        semantic = exact
        label = chr(ord("A") + index)
        for prefix in (f"{label}. ", f"{label}) ", f"{label}: "):
            if semantic.startswith(prefix):
                semantic = semantic[len(prefix):].strip()
                break
        if candidate.casefold() in {exact.casefold(), semantic.casefold()}:
            return label
    return None


def _hypothesis_verifier_request(
    *,
    question: str,
    option_candidates: list[Mapping[str, Any]],
    evidence_states: list[Mapping[str, Any]],
    observation_coverage: Mapping[str, Any] | None = None,
    active_search_need: Mapping[str, Any] | None = None,
    operator_gate_packet: Mapping[str, Any] | None = None,
    allow_legacy_exact_count: bool = True,
    need_answers: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered_evidence_states: list[dict[str, Any]] = []
    for state in evidence_states:
        item = dict(state)
        item["schema_version"] = EVIDENCE_AUDIT_SCHEMA_VERSION
        item["source_alignment"] = evidence_alignment_from_record(item)
        item.pop("status", None)
        item.pop("verdict", None)
        item["latest"] = item.get("latest") is not False
        item["updates_claim_ids"] = list(item.get("updates_claim_ids") or [])
        ordered_evidence_states.append(item)

    def evidence_order(item: Mapping[str, Any]) -> tuple[int, int, str]:
        try:
            round_order = int(item.get("round_id"))
        except (TypeError, ValueError):
            round_order = -1
        return (
            0 if item["latest"] else 1,
            round_order,
            str(item.get("claim_id") or ""),
        )

    ordered_evidence_states.sort(key=evidence_order)
    payload: dict[str, Any] = {
        "question": question,
        "option_hypotheses": option_candidates,
        "evidence_audit_states": ordered_evidence_states,
    }
    if need_answers is not None:
        payload["need_answers"] = copy.deepcopy(need_answers)
    if observation_coverage is not None:
        payload["observation_coverage"] = dict(observation_coverage)
    if active_search_need is not None:
        payload["active_search_need"] = {
            "need_id": str(active_search_need.get("need_id") or ""),
            "revision": int(active_search_need.get("revision") or 1),
            "missing_proposition": str(
                active_search_need.get("missing_proposition") or ""
            ),
            "affected_options": list(
                active_search_need.get("affected_options") or []
            ),
        }
    if operator_gate_packet is not None:
        payload["operator_gate_packet"] = copy.deepcopy(operator_gate_packet)
    legacy_exact_count_guidance = (
        "For an exact-count hypothesis, if complete whole-video coverage and the "
        "ALIGNED claims establish the stated distinct items, do not keep it OPEN "
        "solely because an unseen video interval might contain another item; keep it "
        "OPEN for any unresolved identity, distinctness, duplication, or count link. "
        "For an exact-count question about different values, views, categories, or "
        "items mentioned, first form the deduplicated set established by ALIGNED "
        "claims, then compare its cardinality with each numeric option. Count an "
        "explicitly mentioned proposed, disputed, hypothetical, or questioned item "
        "when the question asks what is mentioned; do not exclude it merely because "
        "it is not established as true, conventional, or accepted. Do not double-count "
        "repeated mentions of the same item. When the question asks how many and an "
        "option is numeric, that option denotes the set cardinality, not a set member; "
        "do not require the numeric option itself to enumerate the members or confuse "
        "an item's numeric label with the count. If the supported set has cardinality "
        "N, support the numeric option N and contradict other numeric cardinalities. "
        "Admit a set member only when a cited ALIGNED claim explicitly binds it as "
        "a distinct value, view, category, or item along the dimension asked by the "
        "question. Topical background, historical states, names, components, or related "
        "facts do not create another member merely because they concern the same subject. "
        "Do not infer an extra member from relevance or bundle several background facts "
        "into an invented view. For a how-many question whose answer options are numeric, "
        "also return exact_count_summary with exactly one members list. Each member must "
        "contain member and claim_ids, use a short normalized label for one distinct set "
        "member, and cite every latest ALIGNED claim needed to bind that member. Do not "
        "put source_ids in exact_count_summary; they are projected from claim_ids. A "
        "SUPPORTED count verdict must cite every claim needed to establish the set "
        "members used in the count. "
        if allow_legacy_exact_count
        else ""
    )
    operator_gate_guidance = (
        "operator_gate_packet is a soft, question-specific computation over the supplied "
        "audited claims; it is not a new Evidence claim and it does not hide any raw "
        "Evidence state. When status is APPLIED, treat its deterministic SET, COUNT, or "
        "source-receipt TEMPORAL calculation as frozen: do not reselect operands, "
        "deduplicate, recount, or reorder them. Use its claim bindings and calculation "
        "when evaluating and mapping complete hypotheses to options. If "
        "operand_resolution_complete is false, the unresolved operands remain an explicit "
        "gap. BYPASS means ordinary direct semantic entailment may still be used. "
        "UNHANDLED means the fixed Gate menu does not cover the needed operation. INVALID, "
        "UNAVAILABLE, or INPUT_TOO_LARGE means no Gate calculation exists; never interpret "
        "that absence as zero, an empty set, or proof of any option. projection_closure "
        "only says whether the current Evidence Store claims were accounted for and never "
        "proves that all relevant video evidence was discovered. sampling_coverage is "
        "transport scope metadata and does not make a semantic claim true. "
        if operator_gate_packet is not None
        else ""
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "For receipt_only evidence, ALIGNED qualifies a local Observe fact based on "
                "its actual source binding, not independent visual confirmation or answer sufficiency. "
                "Use these local facts as premises, preserving observation_uncertainty and source_scopes. "
                "Do not extrapolate a sampled local claim into a whole-video theme, exhaustive count, "
                "or absence claim. Assess the complete evidence set against the hypothesis scope. "
                "Conflicting local facts remain unresolved; do not silently pick one. "
                "You are the Hypothesis Verifier for a video MCQ. Options are "
                "source-free hypotheses, never Evidence. For each option, interpret the "
                "complete hypothesis formed by the question together with that proposed "
                "answer; never judge the option text in isolation. Assess that hypothesis "
                "only against the supplied Evidence Audit states. Internally identify its "
                "material factual commitments, including any scope, main-story, all/common, "
                "counting, identity, temporal, causal, purpose, or comparison commitment "
                "introduced by either the question or the option. Do not return a new "
                "required-claims schema. SUPPORTED means the cited ALIGNED Evidence "
                "collectively covers every material commitment of the complete hypothesis. "
                "A locally true detail, shared subfact, or isolated stage is only partial "
                "coverage and must remain OPEN when a material commitment is unresolved. "
                "For main-story hypotheses, assess a coherent answer-specific temporal or causal arc; "
                "complete whole-video dense coverage is not automatically required. "
                "Judge scope from the actual observations and their support relation, not "
                "the number of claims or source ranges. Explain material gaps rather than "
                "requiring a claim to repeat the question's wording. A grounded counterexample "
                "can contradict a universal claim. CONTRADICTED means "
                "the cited ALIGNED Evidence establishes an incompatible fact for at least "
                "one material commitment; missing or incomplete support alone is not a "
                "contradiction. Otherwise return OPEN. UNCERTAIN Evidence is an unresolved gap "
                "and cannot support a hypothesis; CONFLICTING Evidence cannot be used as an "
                "established premise. Do not observe the video, rewrite Evidence source alignment, "
                "use outside knowledge, compare which option is best, or select a final "
                "answer. Evidence states are ordered with every latest=true thread head "
                "first, followed by latest=false audit history. Only latest=true states are "
                "the current semantic state. Historical states may explain progression but "
                "must not override or support a verdict. observation_coverage, when present, "
                "is transport-owned scope "
                "metadata, not semantic Evidence: it may close only the possibility of an "
                "unseen source interval. It never proves that an Evidence claim is true. "
                + legacy_exact_count_guidance
                + operator_gate_guidance
                + "In brief_reason, state the covered material commitments for "
                "SUPPORTED, the specific incompatible commitment for CONTRADICTED, or the "
                "specific unresolved commitment for OPEN. Return one result per option with "
                "option, verdict "
                "(SUPPORTED, OPEN, CONTRADICTED, or UNCHECKED), claim_ids, source_ids, and a "
                "brief_reason. JSON is preferred, but labeled lines are accepted. A decisive "
                "verdict must identify the ALIGNED Evidence claim_ids it relies on; cite "
                "only source_ids attached to those claims. Prefer one JSON object. Return "
                "exactly one final tuple per option inside results and one optional "
                "observation_need field. If proposing a new need, observation_need contains "
                "missing_proposition and affected_options. missing_proposition states "
                "one concrete video-observable variable or outcome whose observation can "
                "change at least one option verdict; never merely say to find the correct "
                "answer or continue watching. Keep it neutral, without the original question, "
                "option labels or answer judgments. affected_options lists every supplied option "
                "whose verdict can change, from one option through all supplied options. "
                "It must not prescribe timestamps, ranges, fps, scanning, "
                "sampling, OCR, chunking, or any other execution plan; Planner owns those "
                "choices. "
                + ("Local Audit owns need completion. Read need_answers and their source quotations, "
                   "remaining gaps and failed searches before judging the global options. "
                   "Do not return need_resolution. Judge options from the supplied tool evidence. "
                   "If no useful different need remains, observation_need may be null. "
                   "Do not repeat a limited need without a concrete new observable distinction. "
                   if need_answers is not None else
                "If active_search_need is absent and at least one option is "
                "SUPPORTED, observation_need must be null. When active_search_need is "
                "present, do not replace it merely because one option has some supporting "
                "content. Return need_resolution with status, finding, and claim_ids. "
                "RESOLVED means the cited facts answer the missing proposition; otherwise "
                "use UNRESOLVED and name the remaining gap. Cite valid latest ALIGNED facts "
                "from any round. An UNRESOLVED report may cite no facts when nothing relevant "
                "was found. Option verdicts belong only in results and remain independent "
                "of need completion. ")
                +
                "Never return multiple needs. Legacy labeled tuples remain accepted, but do "
                "not add prose outside the final tuples. Do not show deliberation, reconsideration, "
                "alternatives, or drafts. If support is incomplete or you are uncertain, "
                "output OPEN immediately. Each results[].brief_reason must be one sentence of at most "
                "30 words. Never emit two verdicts for the same option."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]
    estimated_input_tokens = _estimate_request_input_tokens(messages)
    return {
        "response_mode": "text",
        "max_tokens": 8192,
        "messages": messages,
        "estimated_input_tokens": estimated_input_tokens,
        "enable_thinking": False,
    }


def _observation_need_recovery_request(
    *,
    question: str,
    option_candidates: list[Mapping[str, Any]],
    verifier_results: list[Mapping[str, Any]],
    rejection_reason: str,
) -> dict[str, Any]:
    """Request one semantic need without reopening frozen option verdicts."""

    payload = {
        "question": question,
        "option_hypotheses": option_candidates,
        "frozen_hypothesis_results": verifier_results,
        "inline_need_rejection": rejection_reason,
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Recover exactly one missing observation need for a video MCQ whose "
                "Hypothesis Verifier results are frozen and contain zero SUPPORTED "
                "options. Never change, repeat, rank, or reconsider the option verdicts. "
                "Return one JSON object with exactly one observation_need field. That "
                "field contains exactly missing_proposition and affected_options. "
                "missing_proposition names one concrete video-observable variable or "
                "outcome that could change at least one frozen option verdict; do not "
                "merely ask for the correct answer. Keep it neutral, without the original question, "
                "option labels or answer judgments. affected_options lists every supplied "
                "option label whose verdict could change, from one through all options. "
                "Do not prescribe timestamps, ranges, fps, scanning, sampling, OCR, "
                "chunking, tools, or any execution plan. Do not use outside knowledge."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]
    return {
        "response_mode": "text",
        "max_tokens": 512,
        "messages": messages,
        "estimated_input_tokens": _estimate_request_input_tokens(messages),
        "enable_thinking": False,
    }


def _parse_hypothesis_verifier_results(
    text: str,
    *,
    option_candidates: list[Mapping[str, Any]],
    evidence_states: list[Mapping[str, Any]],
    question: str = "",
    allow_legacy_exact_count: bool = True,
) -> list[dict[str, Any]]:
    def string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[,;|]", value)
        if not isinstance(value, (list, tuple)):
            return []
        return list(
            dict.fromkeys(str(item).strip() for item in value if str(item).strip())
        )

    option_order = {
        str(item.get("option") or "").strip().upper(): index
        for index, item in enumerate(option_candidates)
    }
    allowed_options = set(option_order)
    states_by_claim = {
        str(state.get("claim_id") or "").strip(): state
        for state in evidence_states
        if str(state.get("claim_id") or "").strip()
    }
    aligned_states = {
        claim_id: state
        for claim_id, state in states_by_claim.items()
        if evidence_alignment_from_record(state) == "ALIGNED"
        and state.get("latest") is not False
    }
    aligned_claims_by_source: dict[str, list[str]] = {}
    for claim_id, state in aligned_states.items():
        for source_id in state.get("source_ids", []):
            normalized_source = str(source_id).strip()
            if normalized_source:
                aligned_claims_by_source.setdefault(normalized_source, []).append(claim_id)
    allowed_sources = set(aligned_claims_by_source)
    question_stem = re.split(r"(?im)^\s*options\s*:", str(question or ""), maxsplit=1)[0]
    raw_items: list[Mapping[str, Any]] = []
    parsed = _json_object_with_minor_repair(text)
    if parsed is not None:
        candidate_items = (
            parsed.get("results")
            or parsed.get("hypotheses")
            or parsed.get("candidate_results")
        )
        if isinstance(candidate_items, list):
            raw_items = [item for item in candidate_items if isinstance(item, Mapping)]
    else:
        try:
            candidate_items = json.loads(text)
        except (TypeError, ValueError):
            candidate_items = None
        if isinstance(candidate_items, list):
            raw_items = [item for item in candidate_items if isinstance(item, Mapping)]
    if not raw_items:
        recovered_items: list[Mapping[str, Any]] = []
        for fragment in _balanced_json_mapping_fragments(text):
            candidate_items = (
                fragment.get("results")
                or fragment.get("hypotheses")
                or fragment.get("candidate_results")
            )
            if isinstance(candidate_items, list):
                recovered_items.extend(
                    item for item in candidate_items if isinstance(item, Mapping)
                )
            elif fragment.get("option") is not None and (
                fragment.get("verdict") is not None
                or fragment.get("status") is not None
            ):
                recovered_items.append(fragment)
        # Damaged model output sometimes reasons past an earlier tuple and emits
        # a revised tuple later. Validate the latest complete tuple first; an
        # invalid revision still falls back to an earlier valid tuple.
        raw_items = list(reversed(recovered_items))
    if not raw_items:
        line_pattern = re.compile(
            r"(?im)^\s*(?:option\s+)?([A-Z])\s*[:=\-]\s*"
            r"(UNCHECKED|SUPPORTED|OPEN|CONTRADICTED)\b(.*)$"
        )
        for match in line_pattern.finditer(text):
            tail = match.group(3)
            source_match = re.search(
                r"(?i)source(?:_ids|s)?\s*[:=]\s*\[?([^\];|]+)", tail
            )
            source_ids = (
                [item.strip().strip("\"'") for item in source_match.group(1).split(",")]
                if source_match
                else []
            )
            raw_items.append(
                {
                    "option": match.group(1),
                    "verdict": match.group(2),
                    "source_ids": source_ids,
                    "claim_ids": [],
                    "brief_reason": tail.strip(" ;|"),
                }
            )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        option = str(item.get("option") or "").strip().upper()
        verdict = str(item.get("verdict") or item.get("status") or "").strip().upper()
        source_ids = string_list(item.get("source_ids"))
        claim_ids = string_list(item.get("claim_ids"))
        if option not in allowed_options or option in seen:
            continue
        if verdict not in {"UNCHECKED", "SUPPORTED", "OPEN", "CONTRADICTED"}:
            continue
        decisive_verdict = verdict in {"SUPPORTED", "CONTRADICTED"}
        downgrade_reason = ""
        if any(claim_id not in states_by_claim for claim_id in claim_ids):
            if decisive_verdict:
                downgrade_reason = (
                    "Decisive verdict lacked eligible latest ALIGNED Evidence; "
                    "kept OPEN."
                )
            else:
                continue
        if not claim_ids and any(
            source_id not in allowed_sources for source_id in source_ids
        ):
            if decisive_verdict:
                downgrade_reason = (
                    "Decisive verdict lacked eligible latest ALIGNED Evidence; "
                    "kept OPEN."
                )
            else:
                continue
        if not claim_ids and source_ids:
            claim_ids = list(
                dict.fromkeys(
                    claim_id
                    for source_id in source_ids
                    for claim_id in aligned_claims_by_source.get(source_id, [])
                )
            )
        if decisive_verdict and not downgrade_reason:
            if not claim_ids:
                downgrade_reason = (
                    "Decisive verdict lacked eligible latest ALIGNED Evidence; "
                    "kept OPEN."
                )
            elif any(claim_id not in aligned_states for claim_id in claim_ids):
                downgrade_reason = (
                    "Decisive verdict lacked eligible latest ALIGNED Evidence; "
                    "kept OPEN."
                )
            else:
                derived_sources = list(
                    dict.fromkeys(
                        str(source_id).strip()
                        for claim_id in claim_ids
                        for source_id in aligned_states[claim_id].get(
                            "source_ids", []
                        )
                        if str(source_id).strip()
                    )
                )
                if not derived_sources:
                    downgrade_reason = (
                        "Decisive verdict lacked eligible latest ALIGNED Evidence; "
                        "kept OPEN."
                    )
                else:
                    # Claim IDs are the semantic binding. Project their admitted
                    # sources instead of trusting a model-synthesized range ID.
                    source_ids = derived_sources
        if downgrade_reason:
            verdict = "OPEN"
            claim_ids = []
            source_ids = []
        normalized.append(
            {
                "option": option,
                "verdict": verdict,
                "claim_ids": claim_ids,
                "source_ids": source_ids,
                "brief_reason": downgrade_reason
                or str(item.get("brief_reason") or "").strip(),
            }
        )
        seen.add(option)
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    def numeric_cardinality(value: Any) -> int | None:
        text_value = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(value or "").strip().casefold(),
        ).strip()
        if text_value.isdigit():
            return int(text_value)
        return number_words.get(text_value)

    candidate_cardinalities = {
        str(item.get("option") or "").strip().upper(): numeric_cardinality(
            item.get("claim")
        )
        for item in option_candidates
    }
    exact_count_question = bool(
        allow_legacy_exact_count
    ) and bool(
        re.search(r"(?i)\bhow many\b", question_stem)
    ) and all(
        value is not None for value in candidate_cardinalities.values()
    )
    count_summary = (
        parsed.get("exact_count_summary")
        if exact_count_question and isinstance(parsed, Mapping)
        else None
    )
    raw_members = (
        count_summary.get("members")
        if isinstance(count_summary, Mapping)
        else None
    )
    if isinstance(raw_members, list) and raw_members:
        members_by_label: dict[str, list[str]] = {}
        summary_valid = True
        for raw_member in raw_members:
            if not isinstance(raw_member, Mapping):
                summary_valid = False
                break
            member = " ".join(str(raw_member.get("member") or "").split())
            member_claim_ids = string_list(raw_member.get("claim_ids"))
            if (
                not member
                or not member_claim_ids
                or any(
                    claim_id not in aligned_states
                    for claim_id in member_claim_ids
                )
            ):
                summary_valid = False
                break
            member_key = member.casefold()
            members_by_label.setdefault(member_key, [])
            members_by_label[member_key].extend(member_claim_ids)
        if summary_valid:
            summary_claim_ids = list(
                dict.fromkeys(
                    claim_id
                    for member_claim_ids in members_by_label.values()
                    for claim_id in member_claim_ids
                )
            )
            summary_source_ids = list(
                dict.fromkeys(
                    str(source_id).strip()
                    for claim_id in summary_claim_ids
                    for source_id in aligned_states[claim_id].get("source_ids", [])
                    if str(source_id).strip()
                )
            )
            member_count = len(members_by_label)
            matching_options = [
                option
                for option, cardinality in candidate_cardinalities.items()
                if cardinality == member_count
            ]
            if summary_source_ids and len(matching_options) == 1:
                normalized_by_option = {
                    item["option"]: item for item in normalized
                }
                for option in option_order:
                    cardinality = candidate_cardinalities[option]
                    normalized_by_option[option] = {
                        "option": option,
                        "verdict": (
                            "SUPPORTED"
                            if option == matching_options[0]
                            else "CONTRADICTED"
                        ),
                        "claim_ids": summary_claim_ids,
                        "source_ids": summary_source_ids,
                        "brief_reason": (
                            "Validated exact-count summary contains "
                            f"{member_count} distinct supported members."
                        ),
                    }
                normalized = list(normalized_by_option.values())
    normalized.sort(key=lambda item: option_order.get(item["option"], len(option_order)))
    return normalized


class RepairController:
    """Audit one answer-critical delta and dispatch bounded repair observations."""

    def __init__(
        self,
        *,
        run_dir: Path,
        backend: JSONBackend,
        evidence_audit_client: Any | None = None,
        raw_qwen_only: bool = False,
        max_repair_observations: int | None = 1,
        operator_gate_mode: str = "off",
    ) -> None:
        self.run_dir = run_dir
        self.backend = backend
        self.audit_log_path = run_dir / "semantic_delta" / "audit_records.jsonl"
        self.boundary_log_path = run_dir / "semantic_delta" / "raw_boundary.jsonl"
        self.state_log_path = run_dir / "semantic_delta" / "state_events.jsonl"
        self.operator_gate_log_path = (
            run_dir / "semantic_delta" / "operator_gate_packets.jsonl"
        )
        self._receipts_by_root: dict[str, SourceReceipt] = {}
        self._round_receipts: dict[int, tuple[SourceReceipt, ...]] = {}
        self._admitted_delta_by_round: dict[int, str] = {}
        self._evidence_states: dict[str, dict[str, Any]] = {}
        self._last_hypothesis_results: list[dict[str, Any]] = []
        self._last_hypothesis_signature: tuple[Any, ...] | None = None
        self._hypothesis_verifier_status: Literal[
            "NOT_RUN", "COMPLETE", "PARTIAL", "UNAVAILABLE"
        ] = "NOT_RUN"
        self._hypothesis_verifier_reason: str | None = None
        self._hypothesis_verifier_revision = 0
        normalized_gate_mode = str(operator_gate_mode or "off").strip().lower()
        if normalized_gate_mode not in OPERATOR_GATE_MODES:
            raise ValueError("invalid_operator_gate_mode")
        self.operator_gate_mode = normalized_gate_mode
        self._operator_gate_packet: dict[str, Any] | None = None
        self._operator_gate_signature: tuple[Any, ...] | None = None
        self._operator_gate_revision = 0
        self._operator_gate_request_count = 0
        self._observation_coverage: dict[str, Any] | None = None
        self._active_observation_need: dict[str, Any] | None = None
        self._observation_need_history: list[dict[str, Any]] = []
        self._observation_need_counter = 0
        self._observation_need_recovery_attempted = False
        self._observation_need_recovery_status: str | None = None
        self._last_search_need_resolution: dict[str, Any] | None = None
        self._completed_observation_plan_signatures: set[tuple[Any, ...]] = set()
        self._unverified_observation_by_round: dict[int, dict[str, Any]] = {}
        self._selection_done = False
        self._direct_source_facts_available = False
        self._pending: PendingDelta | None = None
        self._repair_observations = 0
        self._expecting_repair = False
        self._expecting_revision = False
        self._consecutive_no_progress = 0
        self._executed_open_diagnoses: set[str] = set()
        self._blocked_candidate: Mapping[str, Any] | None = None
        self._commitment_rejection: str | None = None
        self._visual_evidence_auditor: Any | None = None
        if evidence_audit_client is not None:
            from experiments.semantic_delta_pilot.coarse_mcq_audit_runtime import (
                EvidenceAuditRuntime,
            )

            self._visual_evidence_auditor = EvidenceAuditRuntime(
                audit_client=evidence_audit_client,
                run_dir=run_dir,
            )
        self.raw_qwen_only = bool(raw_qwen_only)
        self.max_repair_observations = None if max_repair_observations is None else int(max_repair_observations)
        if self.max_repair_observations is not None and self.max_repair_observations <= 0:
            raise ValueError("max_repair_observations_must_be_positive")

    @property
    def repair_observations(self) -> int:
        return self._repair_observations

    def _write_state(self, event: str, **payload: Any) -> None:
        _append_jsonl(
            self.state_log_path,
            {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **payload},
        )

    def active_observation_need(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._active_observation_need)

    def ensure_initial_observation_need(self, *, question, option_candidates):
        """Verifier owns the first neutral need; never fall back to the question."""
        active = self.active_observation_need()
        if active is not None:
            return active
        if (self._observation_need_counter or self._evidence_states
                or getattr(self, "_initial_observation_need_attempted", False)):
            raise RuntimeError("active_observation_need_required")
        self._initial_observation_need_attempted = True
        if getattr(self, "uniform_seed_content", None) or (getattr(self, "asr_context", None) is not None
                and self.asr_context.seed_text()):
            return self._prepare_initial_asr_answer(question, option_candidates)
        messages = [
            {"role": "system", "content": (
                "As Verifier, identify the first missing video-observable fact needed "
                "to assess this question. Return JSON observation_need containing "
                "missing_proposition and affected_options. missing_proposition is a "
                "neutral observation goal, without the original question, option labels, "
                "answer judgments, timestamps, tools or sampling instructions. "
                "affected_options lists the supplied option labels it can inform. "
                "No video has been observed; do not invent facts.")},
            {"role": "user", "content": json.dumps({"question": question,
                "option_hypotheses": option_candidates}, ensure_ascii=False)},
        ]
        asr_background = getattr(self, "asr_background", None)
        if asr_background is not None:
            messages[0]["content"] += (
                " The ASR background is a derived overview of speech, not a visual observation "
                "or a need-audited fact. Use it to identify a neutral information need, "
                "including language-only or cross-transcript needs. It may omit or misinterpret "
                "material; do not assume an event exists because the overview describes it.")
            payload = json.loads(messages[1]["content"])
            payload["asr_background"] = copy.deepcopy(asr_background)
            messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
        request = {"response_mode": "text", "max_tokens": 512,
            "enable_thinking": False, "messages": messages,
            "estimated_input_tokens": _estimate_request_input_tokens(messages)}
        try:
            if not option_candidates:
                raise ValueError("option_candidates_required")
            raw = self.backend.call("initial_observation_need", request)
            if raw.get("finish_reason") == "length":
                raise ValueError("truncated_output")
            response_text = _response_text(raw)
            candidate, error = _extract_observation_need_candidate(response_text)
            if candidate is None and error is None:
                # A complete flat need only omits its envelope. Preserve its
                # content and require the same semantic validation below.
                try:
                    flat = _strict_json_object(response_text)
                except (TypeError, ValueError):
                    flat = None
                if (isinstance(flat, Mapping)
                        and set(flat) == {"missing_proposition", "affected_options"}
                        and isinstance(flat["missing_proposition"], str)
                        and isinstance(flat["affected_options"], list)
                        and all(isinstance(option, str) for option in flat["affected_options"])):
                    candidate = flat
            need = None
            if candidate is not None and error is None:
                need, error = _validate_observation_need_candidate(
                    candidate, option_candidates=option_candidates)
            if need is None:
                raise ValueError(error or "missing_observation_need")
            self._activate_or_update_search_need(semantic_need=need,
                authority_ref="initial_observation_need")
        except Exception as exc:
            self._write_state("INITIAL_OBSERVATION_NEED_FAILED",
                error_type=type(exc).__name__, reason=str(exc))
            raise RuntimeError("initial_observation_need_failed:" + str(exc)) from exc
        return self.active_observation_need()

    def _prepare_initial_asr_answer(self, question, option_candidates):
        """Give the existing decision stage Q + options + seed images/ASR."""
        from avp.main import PlanSpec, Reflector
        from .asr_mode import write_json

        asr = getattr(self, "asr_context", None)
        visual = getattr(self, "uniform_seed_content", None)
        visual_manifest = getattr(self, "uniform_seed_manifest", {})
        self.asr_seed_text = asr.seed_text() if asr is not None else ""
        options = [str(x["claim"]) for x in option_candidates]
        plan = PlanSpec(plan_version="1.0", query=question, watch=None)
        plan.decision_context = {"asr_transcript": self.asr_seed_text}
        reflection = Reflector(self.asr_decision_client).reflect(
            query=question, plan=plan, evidence_list=[], interaction_history=[],
            video_path=visual_manifest.get("video_path") or str(asr.video_path),
            duration_sec=visual_manifest.get("end_sec") or asr.duration,
            # The seed has no audited option verdicts yet. Let the existing
            # decision assess the transcript without the post-audit answer gate.
            is_last_round=False, options=options, option_candidates=None,
            **({"initial_visual_content": visual} if visual else {}))
        # Only the existing Answer/decision parser reads its ordinary output.
        # There is no additional five-field ASR reply or ASR candidate parser.
        self.asr_initial_decision = copy.deepcopy(reflection)
        self.asr_initial_answer = copy.deepcopy(
            reflection.get("final_answer") or reflection.get("candidate_answer"))
        self.asr_initial_analysis = str(reflection.get("justification") or reflection.get("reasoning") or "")
        record = {"question": question, "source": asr.source if asr is not None else None,
            "asr_transcript": self.asr_seed_text, "decision": reflection,
            "consumed_segment_ids": [s["id"] for s in asr.records] if asr is not None else [],
            "ground_truth_loaded": False}
        if visual:
            record["uniform_seed"] = visual_manifest
        write_json(self.run_dir / "asr_initial_decision.json", record)
        self._write_state("ASR_SEED_DECISION", **record)
        # Subsequent needs and decisions use the analysis, not repeated image payloads.
        if visual:
            self.uniform_seed_content = None
        if reflection.get("sufficient"):
            return None
        return self.prepare_decision_observation_need(question=question,
            option_candidates=option_candidates, feedback=self.asr_initial_analysis)

    def prepare_decision_observation_need(self, *, question, option_candidates, feedback):
        """Translate an explicit decision-side gap into a neutral need, text only."""
        if self._active_observation_need is not None:
            return self.active_observation_need()
        request = _observation_need_recovery_request(question=question,
            option_candidates=option_candidates, verifier_results=self._last_hypothesis_results,
            rejection_reason=str(feedback or "Decision requests additional evidence."))
        request["messages"][0]["content"] = (
            "As Verifier, turn the decision feedback into one neutral video-observable need. "
            "Return observation_need with missing_proposition and affected_options; no option "
            "labels or answer judgments in the proposition, and no execution parameters. "
            "Use existing facts and unresolved gaps. If no useful observable need remains, "
            "return observation_need:null. Do not change option verdicts or invent observations.")
        payload = json.loads(request["messages"][1]["content"])
        payload["decision_feedback"] = str(feedback or "")
        if getattr(self, "asr_initial_analysis", ""):
            payload["initial_asr_analysis"] = self.asr_initial_analysis
        if getattr(self, "asr_seed_text", "") and self._hypothesis_verifier_revision == 0:
            payload["asr_transcript"] = self.asr_seed_text
        payload["local_facts"] = self._evidence_state_snapshot()
        if getattr(self, "need_local_audit_enabled", False):
            payload["need_answers"] = copy.deepcopy(self._need_answers)
            payload["verifier_diagnostic"] = self._hypothesis_verifier_reason
        request["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
        request["estimated_input_tokens"] = _estimate_request_input_tokens(request["messages"])
        from .upper_observation_only import need_request
        request = need_request(request, self)
        for attempt in range(2):
            raw = self.backend.call("observation_need_recovery", request)
            text = _response_text(raw)
            parsed = _json_object_with_minor_repair(text)
            if isinstance(parsed, Mapping) and "observation_need" in parsed and parsed["observation_need"] is None:
                self._write_state("DECISION_NEED_UNAVAILABLE", feedback=feedback)
                return None
            candidate, error = _extract_observation_need_candidate(text)
            need = None
            if candidate is not None and error is None:
                need, error = _validate_observation_need_candidate(candidate, option_candidates=option_candidates)
            if need is not None:
                self._activate_or_update_search_need(semantic_need=need, authority_ref="decision_feedback")
                return self.active_observation_need()
            if attempt:
                raise RuntimeError("decision_need_format_failed:" + str(error or "missing_observation_need"))
            self._write_state("DECISION_NEED_FORMAT_REPAIR", reason=error)
            request = copy.deepcopy(request)
            request["messages"].extend([{"role":"assistant","content":text},
                {"role":"user","content":"Repair the need format using supplied context only."}])

    def _close_active_observation_need(
        self,
        status: Literal["RESOLVED", "REVISED", "UNRESOLVED", "UNSUPPORTED"],
        *,
        reason: str,
    ) -> dict[str, Any] | None:
        if self._active_observation_need is None:
            return None
        closed = copy.deepcopy(self._active_observation_need)
        closed["status"] = status
        closed["status_reason"] = reason
        self._observation_need_history.append(closed)
        self._active_observation_need = None
        self._write_state(
            "OBSERVATION_NEED_STATUS_CHANGED",
            observation_need=copy.deepcopy(closed),
        )
        return closed

    def _new_observation_need(
        self,
        *,
        origin: Literal["VERIFIER_SEARCH"],
        missing_proposition: str,
        affected_options: list[str],
        authority_ref: str,
    ) -> dict[str, Any]:
        self._observation_need_counter += 1
        need = {
            "need_id": f"need_{self._observation_need_counter}",
            "revision": 1,
            "status": "ACTIVE",
            "origin": origin,
            "missing_proposition": missing_proposition,
            "affected_options": list(affected_options),
            "authority_ref": authority_ref,
        }
        self._active_observation_need = need
        self._write_state(
            "OBSERVATION_NEED_ACTIVATED",
            observation_need=copy.deepcopy(need),
        )
        return need

    def _activate_or_update_search_need(
        self,
        *,
        semantic_need: Mapping[str, Any],
        authority_ref: str,
    ) -> None:
        proposition = str(semantic_need["missing_proposition"])
        options = list(semantic_need["affected_options"])
        active = self._active_observation_need
        if active is None or active.get("origin") != "VERIFIER_SEARCH":
            if active is not None:
                self._close_active_observation_need(
                    "REVISED",
                    reason="Verifier SEARCH replaced the prior active need.",
                )
            self._new_observation_need(
                origin="VERIFIER_SEARCH",
                missing_proposition=proposition,
                affected_options=options,
                authority_ref=authority_ref,
            )
            return

        same_semantics = (
            active.get("missing_proposition") == proposition
            and list(active.get("affected_options") or []) == options
        )
        previous = copy.deepcopy(active)
        previous["status"] = "UNRESOLVED" if same_semantics else "REVISED"
        previous["status_reason"] = (
            "Audited observation did not resolve the same need."
            if same_semantics
            else "Verifier replaced the missing proposition after audited observation."
        )
        self._observation_need_history.append(previous)
        self._active_observation_need = {
            **copy.deepcopy(active),
            "revision": int(active.get("revision") or 1) + 1,
            "status": "ACTIVE",
            "missing_proposition": proposition,
            "affected_options": options,
            "authority_ref": authority_ref,
        }
        self._write_state(
            "OBSERVATION_NEED_REVISED",
            previous=previous,
            observation_need=copy.deepcopy(self._active_observation_need),
        )


    def _recover_search_observation_need(
        self,
        *,
        question: str,
        option_candidates: list[Mapping[str, Any]],
        results: list[Mapping[str, Any]],
        rejection_reason: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Make at most one need-only recovery call for the entire question."""

        if self._observation_need_recovery_attempted:
            return None, "recovery_budget_exhausted"
        self._observation_need_recovery_attempted = True
        self._observation_need_recovery_status = "ATTEMPTED"
        self._write_state(
            "OBSERVATION_NEED_RECOVERY_ATTEMPTED",
            origin="VERIFIER_SEARCH",
            authority_ref=(
                f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}"
            ),
            inline_rejection=rejection_reason,
        )
        request = _observation_need_recovery_request(
            question=question,
            option_candidates=option_candidates,
            verifier_results=results,
            rejection_reason=rejection_reason,
        )
        try:
            raw = self.backend.call("observation_need_recovery", request)
            response_text = _response_text(raw)
            raw_need, extraction_rejection = _extract_observation_need_candidate(
                response_text
            )
        except Exception as exc:
            self._observation_need_recovery_status = "UNAVAILABLE"
            self._write_state(
                "OBSERVATION_NEED_RECOVERY_UNAVAILABLE",
                origin="VERIFIER_SEARCH",
                error_type=type(exc).__name__,
            )
            return None, rejection_reason
        semantic_need: dict[str, Any] | None = None
        rejection = extraction_rejection
        if rejection is None and raw_need is not None:
            semantic_need, rejection = _validate_observation_need_candidate(
                raw_need,
                option_candidates=option_candidates,
            )
        if semantic_need is None:
            self._observation_need_recovery_status = "REJECTED"
            self._write_state(
                "OBSERVATION_NEED_RECOVERY_REJECTED",
                origin="VERIFIER_SEARCH",
                reason=rejection or "malformed_output",
            )
            return None, rejection or "malformed_output"
        self._observation_need_recovery_status = "ACCEPTED"
        self._write_state(
            "OBSERVATION_NEED_RECOVERY_ACCEPTED",
            origin="VERIFIER_SEARCH",
            observation_need=copy.deepcopy(semantic_need),
        )
        return semantic_need, None

    def _update_search_observation_need(
        self,
        *,
        results: list[Mapping[str, Any]],
        raw_need: Mapping[str, Any] | None,
        extraction_rejection: str | None,
        raw_resolution: Mapping[str, Any] | None,
        resolution_extraction_rejection: str | None,
        option_candidates: list[Mapping[str, Any]],
        evidence_states: list[Mapping[str, Any]],
        question: str,
        plan_need_id: str | None,
        reused_claim_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_results = [dict(item) for item in results]
        active = self._active_observation_need
        if active is not None and active.get("origin") == "VERIFIER_SEARCH":
            resolution: dict[str, Any] | None = None
            rejection = resolution_extraction_rejection
            if str(plan_need_id or "") != str(active.get("need_id") or ""):
                rejection = "need_plan_lineage_mismatch"
            elif rejection is None and raw_resolution is not None:
                resolution, rejection = _validate_need_resolution_candidate(
                    raw_resolution,
                    active_need=active,
                    option_candidates=option_candidates,
                    evidence_states=evidence_states,
                    results=normalized_results,
                    reused_claim_ids=reused_claim_ids,
                )
            elif rejection is None:
                rejection = "missing_need_resolution"

            if resolution is not None:
                self._last_search_need_resolution = copy.deepcopy(resolution)
                self._write_state(
                    "OBSERVATION_NEED_RESOLUTION_ACCEPTED",
                    need_id=active.get("need_id"),
                    revision=active.get("revision"),
                    need_resolution=copy.deepcopy(resolution),
                )
                if resolution["status"] == "RESOLVED":
                    self._close_active_observation_need(
                        "RESOLVED",
                        reason=(
                            "Hypothesis Verifier answered the missing proposition: "
                            f"{resolution['finding']}"
                        ),
                    )
                    active = None
                else:
                    replacement = None
                    if raw_need is not None and extraction_rejection is None:
                        replacement, _ = _validate_observation_need_candidate(
                            raw_need, option_candidates=option_candidates)
                    if replacement and replacement["missing_proposition"] != active["missing_proposition"]:
                        self._close_active_observation_need("UNSUPPORTED", reason=resolution["finding"])
                        self._activate_or_update_search_need(semantic_need=replacement,
                            authority_ref=f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}:redirect")
                        return normalized_results
                    self._activate_or_update_search_need(
                        semantic_need={
                            "missing_proposition": active["missing_proposition"],
                            "affected_options": list(
                                active.get("affected_options") or []
                            ),
                        },
                        authority_ref=(
                            f"hypothesis_verifier_revision_"
                            f"{self._hypothesis_verifier_revision}:unresolved"
                        ),
                    )
                    return normalized_results
            else:
                self._last_search_need_resolution = {
                    "status": "REJECTED",
                    "reason": rejection or "malformed_need_resolution",
                }
                self._write_state(
                    "OBSERVATION_NEED_RESOLUTION_REJECTED",
                    need_id=active.get("need_id"),
                    revision=active.get("revision"),
                    reason=rejection or "malformed_need_resolution",
                )
                self._activate_or_update_search_need(
                    semantic_need={
                        "missing_proposition": active["missing_proposition"],
                        "affected_options": list(
                            active.get("affected_options") or []
                        ),
                    },
                    authority_ref=(
                        f"hypothesis_verifier_revision_"
                        f"{self._hypothesis_verifier_revision}:resolution_rejected"
                    ),
                )
                return normalized_results

        supported = [
            str(item.get("option") or "").strip().upper()
            for item in normalized_results
            if str(item.get("verdict") or "").strip().upper() == "SUPPORTED"
        ]
        if supported:
            if raw_need is not None or extraction_rejection is not None:
                self._write_state(
                    "OBSERVATION_NEED_REJECTED",
                    reason=extraction_rejection or "malformed_output",
                    origin="VERIFIER_SEARCH",
                    authority_ref=f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}",
                )
            if (
                self._active_observation_need is not None
                and self._active_observation_need.get("origin") == "VERIFIER_SEARCH"
            ):
                self._close_active_observation_need(
                    "RESOLVED",
                    reason="Hypothesis Verifier found at least one SUPPORTED option.",
                )
            return normalized_results
        semantic_need: dict[str, Any] | None = None
        rejection = extraction_rejection
        if rejection is None and raw_need is not None:
            semantic_need, rejection = _validate_observation_need_candidate(
                raw_need,
                option_candidates=option_candidates,
            )
        if semantic_need is None:
            inline_rejection = rejection or "malformed_output"
            self._write_state(
                "OBSERVATION_NEED_INLINE_REJECTED",
                reason=inline_rejection,
                origin="VERIFIER_SEARCH",
                authority_ref=f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}",
            )
            semantic_need, rejection = self._recover_search_observation_need(
                question=question,
                option_candidates=option_candidates,
                results=normalized_results,
                rejection_reason=inline_rejection,
            )
            if semantic_need is None:
                self._write_state(
                    "OBSERVATION_NEED_REJECTED",
                    reason=rejection or inline_rejection,
                    origin="VERIFIER_SEARCH",
                    authority_ref=(
                        f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}"
                    ),
                )
                active = self._active_observation_need
                if active is not None and active.get("origin") == "VERIFIER_SEARCH":
                    self._activate_or_update_search_need(
                        semantic_need={
                            "missing_proposition": active["missing_proposition"],
                            "affected_options": list(
                                active.get("affected_options") or []
                            ),
                        },
                        authority_ref=(
                            f"hypothesis_verifier_revision_"
                            f"{self._hypothesis_verifier_revision}:preserved"
                        ),
                    )
                return normalized_results
            authority_ref = (
                f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}"
                ":need_recovery"
            )
        else:
            authority_ref = (
                f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}"
            )
        self._activate_or_update_search_need(
            semantic_need=semantic_need,
            authority_ref=authority_ref,
        )
        return normalized_results

    @staticmethod
    def _open_diagnosis_key(diagnosis: str) -> str:
        return " ".join(diagnosis.casefold().split())

    def _open_check(
        self,
        *,
        alignment: AdmissionAlignment,
        evidence_revision: int,
    ) -> OpenCheck | None:
        if alignment.source_alignment != "UNCERTAIN":
            return None
        diagnosis = "; ".join(alignment.missing_premises).strip()
        if not diagnosis:
            return OpenCheck(
                status="SATURATED",
                diagnosis="",
                evidence_revision=evidence_revision,
            )
        status: Literal["ACTIONABLE", "SATURATED"] = (
            "SATURATED"
            if self._open_diagnosis_key(diagnosis) in self._executed_open_diagnoses
            else "ACTIONABLE"
        )
        return OpenCheck(
            status=status,
            diagnosis=diagnosis,
            evidence_revision=evidence_revision,
        )

    def _saturate_current_open_check(self) -> None:
        if self._pending is None or self._pending.open_check is None:
            return
        self._pending = replace(
            self._pending,
            open_check=replace(self._pending.open_check, status="SATURATED"),
        )

    def _record_evidence_state(
        self,
        *,
        packet: DeltaPacket,
        alignment: AdmissionAlignment,
        round_id: int | None = None,
    ) -> None:
        self._evidence_states[packet.delta_id] = {
            "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
            "claim_id": packet.delta_id,
            "claim": packet.delta_claim,
            "source_alignment": alignment.source_alignment,
            "source_ids": list(alignment.source_span_ids),
            "missing_premises": list(alignment.missing_premises),
            "round_id": round_id,
            "latest": True,
            "updates_claim_ids": [],
            "superseded_by_claim_ids": [],
        }

    def _record_visual_evidence_states(
        self,
        *,
        round_id: int,
        audit_event: Mapping[str, Any],
        receipts: tuple[SourceReceipt, ...],
    ) -> list[str]:
        claims = {
            str(item.get("claim_id") or ""): str(item.get("text") or "").strip()
            for item in audit_event.get("claims", [])
            if isinstance(item, Mapping)
        }
        audit_sources = {
            str(item.get("source_id") or ""): item
            for item in audit_event.get("sources", [])
            if isinstance(item, Mapping)
        }
        claim_records = {str(item.get("claim_id") or ""): item
            for item in audit_event.get("claims", []) if isinstance(item, Mapping)}
        admitted_span_claims: list[str] = []
        for result in audit_event.get("observation_results", []):
            if not isinstance(result, Mapping):
                continue
            local_claim_id = str(result.get("claim_id") or "").strip()
            claim = claims.get(local_claim_id, "")
            if not local_claim_id or not claim:
                continue
            translated_sources: list[str] = []
            for source_id in result.get("source_ids", []):
                source = audit_sources.get(str(source_id))
                if source is None:
                    continue
                start, end = float(source["start_sec"]), float(source["end_sec"])
                match = next(
                    (
                        receipt
                        for receipt in receipts
                        if abs(receipt.start_sec - start) <= 1e-6
                        and abs(receipt.end_sec - end) <= 1e-6
                    ),
                    None,
                )
                if match is not None:
                    translated_sources.append(match.record.source_id)
            try:
                source_alignment = evidence_alignment_from_record(result)
            except ValueError:
                source_alignment = "UNCERTAIN"
            if (
                source_alignment in {"ALIGNED", "CONFLICTING"}
                and not translated_sources
            ):
                source_alignment = "UNCERTAIN"
            state_id = f"round_{round_id}_{local_claim_id}"
            raw_updates = result.get("updates_claim_ids") or []
            if isinstance(raw_updates, str):
                raw_updates = [raw_updates]
            updates_claim_ids = (
                list(
                    dict.fromkeys(
                        str(value).strip()
                        for value in raw_updates
                        if str(value).strip() in self._evidence_states
                        and self._evidence_states[str(value).strip()].get("latest")
                        is not False
                    )
                )
                if isinstance(raw_updates, (list, tuple))
                else []
            )
            for prior_claim_id in updates_claim_ids:
                prior = self._evidence_states[prior_claim_id]
                prior["latest"] = False
                successors = list(prior.get("superseded_by_claim_ids") or [])
                if state_id not in successors:
                    successors.append(state_id)
                prior["superseded_by_claim_ids"] = successors
            self._evidence_states[state_id] = {
                "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
                "claim_id": state_id,
                "claim": claim,
                "source_alignment": source_alignment,
                "source_ids": list(dict.fromkeys(translated_sources)),
                "missing_premises": (
                    [str(result.get("brief_reason") or "Evidence remains unresolved.")]
                    if source_alignment == "UNCERTAIN"
                    else []
                ),
                "round_id": round_id,
                "latest": True,
                "updates_claim_ids": updates_claim_ids,
                "superseded_by_claim_ids": [],
            }
            if audit_event.get("audit_mode") == "receipt_only":
                self._evidence_states[state_id].update(
                    audit_mode="receipt_only", semantic_verification="NOT_PERFORMED",
                    observation_uncertainty=result.get("observation_uncertainty"),
                    claim_scope={k:claim_records[local_claim_id].get(k)
                        for k in ("timestamp_start", "timestamp_end")},
                    source_scopes=[{
                        "start_sec": audit_sources[sid]["start_sec"],
                        "end_sec": audit_sources[sid]["end_sec"],
                        "sampling_fps": audit_sources[sid]["submitted_fps"],
                    } for sid in result.get("source_ids", []) if sid in audit_sources])
            if (
                source_alignment == "ALIGNED"
                and local_claim_id.startswith("observe_span_")
            ):
                admitted_span_claims.append(claim)
        return admitted_span_claims

    def _admit_visually_audited_observation(
        self,
        *,
        query: str,
        evidence: Any,
        round_id: int,
        receipts: tuple[SourceReceipt, ...],
        video_path: str,
    ) -> dict[str, Any] | None:
        if self._visual_evidence_auditor is None:
            return None
        prior_evidence_states = [
            copy.deepcopy(self._evidence_states[key])
            for key in sorted(self._evidence_states)
            if self._evidence_states[key].get("latest") is not False
            and self._evidence_states[key].get("round_id") is not None
        ]
        try:
            audit_event = self._visual_evidence_auditor.audit_observation_claims(
                query=query,
                evidence=evidence,
                video_path=video_path,
                prior_evidence_states=prior_evidence_states,
            )
        except Exception as exc:
            event = {
                "status": "AUDIT_UNAVAILABLE",
                "source_alignment": "UNCERTAIN",
                "permission": "SEARCH_ONLY",
                "round_id": round_id,
                "reason": "evidence_audit_unavailable",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            self._write_state("EVIDENCE_AUDIT_UNAVAILABLE", **event)
            return event

        admitted = self._record_visual_evidence_states(
            round_id=round_id,
            audit_event=audit_event,
            receipts=receipts,
        )
        self._round_receipts[round_id] = receipts
        for receipt in receipts:
            self._receipts_by_root.setdefault(receipt.record.provenance_root_id, receipt)
        if admitted:
            self._admitted_delta_by_round[round_id] = "\n".join(admitted)
        self._direct_source_facts_available = bool(admitted)
        if self._expecting_repair or self._expecting_revision:
            self._expecting_repair = False
            self._expecting_revision = False
            self._pending = None
        alignments = [
            evidence_alignment_from_record(state)
            for state in self._evidence_states.values()
            if state.get("round_id") == round_id
        ]
        event = {
            "status": "SOURCE_ADMISSION_COMPLETE",
            "permission": "ANSWER_ELIGIBLE" if admitted else "SEARCH_ONLY",
            "round_id": round_id,
            "evidence_audit_states": alignments,
            "admitted_claim_count": len(admitted),
        }
        self._write_state("EVIDENCE_AUDIT_COMPLETE", **event)
        return event

    def _evidence_state_snapshot(self) -> list[dict[str, Any]]:
        states = [
            copy.deepcopy(self._evidence_states[key])
            for key in sorted(self._evidence_states)
        ]
        for state in states:
            state["schema_version"] = EVIDENCE_AUDIT_SCHEMA_VERSION
            state["source_alignment"] = evidence_alignment_from_record(state)
            state.pop("status", None)
            state.pop("verdict", None)
            state.setdefault("latest", True)
            state.setdefault("updates_claim_ids", [])
            state.setdefault("superseded_by_claim_ids", [])
        recorded_sources = {
            source_id
            for state in states
            for source_id in state.get("source_ids", [])
        }
        # Retrieved captions are source records rather than same-generation raw
        # Observer interpretations. Preserve their existing direct-source role,
        # while never promoting an unaudited observer_claim here.
        for receipt in self._receipts_by_root.values():
            record = receipt.record
            if record.source_kind != "caption" or record.source_id in recorded_sources:
                continue
            states.append(
                {
                    "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
                    "claim_id": f"caption:{record.source_id}",
                    "claim": record.exact_caption,
                    "source_alignment": "ALIGNED",
                    "source_ids": [record.source_id],
                    "missing_premises": [],
                    "latest": True,
                    "updates_claim_ids": [],
                    "superseded_by_claim_ids": [],
                }
            )
        if getattr(self, "need_local_audit_enabled", False):
            from .need_answer_audit import answer_states
            states += answer_states(self)
        return states

    def _operator_gate_source_scopes(self) -> dict[str, list[dict[str, Any]]]:
        """Project transport-owned time scopes by exact source ID."""

        scopes: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, float, float, str]] = set()
        for round_id in sorted(self._round_receipts):
            for receipt in self._round_receipts[round_id]:
                source_id = receipt.record.source_id
                identity = (
                    source_id,
                    float(receipt.start_sec),
                    float(receipt.end_sec),
                    receipt.transaction_id,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                scopes.setdefault(source_id, []).append(
                    {
                        "source_id": source_id,
                        "source_kind": receipt.record.source_kind,
                        "locator": receipt.record.locator,
                        "producer": receipt.producer,
                        "transaction_id": receipt.transaction_id,
                        "media_identity": receipt.media_identity,
                        "sampling_fps": receipt.sampling_fps,
                        "start_sec": float(receipt.start_sec),
                        "end_sec": float(receipt.end_sec),
                    }
                )
        for items in scopes.values():
            items.sort(
                key=lambda item: (
                    item["start_sec"],
                    item["end_sec"],
                    item["transaction_id"],
                )
            )
        return scopes

    def _run_operator_gate(
        self,
        *,
        question: str,
        option_candidates: list[Mapping[str, Any]],
        evidence_states: list[Mapping[str, Any]],
        evidence_signature: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        if self.operator_gate_mode == "off":
            return None
        if evidence_signature == self._operator_gate_signature:
            return copy.deepcopy(self._operator_gate_packet)

        self._operator_gate_revision += 1
        source_scopes = self._operator_gate_source_scopes()
        request = build_operator_gate_request(
            question=question,
            option_candidates=option_candidates,
            evidence_states=evidence_states,
            source_scopes=source_scopes,
            sampling_coverage=self._observation_coverage,
        )
        try:
            self._operator_gate_request_count += 1
            raw = self.backend.call("operator_gate", request)
            packet = execute_operator_gate_response(
                text=_response_text(raw),
                evidence_revision=self._operator_gate_revision,
                evidence_states=evidence_states,
                source_scopes=source_scopes,
                sampling_coverage=self._observation_coverage,
            )
        except OperatorGateValidationError as exc:
            packet = failed_operator_gate_packet(
                evidence_revision=self._operator_gate_revision,
                evidence_states=evidence_states,
                source_scopes=source_scopes,
                sampling_coverage=self._observation_coverage,
                status="INVALID",
                reason_code=exc.reason_code,
            )
        except Exception as exc:
            input_too_large = is_input_too_large_error(exc)
            packet = failed_operator_gate_packet(
                evidence_revision=self._operator_gate_revision,
                evidence_states=evidence_states,
                source_scopes=source_scopes,
                sampling_coverage=self._observation_coverage,
                status="INPUT_TOO_LARGE" if input_too_large else "UNAVAILABLE",
                reason_code=(
                    "INPUT_TOO_LARGE" if input_too_large else "GATE_UNAVAILABLE"
                ),
            )
            packet["error_type"] = type(exc).__name__

        self._operator_gate_signature = evidence_signature
        self._operator_gate_packet = copy.deepcopy(packet)
        _append_jsonl(self.operator_gate_log_path, packet)
        self._write_state(
            "OPERATOR_GATE_EVALUATED",
            mode=self.operator_gate_mode,
            evidence_revision=self._operator_gate_revision,
            status=packet["status"],
            reason_code=packet["reason_code"],
            projection_closure=copy.deepcopy(packet["projection_closure"]),
        )
        return copy.deepcopy(packet)

    def verify_option_hypotheses(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        """Run the option-only Hypothesis Verifier after Evidence Audit.

        The verifier receives tool evidence text, not Audit answers or video. Option
        verdicts are never retried; only a missing or invalid zero-support
        observation need may receive one bounded need-only recovery call.
        """

        option_candidates = list(kwargs.get("option_candidates") or [])
        if not option_candidates:
            return None
        evidence_states = self._evidence_state_snapshot()
        local_audit = getattr(self, "need_local_audit_enabled", False)
        if not evidence_states and not getattr(self, "need_acquisition_enabled", False):
            return None
        evidence_signature = tuple(
            (
                state["claim_id"],
                state["source_alignment"],
                tuple(state.get("source_ids", [])),
                tuple(state.get("missing_premises", [])),
                state.get("latest") is not False,
                tuple(state.get("updates_claim_ids", [])),
            )
            for state in evidence_states
        )
        acquisition = getattr(kwargs.get("plan"), "acquisition", None) or {}
        reused_claim_ids = acquisition.get("claim_ids") if acquisition.get("action") == "reuse" else None
        if local_audit:
            reused_claim_ids = None  # Reuse was already consumed by the local Audit.
        signature = evidence_signature
        if local_audit:
            signature += (("__need_answers__", len(self._need_answers)),)
        if getattr(self, "need_acquisition_enabled", False):
            active = self._active_observation_need or {}
            signature += (("__need__", active.get("need_id"), active.get("revision"),
                           tuple(reused_claim_ids or []), acquisition.get("action"),
                           acquisition.get("reason") if acquisition.get("action") == "unavailable" else None),)
        if self._observation_coverage is not None:
            signature = signature + (
                (
                    "__observation_coverage__",
                    json.dumps(
                        self._observation_coverage,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )
        operator_gate_packet = self._run_operator_gate(
            question=str(kwargs.get("query") or ""),
            option_candidates=option_candidates,
            evidence_states=evidence_states,
            evidence_signature=signature,
        )
        if signature == self._last_hypothesis_signature:
            return copy.deepcopy(self._last_hypothesis_results) or None
        active_search_need = (
            copy.deepcopy(self._active_observation_need)
            if self._active_observation_need is not None
            and self._active_observation_need.get("origin") == "VERIFIER_SEARCH"
            else None
        )
        request = _hypothesis_verifier_request(
            question=str(kwargs.get("query") or ""),
            option_candidates=option_candidates,
            evidence_states=evidence_states,
            observation_coverage=self._observation_coverage,
            active_search_need=active_search_need,
            operator_gate_packet=(
                operator_gate_packet
                if self.operator_gate_mode == "active"
                else None
            ),
            allow_legacy_exact_count=self.operator_gate_mode != "active",
            need_answers=self._need_answers if local_audit else None,
        )
        if reused_claim_ids is not None:
            active = self._active_observation_need or {}
            eligible = {state["claim_id"] for state in evidence_states
                if state.get("audit_mode") == "receipt_only"
                and state.get("source_alignment") == "ALIGNED" and state.get("latest") is not False}
            plan = kwargs.get("plan")
            if (not getattr(self, "need_acquisition_enabled", False)
                    or kwargs.get("evidence") is not None or not reused_claim_ids
                    or any(cid not in eligible for cid in reused_claim_ids)
                    or getattr(plan, "need_id", None) != active.get("need_id")
                    or getattr(plan, "need_revision", None) != active.get("revision")):
                raise ValueError("invalid_evidence_reuse_lineage")
            # Planning alone does not consume reuse; record only an actual Verifier attempt.
            self._need_reuse_attempts.add((
                " ".join(str(active["missing_proposition"]).casefold().split()),
                tuple(sorted(reused_claim_ids))))
            payload = json.loads(request["messages"][1]["content"])
            payload["evidence_reuse"] = {"claim_ids": reused_claim_ids,
                "new_observation": False, "original_rounds_preserved": True}
            request["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
            request["messages"][0]["content"] += (
                " For this evidence-reuse decision there is no new observation round. "
                "Resolve the current need from existing facts and cite at least one evidence_reuse "
                "claim_id. Original scopes and coverage still apply.")
        if acquisition.get("action") == "unavailable":
            payload = json.loads(request["messages"][1]["content"])
            payload["acquisition_limit"] = {"reason": acquisition["reason"], "new_observation": False}
            request["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
            request["messages"][0]["content"] += (
                " The acquisition_limit reports an unavailable operation, not absence in the video. "
                "Reassess existing facts. If this need remains unresolved, you may supply one "
                "different neutral observation_need for an alternative observable lead.")
        if local_audit or getattr(self, "asr_context", None) is not None:
            from .verifier_compact_view import project
            request, evidence_states, projection = project(request)
            self._write_state("VERIFIER_COMPACT_VIEW", **projection)
        initial_analysis = getattr(self, "asr_initial_analysis", "")
        if initial_analysis and self._hypothesis_verifier_revision == 0:
            user_payload = json.loads(request["messages"][1]["content"])
            user_payload["initial_asr_analysis"] = initial_analysis
            request["messages"][1]["content"] = json.dumps(user_payload, ensure_ascii=False)
            request["messages"][0]["content"] += (
                " The initial_asr_analysis is prior reasoning, not an audited fact. "
                "Reconsider it using the original question and current evidence.")
            request["estimated_input_tokens"] = _estimate_request_input_tokens(request["messages"])
        if local_audit:
            from .upper_observation_only import candidate_request
            request = candidate_request(request)
        self._hypothesis_verifier_revision += 1
        raw_need: Mapping[str, Any] | None = None
        need_extraction_rejection: str | None = None
        raw_resolution: Mapping[str, Any] | None = None
        resolution_extraction_rejection: str | None = None
        try:
            expected = {str(item["option"]).strip().upper() for item in option_candidates}
            attempts = 2 if getattr(self, "need_acquisition_enabled", False) else 1
            for attempt in range(attempts):
                raw = self.backend.call("hypothesis_verifier", request)
                response_text = _response_text(raw)
                results = _parse_hypothesis_verifier_results(
                    response_text, option_candidates=option_candidates,
                    evidence_states=evidence_states, question=str(kwargs.get("query") or ""),
                    allow_legacy_exact_count=self.operator_gate_mode != "active")
                raw_need, need_extraction_rejection = _extract_observation_need_candidate(response_text)
                raw_resolution, resolution_extraction_rejection = _extract_need_resolution_candidate(response_text)
                issue = None
                if {r["option"] for r in results} != expected:
                    issue = "one complete result per supplied option is required"
                elif active_search_need is not None and not local_audit:
                    if resolution_extraction_rejection or raw_resolution is None:
                        issue = resolution_extraction_rejection or "missing_need_resolution"
                    else:
                        _, issue = _validate_need_resolution_candidate(
                            raw_resolution, active_need=active_search_need,
                            option_candidates=option_candidates, evidence_states=evidence_states,
                            results=results, reused_claim_ids=reused_claim_ids)
                if issue is None or attempt == attempts - 1:
                    break
                self._write_state("HYPOTHESIS_VERIFIER_FORMAT_REPAIR", reason=issue, attempt=1)
                request = copy.deepcopy(request)
                request["messages"].extend([
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": "Repair only this response's format or invalid references: "
                     + issue + ". Use the supplied facts; do not invent evidence or request new video. "
                     "Return the complete corrected response."}])
        except Exception as exc:
            self._last_hypothesis_results = []
            self._hypothesis_verifier_status = "UNAVAILABLE"
            self._hypothesis_verifier_reason = f"{type(exc).__name__}:{exc}"
            self._write_state(
                "HYPOTHESIS_VERIFIER_UNAVAILABLE",
                error_type=type(exc).__name__, error=str(exc),
            )
            self._last_hypothesis_signature = signature
            return None
        self._last_hypothesis_signature = signature
        if not results:
            self._last_hypothesis_results = []
            self._hypothesis_verifier_status = "UNAVAILABLE"
            self._hypothesis_verifier_reason = "no_valid_results"
            self._write_state("HYPOTHESIS_VERIFIER_UNAVAILABLE", reason="no_valid_results")
            return None
        expected_options = {
            str(item.get("option") or "").strip().upper()
            for item in option_candidates
            if str(item.get("option") or "").strip()
        }
        received_options = {
            str(item.get("option") or "").strip().upper()
            for item in results
            if str(item.get("option") or "").strip()
        }
        missing_options = sorted(expected_options - received_options)
        new_need_rejection = None
        if not missing_options:
            plan = kwargs.get("plan")
            plan_need_id = (
                str(plan.get("need_id") or "").strip()
                if isinstance(plan, Mapping)
                else str(getattr(plan, "need_id", "") or "").strip()
            )
            if local_audit:
                # A malformed next task cannot discard valid option judgments.
                error = need_extraction_rejection
                need = None
                if raw_need is not None and need_extraction_rejection is None:
                    need, error = _validate_observation_need_candidate(raw_need, option_candidates=option_candidates)
                if error:
                    new_need_rejection = "new_observation_need_rejected:" + error
                    self._write_state("OBSERVATION_NEED_REJECTED", reason=error,
                        origin="VERIFIER_SEARCH", results_preserved=True)
                elif need is not None:
                    self._activate_or_update_search_need(semantic_need=need,
                        authority_ref=f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}")
            else:
                results = self._update_search_observation_need(
                    results=results, raw_need=raw_need,
                    extraction_rejection=need_extraction_rejection,
                    raw_resolution=raw_resolution,
                    resolution_extraction_rejection=resolution_extraction_rejection,
                    option_candidates=option_candidates, evidence_states=evidence_states,
                    question=str(kwargs.get("query") or ""),
                    plan_need_id=plan_need_id or None, reused_claim_ids=reused_claim_ids)
        elif raw_need is not None or need_extraction_rejection is not None:
            self._write_state(
                "OBSERVATION_NEED_REJECTED",
                reason=need_extraction_rejection or "malformed_output",
                origin="VERIFIER_SEARCH",
                authority_ref=(
                    f"hypothesis_verifier_revision_{self._hypothesis_verifier_revision}"
                ),
            )
        self._last_hypothesis_results = copy.deepcopy(results)
        self._hypothesis_verifier_status = (
            "COMPLETE" if not missing_options else "PARTIAL"
        )
        self._hypothesis_verifier_reason = (
            new_need_rejection
            if not missing_options
            else "missing_option_results:" + ",".join(missing_options)
        )
        if (getattr(self, "need_acquisition_enabled", False)
                and (self._last_search_need_resolution or {}).get("status") == "REJECTED"):
            self._hypothesis_verifier_status = "UNAVAILABLE"
            self._hypothesis_verifier_reason = "need_resolution_rejected:" + str(
                self._last_search_need_resolution.get("reason"))
        if missing_options:
            self._write_state(
                "HYPOTHESIS_VERIFIER_PARTIAL",
                evidence_claim_ids=[state["claim_id"] for state in evidence_states],
                expected_options=sorted(expected_options),
                received_options=sorted(received_options),
                missing_options=missing_options,
                observation_coverage=copy.deepcopy(self._observation_coverage),
                operator_gate_status=(
                    operator_gate_packet.get("status")
                    if operator_gate_packet is not None
                    else None
                ),
                results=results,
            )
        else:
            self._write_state(
                "HYPOTHESIS_VERIFIED",
                evidence_claim_ids=[state["claim_id"] for state in evidence_states],
                observation_coverage=copy.deepcopy(self._observation_coverage),
                operator_gate_status=(
                    operator_gate_packet.get("status")
                    if operator_gate_packet is not None
                    else None
                ),
                results=results,
            )
        return results

    def _unverified_fact_bundle(self) -> str | None:
        """Return timestamped Observer facts without question/option language."""

        facts: list[str] = []
        seen: set[str] = set()
        for round_id in sorted(self._unverified_observation_by_round):
            observation = self._unverified_observation_by_round[round_id]
            for item in observation.get("key_evidence", []):
                if not isinstance(item, Mapping):
                    continue
                description = str(item.get("description") or "").strip()
                if not description or description in seen:
                    continue
                seen.add(description)
                facts.append(description)
        if not facts:
            return None
        return "Observed factual claims requiring independent support:\n" + "\n".join(
            f"- {fact}" for fact in facts
        )

    def _has_answer_reviewable_facts(self) -> bool:
        """True when AVP has source observations available for global reasoning.

        This is only a prerequisite for reviewing AVP's answer.  It does not
        itself support any option. Any UNCERTAIN or CONFLICTING delta is
        still caught earlier by ``has_blocking_delta``.
        """

        return any(
            evidence_alignment_from_record(state) == "ALIGNED"
            for state in self._evidence_state_snapshot()
        )

    def _selected_hypothesis_is_contradicted(
        self,
        *,
        selected_option: str,
        options: list[str],
    ) -> bool:
        selected = _option_label(selected_option, options)
        if selected is None:
            return False
        current_states = self._evidence_state_snapshot()
        if getattr(self, "need_local_audit_enabled", False):
            from .verifier_compact_view import evidence_view
            current_states, _, _ = evidence_view(current_states, self._need_answers)
        evidence_states = {
            str(state.get("claim_id") or ""): state
            for state in current_states
        }
        for result in self._last_hypothesis_results:
            if str(result.get("option") or "").strip().upper() != selected:
                continue
            if str(result.get("verdict") or "").strip().upper() != "CONTRADICTED":
                return False
            claim_ids = [
                str(claim_id).strip()
                for claim_id in result.get("claim_ids", [])
                if str(claim_id).strip()
            ]
            return bool(claim_ids) and all(
                claim_id in evidence_states
                and evidence_alignment_from_record(evidence_states[claim_id])
                == "ALIGNED"
                and evidence_states[claim_id].get("latest") is not False
                for claim_id in claim_ids
            )
        return False

    def _supported_hypothesis_labels(self) -> list[str]:
        if self._hypothesis_verifier_status != "COMPLETE":
            return []
        return [
            str(result.get("option") or "").strip().upper()
            for result in self._last_hypothesis_results
            if str(result.get("option") or "").strip()
            and str(result.get("verdict") or "").strip().upper() == "SUPPORTED"
        ]

    def answer_route(self) -> Literal["VERIFIER_GAP", "SEARCH", "DECIDE", "UNIQUE", "COMPARE"]:
        """Describe whether Controller may enter Answer after verification."""

        if self._hypothesis_verifier_status != "COMPLETE":
            return "VERIFIER_GAP"
        if getattr(self, "need_acquisition_enabled", False):
            return "DECIDE"
        if (
            self._active_observation_need is not None
            and self._active_observation_need.get("origin") == "VERIFIER_SEARCH"
        ):
            return "SEARCH"
        supported = self._supported_hypothesis_labels()
        if not supported:
            return "SEARCH"
        if len(supported) == 1:
            return "UNIQUE"
        return "COMPARE"

    def answer_route_justification(self) -> str:
        route = self.answer_route()
        if route == "DECIDE":
            return "Assess current facts, option states and unresolved needs; decide whether to answer or seek a different observable lead."
        if route == "VERIFIER_GAP":
            return (
                "Hypothesis Verifier is incomplete or unavailable. Do not call "
                "Answer. Plan one observation that can produce auditable evidence "
                "for the unresolved option hypotheses."
            )
        if route == "SEARCH":
            if self._active_observation_need is not None:
                return self.observation_need_justification()
            unresolved = [
                str(result.get("option") or "").strip().upper()
                for result in self._last_hypothesis_results
                if str(result.get("verdict") or "").strip().upper() == "OPEN"
            ]
            suffix = f" Open options: {', '.join(unresolved)}." if unresolved else ""
            return (
                "No option is currently SUPPORTED by audited evidence. Do not "
                "guess or call Answer; plan the smallest observation that can "
                "support or contradict a decision-critical option difference."
                + suffix
            )
        return "Audited hypothesis states authorize Answer."

    def observation_need_justification(self, existing: str | None = None) -> str:
        need = self._active_observation_need
        if need is None:
            return str(existing or "").strip()
        prefix = str(existing or "").strip()
        request = (
            f"Active observation need {need['need_id']} revision {need['revision']}: "
            f"determine whether {need['missing_proposition']} "
            "because the observation can change options "
            f"{', '.join(need['affected_options'])}. "
            "Choose the smallest justified observation plan yourself; this need "
            "does not prescribe time, range, fps, sampling, or modality."
        )
        return "\n".join(part for part in (prefix, request) if part)

    def raise_for_verifier_failure(self) -> None:
        """Execution/format failure cannot authorize another visual observation."""
        if self._hypothesis_verifier_status in {"UNAVAILABLE", "PARTIAL"}:
            raise RuntimeError(
                "hypothesis_verifier_execution_failed:"
                + str(self._hypothesis_verifier_reason or self._hypothesis_verifier_status)
            )

    def _observation_plan_signature(self, plan: Any) -> tuple[Any, ...]:
        # This runtime is scoped to one video's QA run. IDs/revisions are not
        # observation content and cannot authorize rereading identical media.
        acquisition = getattr(plan, "acquisition", None) or {}
        if acquisition.get("action") == "asr":
            return ("asr", acquisition.get("tool"),
                    json.dumps(acquisition.get("arguments", {}), sort_keys=True))
        watch = getattr(plan, "watch", None)
        regions = tuple(sorted(set(
            (round(float(start), 6), round(float(end), 6))
            for start, end in list(getattr(watch, "regions", []) or [])
        )))
        spatial = getattr(watch, "spatial_token_rate", "")
        def normalized(value):
            return " ".join(str(value or "").casefold().split())
        return (
            normalized(getattr(watch, "load_mode", "")),
            round(float(getattr(watch, "fps", 0.0) or 0.0), 6),
            normalized(getattr(spatial, "value", spatial)),
            regions,
            normalized(getattr(plan, "description", "")),
            normalized(getattr(plan, "completion_criteria", "")),
        )

    def observation_plan_is_mechanical_repeat(self, plan: Any) -> bool:
        signature = self._observation_plan_signature(plan)
        return (
            signature is not None
            and signature in self._completed_observation_plan_signatures
        )

    def record_completed_observation_plan(
        self,
        plan: Any,
        evidence: Any,
        admission_event: Mapping[str, Any] | None,
    ) -> None:
        signature = self._observation_plan_signature(plan)
        if signature is None:
            return
        audit_completed = bool(
            isinstance(admission_event, Mapping)
            and isinstance(admission_event.get("evidence_audit_states"), list)
        )
        if audit_completed:
            self._completed_observation_plan_signatures.add(signature)

    def reject_mechanical_observation_repeat(self, plan: Any) -> None:
        signature = self._observation_plan_signature(plan)
        if signature is None or signature not in self._completed_observation_plan_signatures:
            return
        need = copy.deepcopy(self._active_observation_need)
        self._write_state(
            "OBSERVATION_PLAN_MECHANICAL_REPEAT_REJECTED",
            need_id=(need or {}).get("need_id"),
            revision=(need or {}).get("revision"),
        )
        self._close_active_observation_need(
            "UNSUPPORTED",
            reason=(
                "Planner repeated the same successful need, scope, fps, method, "
                "and reason without a new discriminant."
            ),
        )


    def _audit(self, packet: DeltaPacket, *, stage: str) -> tuple[AdmissionAlignment, AuditRecord]:
        request = build_admission_request(packet)
        raw = self.backend.call(stage, request)
        assessment = parse_admission_response(raw)
        alignment = derive_admission_alignment(packet, assessment)
        record = freeze_audit_record(
            packet,
            request,
            alignment,
            admission_request_id=f"{stage}_{self.backend.request_count:02d}",
        )
        _append_jsonl(self.audit_log_path, record.model_dump(mode="json"))
        return alignment, record

    def _freeze_raw_boundary(
        self,
        *,
        round_id: int,
        evidence: Any,
        receipts: tuple[SourceReceipt, ...],
    ) -> None:
        observation = {
            "detailed_response": str(evidence.detailed_response or ""),
            "key_evidence": copy.deepcopy(list(evidence.key_evidence or [])),
            "reasoning": str(evidence.reasoning or ""),
        }
        self._unverified_observation_by_round[round_id] = observation
        _append_jsonl(
            self.boundary_log_path,
            {
                "event": "RAW_OBSERVATION_BOUNDARY_FROZEN",
                "round_id": round_id,
                "rule": "SAME_GENERATION_CANNOT_SELF_SUPPORT",
                "source_fidelity": "UNKNOWN",
                "raw_sources": [
                    {
                        "source_id": receipt.record.source_id,
                        "provenance_root_id": receipt.record.provenance_root_id,
                        "locator": receipt.record.locator,
                        "producer": receipt.producer,
                        "transaction_id": receipt.transaction_id,
                    }
                    for receipt in receipts
                ],
                "unverified_observation": observation,
            },
        )

    def _classify_negative_observation(
        self,
        *,
        plan: Any,
        evidence: Any,
        round_id: int,
    ) -> str | None:
        if list(getattr(evidence, "key_evidence", []) or []):
            return None
        detailed_response = str(
            getattr(evidence, "detailed_response", "") or ""
        ).strip()
        observed_scopes = _observed_scope_payload(evidence)
        if not detailed_response or not observed_scopes:
            return None
        observation_goal = ""
        if not observation_goal:
            observation_goal = str(getattr(plan, "description", "") or "").strip()
        if not observation_goal:
            observation_goal = str(
                getattr(plan, "completion_criteria", "") or ""
            ).strip()
        try:
            raw = self.backend.call(
                "negative_observation_check",
                _negative_observation_request(
                    observation_goal=observation_goal,
                    observed_scopes=observed_scopes,
                    detailed_response=detailed_response,
                ),
            )
        except Exception as exc:
            self._write_state(
                "NEGATIVE_OBSERVATION_CHECK_UNAVAILABLE",
                round_id=round_id,
                error_type=type(exc).__name__,
            )
            return None
        observation = _parse_negative_observation(raw)
        self._write_state(
            "NEGATIVE_OBSERVATION_CHECK_COMPLETE",
            round_id=round_id,
            verdict=("NEGATIVE_OBSERVATION" if observation else "NOT_USABLE"),
        )
        return observation

    def _no_grounded_observation_event(self, *, round_id: int) -> dict[str, Any]:
        continuation_expected = (
            self._expecting_repair
            or self._expecting_revision
        )
        continuation_kind = (
            "CONTRADICTION_REVISION" if self._expecting_revision
            else "OPEN_SUPPORT_REPAIR" if self._expecting_repair
            else "OBSERVE"
        )
        self._expecting_repair = False
        self._expecting_revision = False
        self._consecutive_no_progress += 1
        self._saturate_current_open_check()
        event = {
            "source_alignment": (
                self._pending.alignment.source_alignment
                if self._pending
                else "UNCERTAIN"
            ),
            "permission": self._pending.permission if self._pending else "SEARCH_ONLY",
            "round_id": round_id,
            "continuation_kind": continuation_kind,
            "novel_provenance_roots": [],
            "gate_recalled": False,
            "open_check_status": (
                self._pending.open_check.status
                if self._pending and self._pending.open_check
                else None
            ),
        }
        self._write_state(
            (
                "REPAIR_NO_GROUNDED_OBSERVATION"
                if continuation_expected
                else "OBSERVATION_NO_PROGRESS"
            ),
            **event,
        )
        return event

    def admit_observation(
        self,
        *,
        query: str,
        plan: Any,
        evidence: Any,
        interaction_history: list[dict[str, Any]],
        current_confidence: float | None,
        video_path: str,
    ) -> dict[str, Any]:
        if (getattr(plan, "acquisition", None) or {}).get("action") == "asr":
            asr = getattr(self, "asr_context", None)
            if asr is None:
                raise ValueError("asr_mode_not_enabled")
            return asr.admit(self, plan=plan, evidence=evidence, video_path=video_path)
        round_id = int(evidence.round_id)
        if self._active_observation_need is not None:
            expected_need_id = str(
                self._active_observation_need.get("need_id") or ""
            )
            plan_need_id = str(getattr(plan, "need_id", "") or "")
            if plan_need_id != expected_need_id:
                self._write_state(
                    "OBSERVATION_NEED_PLAN_MISMATCH",
                    round_id=round_id,
                    expected_need_id=expected_need_id,
                    plan_need_id=plan_need_id or None,
                )
        model_call = evidence.model_call if isinstance(evidence.model_call, dict) else {}
        if self.raw_qwen_only and isinstance(model_call.get("returned_clips"), list):
            raise ValueError("raw_qwen_only_rejected_caption_receipt")
        try:
            observation_coverage = _validated_whole_video_coverage(
                evidence,
                video_path=video_path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._write_state(
                "OBSERVATION_COVERAGE_REJECTED",
                round_id=round_id,
                reason=str(exc),
            )
        else:
            if observation_coverage is not None:
                self._observation_coverage = observation_coverage
                self._write_state(
                    "OBSERVATION_COVERAGE_COMPLETE",
                    round_id=round_id,
                    observation_coverage=copy.deepcopy(observation_coverage),
                )
        evidence_for_admission = evidence
        try:
            receipts = extract_source_receipts(
                evidence_for_admission,
                round_id=round_id,
                video_path=video_path,
                preserve_response_sources=getattr(self, "need_local_audit_enabled", False),
            )
        except ValueError as exc:
            if str(exc) != "no_grounded_observation_statement":
                raise
            negative_observation = self._classify_negative_observation(
                plan=plan,
                evidence=evidence,
                round_id=round_id,
            )
            if negative_observation:
                evidence_for_admission = _with_scoped_negative_observation(
                    evidence,
                    negative_observation,
                )
                receipts = extract_source_receipts(
                    evidence_for_admission,
                    round_id=round_id,
                    video_path=video_path,
                )
            else:
                return self._no_grounded_observation_event(round_id=round_id)
        evidence = evidence_for_admission
        if self.raw_qwen_only and any(
            receipt.producer != "qwen_video_transport" for receipt in receipts
        ):
            raise ValueError("raw_qwen_only_requires_qwen_video_transport")
        if self._visual_evidence_auditor is not None and all(
            receipt.producer == "qwen_video_transport" for receipt in receipts
        ):
            return self._admit_visually_audited_observation(
                query=query,
                evidence=evidence,
                round_id=round_id,
                receipts=receipts,
                video_path=video_path,
            ) or {
                "status": "OPEN",
                "permission": "SEARCH_ONLY",
                "round_id": round_id,
            }
        if self._expecting_revision and self._pending is not None:
            return self._admit_revision_observation(
                round_id=round_id,
                receipts=receipts,
            )
        if self._expecting_repair and self._pending is not None:
            return self._reaudit_with_new_receipts(
                round_id=round_id,
                receipts=receipts,
            )
        if self.raw_qwen_only:
            self._freeze_raw_boundary(
                round_id=round_id,
                evidence=evidence,
                receipts=receipts,
            )
            receipts = _transport_only_receipts(receipts)
        self._round_receipts[round_id] = receipts
        for receipt in receipts:
            self._receipts_by_root.setdefault(receipt.record.provenance_root_id, receipt)
        if self.raw_qwen_only:
            event = {
                "status": "UNVERIFIED_OBSERVATION",
                "permission": "SEARCH_ONLY",
                "round_id": round_id,
                "source_fidelity": "UNKNOWN",
                "boundary_rule": "SAME_GENERATION_CANNOT_SELF_SUPPORT",
                "answer_commitment_deferred": True,
            }
            self._write_state("RAW_OBSERVATION_HELD_AS_HYPOTHESIS", **event)
            return event
        if self._selection_done:
            event = {"status": "SOURCE_ONLY", "round_id": round_id}
            self._write_state("OBSERVATION_SOURCE_ONLY", **event)
            return event

        self._selection_done = True
        all_sources = tuple(receipt.record for receipt in receipts)
        request = selector_request(
            question=query,
            candidate_text=str(evidence.detailed_response or ""),
            candidate_reasoning=str(evidence.reasoning or ""),
            interaction_history=interaction_history,
            current_confidence=current_confidence,
            source_records=all_sources,
        )
        try:
            raw_selection = self.backend.call("context_delta_selector", request)
            selection = DeltaSelection.model_validate(raw_selection)
        except ValidationError as exc:
            event = {
                "status": "SELECTOR_UNAVAILABLE",
                "permission": "SEARCH_ONLY",
                "round_id": round_id,
                "reason": "invalid_selector_output",
                "error_count": len(exc.errors()),
            }
            self._write_state("SELECTOR_UNAVAILABLE", **event)
            return event
        if not selection.delta_claim.strip():
            self._direct_source_facts_available = True
            event = {"status": "NO_DELTA", "round_id": round_id}
            self._write_state("SELECTOR_NO_DELTA", **event)
            return event
        by_id = {record.source_id: record for record in all_sources}
        unknown = set(selection.source_span_ids) - set(by_id)
        if unknown:
            event = {
                "status": "SELECTOR_UNAVAILABLE",
                "permission": "SEARCH_ONLY",
                "round_id": round_id,
                "reason": "unknown_selector_source_id",
            }
            self._write_state("SELECTOR_UNAVAILABLE", **event)
            return event
        selected_sources = tuple(by_id[source_id] for source_id in selection.source_span_ids)
        delta_id = f"round_{round_id}_answer_critical_delta"
        packet = DeltaPacket(
            case_id=self.run_dir.name,
            delta_id=delta_id,
            delta_claim=selection.delta_claim,
            source_records=selected_sources,
        )
        alignment, record = self._audit(packet, stage="evidence_admission_gate")
        self._record_evidence_state(
            packet=packet,
            alignment=alignment,
            round_id=round_id,
        )
        roots = tuple(record.unique_provenance_roots)
        self._pending = PendingDelta(
            delta_id=delta_id,
            packet=packet,
            selection=selection,
            alignment=alignment,
            permission=record.permission,
            revision=1,
            source_root_ids=roots,
            observed_root_ids=tuple(sorted(self._receipts_by_root)),
            observed_transaction_ids=tuple(
                sorted({receipt.transaction_id for receipt in self._receipts_by_root.values()})
            ),
            open_check=self._open_check(
                alignment=alignment,
                evidence_revision=1,
            ),
        )
        if alignment.source_alignment == "ALIGNED":
            self._admitted_delta_by_round[round_id] = selection.delta_claim
        event = {
            "source_alignment": alignment.source_alignment,
            "permission": record.permission,
            "round_id": round_id,
            "delta_id": delta_id,
            "delta_claim": selection.delta_claim,
            "witness_request": selection.witness_request,
            "open_check_status": (
                self._pending.open_check.status if self._pending.open_check else None
            ),
        }
        self._write_state("OBSERVATION_ADMISSION", **event)
        return event

    def _admit_revision_observation(
        self,
        *,
        round_id: int,
        receipts: tuple[SourceReceipt, ...],
    ) -> dict[str, Any]:
        """Retire one contradicted binding and expose only the new source facts."""
        assert self._pending is not None
        contradicted = self._pending
        self._expecting_revision = False
        self._consecutive_no_progress = 0
        if self.raw_qwen_only:
            stored_receipts = _transport_only_receipts(receipts)
        else:
            stored_receipts = receipts
        self._round_receipts[round_id] = stored_receipts
        for receipt in stored_receipts:
            self._receipts_by_root.setdefault(
                receipt.record.provenance_root_id,
                receipt,
            )
        self._direct_source_facts_available = any(
            receipt.record.source_kind in {"caption", "observer_claim"}
            for receipt in stored_receipts
        )
        self._pending = None
        event = {
            "status": "REVISION_SOURCE_ONLY",
            "permission": "SEARCH_ONLY",
            "round_id": round_id,
            "retired_delta_id": contradicted.delta_id,
            "retired_claim": contradicted.packet.delta_claim,
            "retired_source_alignment": contradicted.alignment.source_alignment,
            "novel_provenance_roots": [
                receipt.record.provenance_root_id for receipt in stored_receipts
            ],
        }
        event_name = (
            "CONTRADICTION_REVISION_OBSERVATION"
            if contradicted.alignment.source_alignment == "CONFLICTING"
            else "STALLED_BINDING_REVISION_OBSERVATION"
        )
        self._write_state(event_name, **event)
        return event

    def _reaudit_with_new_receipts(
        self,
        *,
        round_id: int,
        receipts: tuple[SourceReceipt, ...],
    ) -> dict[str, Any]:
        assert self._pending is not None
        self._expecting_repair = False
        old_roots = set(self._pending.observed_root_ids)
        old_transactions = set(self._pending.observed_transaction_ids)
        previous_receipts = tuple(self._receipts_by_root.values())
        if self.raw_qwen_only:
            novel = tuple(
                receipt
                for receipt in receipts
                if receipt.record.provenance_root_id not in old_roots
                and receipt.transaction_id not in old_transactions
                and _adds_new_raw_observation(receipt, previous_receipts)
            )
        else:
            novel = tuple(
                receipt
                for receipt in receipts
                if receipt.record.provenance_root_id not in old_roots
                and receipt.transaction_id not in old_transactions
            )
        if not novel:
            self._consecutive_no_progress += 1
            self._saturate_current_open_check()
            stored_receipts = (
                _transport_only_receipts(receipts) if self.raw_qwen_only else receipts
            )
            self._round_receipts[round_id] = stored_receipts
            for receipt in stored_receipts:
                self._receipts_by_root.setdefault(
                    receipt.record.provenance_root_id,
                    receipt,
                )
            event = {
                "source_alignment": self._pending.alignment.source_alignment,
                "permission": self._pending.permission,
                "revision": self._pending.revision,
                "novel_provenance_roots": [],
                "gate_recalled": False,
                "open_check_status": (
                    self._pending.open_check.status
                    if self._pending.open_check
                    else None
                ),
            }
            self._write_state("REPAIR_NO_NEW_PROVENANCE", **event)
            return event

        self._consecutive_no_progress = 0
        combined: dict[str, SourceRecord] = {
            record.provenance_root_id: record for record in self._pending.packet.source_records
        }
        for receipt in novel:
            combined[receipt.record.provenance_root_id] = receipt.record
        packet = DeltaPacket(
            case_id=self._pending.packet.case_id,
            delta_id=self._pending.packet.delta_id,
            delta_claim=self._pending.packet.delta_claim,
            source_records=tuple(combined[root] for root in sorted(combined)),
        )
        alignment, record = self._audit(packet, stage="repair_readmission_gate")
        self._record_evidence_state(
            packet=packet,
            alignment=alignment,
            round_id=round_id,
        )
        if not self.raw_qwen_only:
            stored_receipts = receipts
        elif alignment.source_alignment == "ALIGNED":
            admitted_source_ids = set(alignment.source_span_ids) & {
                receipt.record.source_id for receipt in novel
            }
            stored_receipts = tuple(
                receipt
                if receipt.record.source_id in admitted_source_ids
                else _transport_only_receipts((receipt,))[0]
                for receipt in receipts
            )
        else:
            stored_receipts = _transport_only_receipts(receipts)
        self._round_receipts[round_id] = stored_receipts
        for receipt in stored_receipts:
            self._receipts_by_root.setdefault(
                receipt.record.provenance_root_id,
                receipt,
            )
        revision = self._pending.revision + 1
        next_open_check = self._open_check(
            alignment=alignment,
            evidence_revision=revision,
        )
        if next_open_check is not None:
            next_open_check = replace(
                next_open_check,
                status="SATURATED",
                executed=True,
            )
        self._pending = PendingDelta(
            delta_id=self._pending.delta_id,
            packet=packet,
            selection=self._pending.selection,
            alignment=alignment,
            permission=record.permission,
            revision=revision,
            source_root_ids=tuple(record.unique_provenance_roots),
            observed_root_ids=tuple(sorted(self._receipts_by_root)),
            observed_transaction_ids=tuple(
                sorted({receipt.transaction_id for receipt in self._receipts_by_root.values()})
            ),
            open_check=next_open_check,
        )
        if alignment.source_alignment == "ALIGNED":
            self._admitted_delta_by_round[max(self._round_receipts)] = packet.delta_claim
        event = {
            "source_alignment": alignment.source_alignment,
            "permission": record.permission,
            "revision": revision,
            "novel_provenance_roots": [item.record.provenance_root_id for item in novel],
            "gate_recalled": True,
            "open_check_status": (
                self._pending.open_check.status if self._pending.open_check else None
            ),
        }
        self._write_state("REPAIR_READMISSION", **event)
        return event

    def project_evidence_list(
        self,
        evidences: list[Any],
        *,
        include_unresolved: bool = False,
    ) -> list[Any]:
        projected: list[Any] = []
        for evidence in evidences:
            clone = copy.deepcopy(evidence)
            round_id = int(evidence.round_id)
            if self._visual_evidence_auditor is not None:
                states = [
                    copy.deepcopy(state)
                    for state in self._evidence_state_snapshot()
                    if state.get("round_id") == round_id
                    and state.get("latest") is not False
                ]
                lines: list[str] = []
                aligned_sources: list[str] = []
                for state in states:
                    source_alignment = evidence_alignment_from_record(state)
                    if source_alignment != "ALIGNED" and not include_unresolved:
                        continue
                    planner_only = source_alignment == "UNCERTAIN" and any(
                        str(reason).startswith("planner_only_unparsed_fact:")
                        for reason in state.get("missing_premises", [])
                    )
                    if planner_only:
                        lines.append(
                            "[Planner-only unparsed hint; first decide whether this "
                            "text contains a concrete, observable, question-relevant "
                            "fact. Discard garbage, vague summaries, answer-only "
                            "assertions, or repetition. If useful, use it only to "
                            "plan a new observation. This unbound text is not itself "
                            "Evidence; a later observation may establish Evidence] "
                            f"{state['claim']}"
                        )
                        continue
                    qualifier = (
                        "answer-eligible fact"
                        if source_alignment == "ALIGNED"
                        else (
                            "unresolved; not answer support"
                            if source_alignment == "UNCERTAIN"
                            else "constrained; must not be used as a premise"
                        )
                    )
                    lines.append(
                        f"[Evidence Audit {source_alignment}; {qualifier}; "
                        f"claim_id={state['claim_id']}; "
                        f"claim_scope={state.get('claim_scope')}; source_scopes={state.get('source_scopes', [])}; "
                        f"observation_uncertainty={state.get('observation_uncertainty')}; "
                        f"local facts do not certify global conclusions] {state['claim']}"
                    )
                    if source_alignment == "ALIGNED":
                        aligned_sources.extend(state.get("source_ids", []))
                clone.detailed_response = "\n".join(lines)
                clone.reasoning = ""
                aligned_claims = {
                    state["claim"]: state
                    for state in states
                    if evidence_alignment_from_record(state) == "ALIGNED"
                    and state.get("latest") is not False
                    and "_observe_span_" in state["claim_id"]
                }
                clone.key_evidence = []
                for item in evidence.key_evidence:
                    if not isinstance(item, Mapping):
                        continue
                    description = str(item.get("description") or "").strip()
                    state = aligned_claims.get(description)
                    if state is None:
                        continue
                    timestamp_start = item.get("timestamp_start")
                    timestamp_end = item.get("timestamp_end")
                    if timestamp_start is None or timestamp_end is None:
                        source_ids = set(state.get("source_ids", []))
                        source_receipts = [
                            receipt
                            for receipt in self._round_receipts.get(round_id, ())
                            if receipt.record.source_id in source_ids
                        ]
                        if source_receipts:
                            timestamp_start = min(
                                receipt.start_sec for receipt in source_receipts
                            )
                            timestamp_end = max(
                                receipt.end_sec for receipt in source_receipts
                            )
                    clone.key_evidence.append(
                        {
                            "timestamp_start": timestamp_start,
                            "timestamp_end": timestamp_end,
                            "description": description,
                            "claim_id": state["claim_id"],
                            "source_ids": state.get("source_ids", []),
                        }
                    )
                clone.model_call = {
                    "producer": "evidence_audit_projection",
                    "source_ids": list(dict.fromkeys(aligned_sources)),
                }
                projected.append(clone)
                continue
            receipts = self._round_receipts.get(int(evidence.round_id), ())
            lines = [
                (
                    "[Exact caption source "
                    if receipt.record.source_kind == "caption"
                    else (
                        "[Observer claim "
                        if receipt.record.source_kind == "observer_claim"
                        else "[Transport receipt "
                    )
                )
                + f"{receipt.record.source_id} @ {receipt.record.locator}] "
                f"{receipt.record.exact_caption}"
                for receipt in receipts
            ]
            unverified = self._unverified_observation_by_round.get(
                int(evidence.round_id)
            )
            if unverified:
                hypothesis = str(
                    unverified.get("detailed_response")
                    or unverified.get("reasoning")
                    or ""
                ).strip()
                if hypothesis:
                    lines.append(
                        "[Unverified observation hypothesis; search context only, not "
                        f"answer support] {hypothesis}"
                    )
            admitted = self._admitted_delta_by_round.get(int(evidence.round_id))
            if admitted:
                lines.append(f"[Admitted derived claim] {admitted}")
            clone.detailed_response = "\n".join(lines)
            clone.reasoning = ""
            clone.key_evidence = [
                {
                    "timestamp_start": receipt.start_sec,
                    "timestamp_end": receipt.end_sec,
                    "description": receipt.record.exact_caption,
                    "source_id": receipt.record.source_id,
                }
                for receipt in receipts
            ]
            clone.model_call = {
                "producer": "semantic_delta_answer_projection",
                "source_ids": [receipt.record.source_id for receipt in receipts],
            }
            projected.append(clone)
        return projected

    def observation_blackboard(self, blackboard: Any) -> Any:
        if not (
            self._expecting_repair
            or self._expecting_revision
        ):
            return (
                self.project_blackboard(blackboard)
                if self._visual_evidence_auditor is not None
                else blackboard
            )
        clone = copy.copy(blackboard)
        clone.evidences = []
        clone.meta = copy.deepcopy(blackboard.meta)
        clone.query_confidence = None
        return clone

    def project_blackboard(self, blackboard: Any) -> Any:
        clone = copy.copy(blackboard)
        clone.evidences = self.project_evidence_list(list(blackboard.evidences))
        clone.meta = copy.deepcopy(blackboard.meta)
        clone.query_confidence = None
        return clone

    def project_planning_blackboard(self, blackboard: Any) -> Any:
        clone = copy.copy(blackboard)
        clone.evidences = self.project_evidence_list(
            list(blackboard.evidences),
            include_unresolved=True,
        )
        clone.meta = copy.deepcopy(blackboard.meta)
        clone.meta["evidence_audit_states"] = self._evidence_state_snapshot()
        clone.meta["hypothesis_verifier_status"] = self._hypothesis_verifier_status
        clone.meta["hypothesis_verifier_results"] = copy.deepcopy(
            self._last_hypothesis_results
        )
        clone.meta["hypothesis_verifier_reason"] = self._hypothesis_verifier_reason
        clone.meta["observation_coverage"] = copy.deepcopy(
            self._observation_coverage
        )
        clone.meta["active_observation_need"] = copy.deepcopy(
            self._active_observation_need
        )
        clone.meta["observation_need_history"] = copy.deepcopy(
            self._observation_need_history
        )
        clone.query_confidence = None
        return clone

    def has_blocking_delta(self) -> bool:
        return (
            self._commitment_rejection is not None
            or (
                self._pending is not None
                and self._pending.alignment.source_alignment != "ALIGNED"
            )
        )

    def can_dispatch_repair(self) -> bool:
        if getattr(self, "need_local_audit_enabled", False):
            current = self._need_audit_current or {}
            active = self._active_observation_need or {}
            return current.get("action") == "repair" and current.get("need_id") == active.get("need_id")
        if (self.max_repair_observations is not None and self._repair_observations >= self.max_repair_observations):
            return False
        if self._pending is None:
            return False
        if self._pending.alignment.source_alignment == "CONFLICTING":
            return True
        check = self._pending.open_check
        return (
            self._pending.alignment.source_alignment == "UNCERTAIN"
            and check is not None
            and check.status == "ACTIONABLE"
            and not check.executed
        )

    def repair_budget_exhausted(self) -> bool:
        return (
            self._pending is not None
            and (self.max_repair_observations is not None and self._repair_observations >= self.max_repair_observations)
        )

    def should_revise_current_binding(self) -> bool:
        return (
            self._pending is not None
            and self._pending.alignment.source_alignment == "CONFLICTING"
        )

    def repair_justification(self, existing: str | None = None) -> str:
        if getattr(self, "need_local_audit_enabled", False):
            if not self.can_dispatch_repair():
                raise ValueError("no_local_need_repair")
            return ("Unresolved evidence gap: " + self._need_audit_current["gap"]
                    + " Decide the smallest concrete observation that could resolve it. The diagnosis is not evidence.")
        if self._pending is None:
            raise ValueError("no_pending_delta")
        prefix = "" if self.raw_qwen_only else str(existing or "").strip()
        missing = "; ".join(self._pending.alignment.missing_premises)
        request = self._pending.selection.witness_request.strip()
        if self._pending.alignment.source_alignment == "CONFLICTING":
            conflicts = "; ".join(
                f"{quote.source_id}: {quote.exact_quote}"
                for quote in self._pending.alignment.conflict_quotes
            )
            request = (
                "The previous binding is CONFLICTING and must not be used as an "
                f"established premise: {self._pending.packet.delta_claim} "
                "Revise or replace that candidate and use the next observation to "
                "distinguish the remaining alternatives."
            )
            if conflicts:
                request += f" Conflicting source content: {conflicts}"
        elif self._pending.open_check is not None and self._pending.open_check.diagnosis:
            request = (
                "Unresolved evidence gap: "
                f"{self._pending.open_check.diagnosis} "
                "Decide the smallest concrete video observation or search that could "
                "resolve it. The diagnosis is not evidence."
            )
        elif self._pending.delta_id == "answer_commitment" and missing:
            request = (
                "Acquire a physically new bounded observation that directly resolves "
                f"this missing premise: {missing}"
            )
        elif not request:
            request = "Acquire a new observation that directly supports or refutes the pending claim."
        return "\n".join(part for part in (prefix, f"Evidence admission witness request: {request}") if part)

    def mark_repair_dispatched(self) -> None:
        if getattr(self, "need_local_audit_enabled", False):
            self._repair_observations += 1
            self._write_state("NEED_REPAIR_DISPATCHED", need_answer=self._need_audit_current)
            return
        if (
            (self.max_repair_observations is not None and self._repair_observations >= self.max_repair_observations)
            or self._pending is None
        ):
            raise ValueError("repair_observation_budget_exhausted")
        source_alignment = self._pending.alignment.source_alignment
        if source_alignment == "UNCERTAIN":
            check = self._pending.open_check
            if (
                check is None
                or check.status != "ACTIONABLE"
                or check.executed
            ):
                raise ValueError("open_check_not_actionable")
            self._executed_open_diagnoses.add(
                self._open_diagnosis_key(check.diagnosis)
            )
            self._pending = replace(
                self._pending,
                open_check=replace(check, executed=True),
            )
        self._repair_observations += 1
        self._expecting_revision = self.should_revise_current_binding()
        self._expecting_repair = not self._expecting_revision
        event_name = (
            "CONTRADICTION_REVISION_DISPATCHED"
            if source_alignment == "CONFLICTING"
            else (
                "STALLED_BINDING_REVISION_DISPATCHED"
                if self._expecting_revision
                else "REPAIR_DISPATCHED"
            )
        )
        self._write_state(
            event_name,
            delta_id=self._pending.delta_id,
            witness_request=self._pending.selection.witness_request,
            continuation_kind=(
                "CONTRADICTION_REVISION" if self._expecting_revision else "OPEN_SUPPORT_REPAIR"
            ),
            repair_observation_index=self._repair_observations,
            repair_observation_budget=self.max_repair_observations,
            open_check=(
                {
                    "status": self._pending.open_check.status,
                    "diagnosis": self._pending.open_check.diagnosis,
                    "evidence_revision": self._pending.open_check.evidence_revision,
                    "executed": self._pending.open_check.executed,
                }
                if self._pending.open_check
                else None
            ),
        )

    def audit_answer_candidate(
        self,
        final: Mapping[str, Any],
        *,
        question: str,
        options: list[str] | None,
    ) -> bool:
        self._commitment_rejection = None
        self._blocked_candidate = copy.deepcopy(dict(final))
        if self.has_blocking_delta():
            self._write_state(
                "ANSWER_BLOCKED_BY_PENDING_DELTA",
                permission=self._pending.permission if self._pending else "SEARCH_ONLY",
            )
            return False
        selected_text, binding_error = _build_answer_binding_claim(
            final=final,
            options=options,
            question=question,
        )
        if binding_error is not None:
            self._commitment_rejection = binding_error
            self._write_state(
                "ANSWER_BLOCKED_INVALID_BINDING",
                reason=binding_error,
            )
            return False
        if selected_text is None or not self._receipts_by_root:
            self._commitment_rejection = "empty_or_sourceless_answer_binding"
            self._write_state("ANSWER_BLOCKED_EMPTY_OR_SOURCELESS")
            return False

        supported_hypotheses: list[str] = []
        selected_option = str(final.get("selected_option") or "")
        normalized_selected = _option_label(selected_option, options or [])
        if options:
            if self._hypothesis_verifier_status != "COMPLETE":
                self._commitment_rejection = "hypothesis_verifier_not_complete"
                self._write_state(
                    "ANSWER_BLOCKED_VERIFIER_GAP",
                    selected_option=selected_option,
                    hypothesis_verifier_status=self._hypothesis_verifier_status,
                    hypothesis_verifier_reason=self._hypothesis_verifier_reason,
                    hypothesis_states=copy.deepcopy(self._last_hypothesis_results),
                )
                return False
            supported_hypotheses = self._supported_hypothesis_labels()
            if getattr(self, "need_acquisition_enabled", False):
                unresolved = (normalized_selected not in supported_hypotheses
                              or self._active_observation_need is not None)
                if isinstance(final, dict):
                    forced = final.get("forced_at_budget") is True
                    final["answer_mode"] = "budget_forced" if forced else ("best_effort" if unresolved else "supported")
                    final["admission_unresolved"] = unresolved or forced
                    final["forced_at_budget"] = forced
                    final["observation_need"] = copy.deepcopy(self._active_observation_need)
                self._write_state("ANSWER_COMMITMENT", accepted=True,
                    selected_option=normalized_selected, permission="CANDIDATE_RETAINED" if unresolved else "ANSWER_ELIGIBLE",
                    hypothesis_states=copy.deepcopy(self._last_hypothesis_results),
                    unresolved_need=copy.deepcopy(self._active_observation_need))
                return True
            if not supported_hypotheses:
                self._commitment_rejection = "no_supported_hypothesis"
                self._write_state(
                    "ANSWER_BLOCKED_NO_SUPPORTED_HYPOTHESIS",
                    selected_option=selected_option,
                    hypothesis_states=copy.deepcopy(self._last_hypothesis_results),
                )
                return False
            if normalized_selected not in supported_hypotheses:
                self._commitment_rejection = "selected_hypothesis_not_supported"
                self._write_state(
                    "ANSWER_BLOCKED_UNSUPPORTED_SELECTION",
                    selected_option=selected_option,
                    supported_options=supported_hypotheses,
                    hypothesis_states=copy.deepcopy(self._last_hypothesis_results),
                )
                return False

        if self._has_answer_reviewable_facts():
            if (
                options
                and self._selected_hypothesis_is_contradicted(
                    selected_option=selected_option,
                    options=options,
                )
            ):
                self._commitment_rejection = "selected_hypothesis_contradicted"
                self._write_state(
                    "ANSWER_BLOCKED_HYPOTHESIS_CONTRADICTED",
                    selected_option=selected_option,
                    hypothesis_states=copy.deepcopy(self._last_hypothesis_results),
                )
                return False
            unique_supported = bool(options) and len(supported_hypotheses) == 1
            commitment_status = (
                "AVP_GLOBAL_ANSWER_FACT_SUPPORTED"
                if not options
                else (
                    "AVP_GLOBAL_ANSWER_UNIQUE_SUPPORTED"
                    if unique_supported
                    else "AVP_GLOBAL_ANSWER_SUPPORTED_SET_SELECTION"
                )
            )
            boundary_rule = (
                "ANSWER_MAPPED_FROM_AUDITED_FACTS"
                if not options
                else (
                    "ANSWER_CONFIRMED_SOLE_SUPPORTED_HYPOTHESIS"
                    if unique_supported
                    else "ANSWER_SELECTED_WITHIN_SUPPORTED_HYPOTHESIS_SET"
                )
            )
            self._write_state(
                "ANSWER_COMMITMENT",
                status=commitment_status,
                permission="ANSWER_ELIGIBLE",
                accepted=True,
                boundary_rule=boundary_rule,
                supported_options=supported_hypotheses,
                hypothesis_states=copy.deepcopy(self._last_hypothesis_results),
                admitted_fact_rounds=sorted(self._admitted_delta_by_round),
            )
            return True

        records = tuple(
            self._receipts_by_root[root].record for root in sorted(self._receipts_by_root)
        )
        if self.raw_qwen_only and self._repair_observations == 0:
            fact_bundle = self._unverified_fact_bundle()
            if fact_bundle is None:
                self._commitment_rejection = "no_timestamped_observer_facts"
                self._write_state("ANSWER_BLOCKED_NO_OBSERVER_FACTS")
                return False
            packet = DeltaPacket(
                case_id=self.run_dir.name,
                delta_id="answer_factual_basis",
                delta_claim=fact_bundle,
                source_records=records,
            )
            request = build_admission_request(packet)
            alignment = AdmissionAlignment(
                source_alignment="UNCERTAIN",
                permission="SEARCH_ONLY",
                source_span_ids=(),
                support_quotes=(),
                conflict_quotes=(),
                missing_premises=(
                    "An independently generated semantic observation is required.",
                ),
            )
            record = freeze_audit_record(
                packet,
                request,
                alignment,
                admission_request_id="answer_commitment_boundary_00",
            )
            _append_jsonl(self.audit_log_path, record.model_dump(mode="json"))
            selection = DeltaSelection(
                delta_claim=fact_bundle,
                source_span_ids=tuple(record.source_id for record in records),
                witness_request=(
                    "Independently re-observe a bounded raw-video scope and report "
                    "directly visible or audible facts that confirm or contradict the "
                    "pending factual basis. Do not evaluate an answer option."
                ),
            )
            self._pending = PendingDelta(
                delta_id=packet.delta_id,
                packet=packet,
                selection=selection,
                alignment=alignment,
                permission=record.permission,
                revision=1,
                source_root_ids=tuple(record.unique_provenance_roots),
                observed_root_ids=tuple(sorted(self._receipts_by_root)),
                observed_transaction_ids=tuple(
                    sorted(
                        {
                            receipt.transaction_id
                            for receipt in self._receipts_by_root.values()
                        }
                    )
                ),
                open_check=self._open_check(
                    alignment=alignment,
                    evidence_revision=1,
                ),
            )
            self._write_state(
                "ANSWER_COMMITMENT",
                source_alignment="UNCERTAIN",
                permission=record.permission,
                accepted=False,
                boundary_rule="FACTUAL_BASIS_NEEDS_INDEPENDENT_OBSERVATION",
            )
            return False
        self._commitment_rejection = "no_answer_eligible_facts"
        self._write_state(
            "ANSWER_COMMITMENT",
            source_alignment="UNCERTAIN",
            permission="SEARCH_ONLY",
            accepted=False,
            boundary_rule="NO_ANSWER_ELIGIBLE_FACTS",
        )
        return False

    def unresolved_final(self) -> dict[str, Any]:
        effective_rejection = self._commitment_rejection
        if effective_rejection is not None:
            permission = "SEARCH_ONLY"
            source_alignment = "UNCERTAIN"
            reasoning = "Answer commitment was rejected before admission."
        else:
            permission = self._pending.permission if self._pending else "SEARCH_ONLY"
            source_alignment = (
                self._pending.alignment.source_alignment
                if self._pending
                else "UNCERTAIN"
            )
            reasoning = "Evidence admission did not authorize answer commitment."
        return {
            "status": "UNRESOLVED",
            "selected_option": "",
            "selected_option_text": "",
            "reasoning": reasoning,
            "confidence": 0.0,
            "admission": {
                "source_alignment": source_alignment,
                "permission": permission,
                "repair_observations": self._repair_observations,
                "blocked_candidate": self._blocked_candidate,
                "commitment_rejection": effective_rejection,
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "need_answers": copy.deepcopy(getattr(self, "_need_answers", [])),
            "need_answer_audit_calls": getattr(self, "_need_audit_calls", 0),
            "evidence_audit_states": self._evidence_state_snapshot(),
            "evidence_audit_requests": (
                self._visual_evidence_auditor.request_count
                if self._visual_evidence_auditor is not None
                else 0
            ),
            "hypothesis_verifier_results": copy.deepcopy(
                self._last_hypothesis_results
            ),
            "hypothesis_verifier_status": self._hypothesis_verifier_status,
            "hypothesis_verifier_reason": self._hypothesis_verifier_reason,
            "hypothesis_verifier_revision": self._hypothesis_verifier_revision,
            "operator_gate_mode": self.operator_gate_mode,
            "operator_gate_packet": copy.deepcopy(self._operator_gate_packet),
            "operator_gate_revision": self._operator_gate_revision,
            "operator_gate_request_count": self._operator_gate_request_count,
            "observation_coverage": copy.deepcopy(self._observation_coverage),
            "active_observation_need": copy.deepcopy(
                self._active_observation_need
            ),
            "observation_need_history": copy.deepcopy(
                self._observation_need_history
            ),
            "observation_need_recovery_attempted": (
                self._observation_need_recovery_attempted
            ),
            "observation_need_recovery_status": (
                self._observation_need_recovery_status
            ),
            "last_search_need_resolution": copy.deepcopy(
                self._last_search_need_resolution
            ),
            "answer_route": self.answer_route(),
            "supported_hypothesis_options": self._supported_hypothesis_labels(),
            "selector_calls": 1 if self._selection_done else 0,
            "audit_calls": len(self.audit_log_path.read_text(encoding="utf-8").splitlines())
            if self.audit_log_path.exists()
            else 0,
            "repair_observations": self._repair_observations,
            "max_repair_observations": self.max_repair_observations,
            "pending_source_alignment": (
                self._pending.alignment.source_alignment if self._pending else None
            ),
            "pending_permission": self._pending.permission if self._pending else None,
            "open_check": (
                {
                    "status": self._pending.open_check.status,
                    "diagnosis": self._pending.open_check.diagnosis,
                    "evidence_revision": self._pending.open_check.evidence_revision,
                    "executed": self._pending.open_check.executed,
                }
                if self._pending and self._pending.open_check
                else None
            ),
            "commitment_rejection": self._commitment_rejection,
            "model_requests": self.backend.request_count,
        }
