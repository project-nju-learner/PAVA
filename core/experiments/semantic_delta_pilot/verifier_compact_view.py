"""Lossless tool-evidence view for Verifier; Audit keeps its original working state."""
import copy
import json


STATE_FIELDS = (
    "claim_id", "claim", "source_alignment", "source_ids", "latest", "round_id",
    "missing_premises", "updates_claim_ids", "superseded_by_claim_ids",
    "observation_uncertainty", "claim_scope", "source_scopes", "audit_mode",
    "semantic_verification", "evidence_kind", "modality", "input_asr_guidance",
)
NEED_FIELDS = ("need_id", "search_need", "gap", "reference_diagnostics", "acquisition_diagnostic")
ATTEMPT_FIELDS = (
    "need_id", "round_id", "task", "modality", "acquisition_reason", "requested", "diagnostic",
)


def _model_state(state):
    """Use the visual experiments' compact fields, with ASR source types intact."""
    item = {k: copy.deepcopy(state[k]) for k in STATE_FIELDS if k in state}
    item["source_scopes"] = [{k: v for k, v in scope.items() if k != "source_video_path"}
                             for scope in item.get("source_scopes") or []]
    uncertainty = item.get("observation_uncertainty")
    if isinstance(uncertainty, dict):
        uncertainty.pop("recognition_metrics", None)
    guidance = item.get("input_asr_guidance")
    if isinstance(guidance, dict):
        item["input_asr_guidance"] = {k: v for k, v in guidance.items()
                                     if k not in ("source", "records", "text")}
    return {k: v for k, v in item.items() if v not in (None, [], {})
            or k in ("source_ids", "source_alignment", "latest")}


def evidence_view(evidence_states, need_answers):
    """Return full citation bindings plus compact needs, without mutating stores.

    Existing states own lifecycle. Raw response IDs are also citable, subject to
    their recorded source qualification; their prose is not semantically certified.
    """
    states = {s["claim_id"]: copy.deepcopy(s) for s in evidence_states
              if s.get("evidence_kind") != "need_answer"}
    origins = {}
    conflicts = {}
    needs = []
    for answer in need_answers:
        need = {k: copy.deepcopy(answer[k]) for k in NEED_FIELDS if k in answer}
        attempts = []
        records = [(ref, record, None) for ref, record in answer.get("materials", {}).items()]
        for attempt in answer.get("attempts", []):
            brief = {k: copy.deepcopy(attempt[k]) for k in ATTEMPT_FIELDS if k in attempt}
            brief["response_ref_ids"] = [r["ref_id"] for r in attempt.get("responses", [])]
            if brief not in attempts:
                attempts.append(brief)
            records.extend((r["ref_id"], r, attempt.get("asr_source"))
                           for r in attempt.get("responses", []))
        need["attempts"] = attempts
        need["material_ref_ids"] = list(answer.get("materials", {}))
        for ref, record, origin in records:
            # No invented IDs. If a saved ID has incompatible text/origin, keep
            # both records visible as a conflict instead of silently coalescing.
            identity = ({k: origin[k] for k in ("origin_run", "producer", "audio_path", "video_id")
                         if k in origin} if origin else {})
            mismatch = (ref in states and states[ref]["claim"] != record["text"])
            mismatch |= bool(identity and ref in origins and origins[ref] != identity)
            if identity and ref not in origins:
                origins[ref] = identity
            if mismatch:
                variant = copy.deepcopy(record)
                if origin:
                    variant["asr_source"] = copy.deepcopy(origin)
                variants = conflicts.setdefault(ref, [])
                if variant not in variants:
                    variants.append(variant)
                states[ref]["source_alignment"] = "UNCERTAIN"
                continue
            if ref in states:
                if origin:
                    states[ref].setdefault("asr_source", copy.deepcopy(origin))
                # A historical material table cannot revive a superseded state.
                for key in ("input_asr_guidance", "modality", "semantic_verification"):
                    if key in record:
                        states[ref].setdefault(key, copy.deepcopy(record[key]))
                continue
            modality = record.get("modality")
            is_asr = modality == "asr_transcript"
            # A historical atomic claim missing from the state store is diagnostic.
            raw_response = is_asr or modality == "sampled_JPEG_frames_no_audio"
            states[ref] = {
                "claim_id": ref, "claim": record["text"],
                "source_alignment": record.get("source_alignment", "ALIGNED")
                    if raw_response and record.get("source_qualified") else "UNCERTAIN",
                "source_ids": copy.deepcopy(record.get("source_ids", [])),
                "latest": record.get("latest") is not False,
                "audit_mode": "receipt_only",
                "evidence_kind": "asr_segment" if is_asr else "observe_response" if raw_response else "historical_material",
                "semantic_verification": record.get("semantic_verification", "NOT_PERFORMED"),
                **{k: copy.deepcopy(record[k]) for k in STATE_FIELDS
                   if k in record and k not in ("claim_id", "claim", "source_ids", "source_alignment", "latest", "evidence_kind")},
            }
            if origin:
                states[ref]["asr_source"] = copy.deepcopy(origin)
        needs.append(need)
    # Preserve colliding text/origin in diagnostics; no global failure or new gate.
    for ref in conflicts:
        states[ref]["source_alignment"] = "UNCERTAIN"
    return list(states.values()), needs, conflicts


def project(request):
    """Project once before sending; return full bindings for existing parsers."""
    out = copy.deepcopy(request)
    payload = json.loads(out["messages"][1]["content"])
    states, needs, conflicts = evidence_view(payload["evidence_audit_states"], payload.get("need_answers", []))
    payload["evidence_audit_states"] = [_model_state(state) for state in states]
    if "need_answers" in payload:
        payload["need_answers"] = needs
    if conflicts:
        payload["material_reference_conflicts"] = {
            ref: [{"claim_id": ref, "claim": variant["text"]} for variant in variants]
            for ref, variants in conflicts.items()}
    if any(s.get("evidence_kind") == "asr_segment" for s in states):
        out["messages"][0]["content"] += (
            " ASR source_scopes describe estimated utterance times, not necessarily the times "
            "of the events described. ASR origin qualification does not verify recognition "
            "or speaker identity; input_asr_guidance marks text supplied to a visual report."
        )
    out["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    from .avp_integration import _estimate_request_input_tokens
    out["estimated_input_tokens"] = _estimate_request_input_tokens(out["messages"])
    return out, states, {
        "original_user_chars": len(request["messages"][1]["content"]),
        "compact_user_chars": len(out["messages"][1]["content"]),
        "projected_evidence_count": len(states),
        "conflicting_reference_ids": list(conflicts),
    }
