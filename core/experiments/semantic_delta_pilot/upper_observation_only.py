"""Model-facing upper projection only; local state and control remain intact."""
import copy
import json

FIELDS = (
    "claim_id", "claim", "source_alignment", "source_ids", "latest", "round_id",
    "updates_claim_ids", "superseded_by_claim_ids", "observation_uncertainty",
    "claim_scope", "source_scopes", "audit_mode", "semantic_verification",
    "evidence_kind", "modality", "asr_source", "input_asr_guidance",
)


def enabled(runtime):
    return runtime is not None and getattr(runtime, "need_local_audit_enabled", False)


def records(states):
    result = []
    for state in states:
        if state.get("evidence_kind") == "need_answer":
            continue
        item = {k: copy.deepcopy(state[k]) for k in FIELDS if k in state}
        guidance = item.get("input_asr_guidance")
        if isinstance(guidance, dict):
            item["input_asr_guidance"] = {k: copy.deepcopy(guidance[k])
                for k in ("claim_ids", "source", "records", "text") if k in guidance}
        result.append(item)
    return result


def observation_payload(runtime):
    from .verifier_compact_view import evidence_view
    answers = list(getattr(runtime, "_need_answers", []))
    # Include original responses independently of which references Audit selected.
    # evidence_view owns reference collisions and existing state lifecycle.
    attempts = getattr(runtime, "_need_observations", [])
    if attempts:
        answers = answers + [{"attempts": attempts}]
    states, _, conflicts = evidence_view(runtime._evidence_state_snapshot(), answers)
    payload = {"retained_observations": records(states)}
    coverage = getattr(runtime, "_observation_coverage", None)
    if coverage is not None:
        payload["observation_coverage"] = copy.deepcopy(coverage)
    if conflicts:
        payload["material_reference_conflicts"] = {
            ref: [{"claim_id": ref, "claim": variant["text"]} for variant in variants]
            for ref, variants in conflicts.items()}
    return payload


def evidence_list(runtime, original):
    if not enabled(runtime):
        return original
    if not (getattr(runtime, "_need_observations", []) or getattr(runtime, "_need_answers", [])):
        return original  # Preserve the original Uniform/ASR initialization path.
    from avp.main import Evidence
    payload = observation_payload(runtime)
    if not payload["retained_observations"]:
        return []
    return [Evidence(detailed_response=json.dumps(payload, ensure_ascii=False))]


def decision_context(context):
    return {k: copy.deepcopy(v) for k, v in (context or {}).items()
            if k in ("hypothesis_states", "initial_asr_analysis", "asr_transcript")}


def candidate_request(request):
    from .avp_integration import _estimate_request_input_tokens
    out = copy.deepcopy(request)
    payload = json.loads(out["messages"][1]["content"])
    for key in ("need_answers", "active_search_need", "acquisition_limit"):
        payload.pop(key, None)
    payload["evidence_audit_states"] = records(payload["evidence_audit_states"])
    old = ("Local Audit owns need completion. Read need_answers and their source quotations, "
           "remaining gaps and failed searches before judging the global options. "
           "Do not return need_resolution. Judge options from the supplied tool evidence. "
           "If no useful different need remains, observation_need may be null. "
           "Do not repeat a limited need without a concrete new observable distinction. ")
    new = ("Local Audit owns need completion. Do not return need_resolution. "
           "Judge options from the supplied tool observations and their source information. "
           "If no useful different need remains, observation_need may be null. ")
    assert old in out["messages"][0]["content"], "unexpected candidate prompt contract"
    out["messages"][0]["content"] = out["messages"][0]["content"].replace(old, new, 1)
    # A tool capability diagnosis is local working state, not observation text.
    out["messages"][0]["content"] = out["messages"][0]["content"].replace(
        " The acquisition_limit reports an unavailable operation, not absence in the video. "
        "Reassess existing facts. If this need remains unresolved, you may supply one "
        "different neutral observation_need for an alternative observable lead.", "")
    out["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    out["estimated_input_tokens"] = _estimate_request_input_tokens(out["messages"])
    return out


def need_request(request, runtime):
    from .avp_integration import _estimate_request_input_tokens
    out = copy.deepcopy(request)
    payload = json.loads(out["messages"][1]["content"])
    payload.pop("need_answers", None)
    observations = observation_payload(runtime)
    payload["local_facts"] = observations.pop("retained_observations")
    payload.update(observations)
    out["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    out["estimated_input_tokens"] = _estimate_request_input_tokens(out["messages"])
    return out
