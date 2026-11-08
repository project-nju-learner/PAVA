from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any, Mapping, Sequence


OPERATOR_GATE_SCHEMA_VERSION = "operator_gate_v0"
OPERATOR_GATE_MODES = {"off", "shadow", "active"}
ALLOWED_OPERATOR_CHAINS = {
    ("SET",),
    ("SET", "COUNT"),
    ("TEMPORAL",),
    ("SET", "TEMPORAL"),
}
ALLOWED_DISPOSITIONS = {
    "SELECTED",
    "NOT_RELEVANT",
    "MEMBERSHIP_UNRESOLVED",
    "TEMPORAL_UNRESOLVED",
}


class OperatorGateValidationError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(str(item).strip() for item in value if str(item).strip())
    )


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _latest_states(
    evidence_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(state) for state in evidence_states if state.get("latest") is not False]


def _aligned_states(
    evidence_states: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(state.get("claim_id") or "").strip(): dict(state)
        for state in _latest_states(evidence_states)
        if str(state.get("claim_id") or "").strip()
        and str(state.get("source_alignment") or "").strip().upper() == "ALIGNED"
    }


def _estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized.encode("utf-8")) / 4))


def build_operator_gate_request(
    *,
    question: str,
    option_candidates: Sequence[Mapping[str, Any]],
    evidence_states: Sequence[Mapping[str, Any]],
    source_scopes: Mapping[str, Sequence[Mapping[str, Any]]],
    sampling_coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    latest = _latest_states(evidence_states)
    claims: list[dict[str, Any]] = []
    for state in latest:
        source_ids = _string_list(state.get("source_ids"))
        claims.append(
            {
                "claim_id": str(state.get("claim_id") or "").strip(),
                "claim": str(state.get("claim") or "").strip(),
                "source_alignment": str(
                    state.get("source_alignment") or ""
                ).strip().upper(),
                "source_ids": source_ids,
                "source_scopes": [
                    dict(scope)
                    for source_id in source_ids
                    for scope in source_scopes.get(source_id, ())
                ],
            }
        )
    payload: dict[str, Any] = {
        "question": question,
        "options": [dict(item) for item in option_candidates],
        "latest_audited_claims": claims,
    }
    if sampling_coverage is not None:
        payload["sampling_coverage"] = dict(sampling_coverage)
    system = (
        "You prepare neutral structured operands for a video-MCQ Operator Gate. "
        "You do not answer the question, choose an option, emit a final set/count/order, "
        "or change Evidence alignment. Use exactly one route: APPLY, BYPASS, UNHANDLED, "
        "or INPUT_CONFLICT. APPLY may use only these chains: [SET], [SET,COUNT], "
        "[TEMPORAL], or [SET,TEMPORAL]. Use only latest ALIGNED claims as operands. "
        "For APPLY or INPUT_CONFLICT, claim_dispositions must mention every latest "
        "ALIGNED claim exactly once with SELECTED, NOT_RELEVANT, "
        "MEMBERSHIP_UNRESOLVED, or TEMPORAL_UNRESOLVED and a concise reason. "
        "For BYPASS or UNHANDLED, return an empty claim_dispositions list. "
        "SET/COUNT operands go in members. One claim may yield multiple members. "
        "Each member has member, normalized_key, claim_ids, resolution "
        "(CONFIRMED or UNRESOLVED), and normalization_reason. Reuse normalized_key "
        "only for aliases you judge to be the same member; uncertain identity stays "
        "UNRESOLVED. TEMPORAL operands go in temporal_items with event and claim_ids; "
        "never invent timestamps or infer story chronology. Source times are owned by "
        "code. If an ALIGNED claim directly states a count without enumerating members, "
        "use BYPASS with route_reason DIRECT_ENTAILMENT_CANDIDATE. If another operation "
        "such as comparison, arithmetic, causality, or semantic story ordering is needed, "
        "use UNHANDLED. Return strict JSON with exactly route, route_reason, "
        "operator_chain, selection_rule, claim_dispositions, members, temporal_items, "
        "and conflicts. Keep every reason short."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return {
        "response_mode": "text",
        "max_tokens": 8192,
        "temperature": 0.0,
        "messages": messages,
        "estimated_input_tokens": _estimate_tokens(messages),
        "enable_thinking": False,
    }


def _parse_strict_json(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text.strip())
    except (TypeError, ValueError) as exc:
        raise OperatorGateValidationError("INVALID_SCHEMA") from exc
    if not isinstance(value, Mapping):
        raise OperatorGateValidationError("INVALID_SCHEMA")
    expected = {
        "route",
        "route_reason",
        "operator_chain",
        "selection_rule",
        "claim_dispositions",
        "members",
        "temporal_items",
        "conflicts",
    }
    if set(value) != expected:
        raise OperatorGateValidationError("INVALID_SCHEMA")
    return value


def _claim_manifest(
    *,
    evidence_states: Sequence[Mapping[str, Any]],
    dispositions: Mapping[str, Mapping[str, Any]],
    source_scopes: Mapping[str, Sequence[Mapping[str, Any]]],
    fallback_disposition: str | None = None,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for state in evidence_states:
        claim_id = str(state.get("claim_id") or "").strip()
        alignment = str(state.get("source_alignment") or "").strip().upper()
        source_ids = _string_list(state.get("source_ids"))
        if state.get("latest") is False:
            disposition = "HISTORICAL_SUPERSEDED"
            reason = "Historical audit state is preserved but is not a current operand."
        elif alignment != "ALIGNED":
            disposition = "INELIGIBLE_ALIGNMENT"
            reason = f"Latest claim has source_alignment={alignment or 'UNKNOWN'}."
        elif fallback_disposition is not None:
            disposition = fallback_disposition
            reason = "Claim is preserved outside an applied operator computation."
        else:
            raw = dispositions[claim_id]
            disposition = str(raw.get("disposition") or "").strip().upper()
            reason = _normalized_text(raw.get("reason"))
        manifest.append(
            {
                "claim_id": claim_id,
                "claim": _normalized_text(state.get("claim")),
                "source_alignment": alignment,
                "source_ids": source_ids,
                "source_scopes": [
                    dict(scope)
                    for source_id in source_ids
                    for scope in source_scopes.get(source_id, ())
                ],
                "latest": state.get("latest") is not False,
                "disposition": disposition,
                "reason": reason,
            }
        )
    return manifest


def _validate_dispositions(
    raw: Any,
    *,
    aligned_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw, list):
        raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
    dispositions: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
        claim_id = str(item.get("claim_id") or "").strip()
        disposition = str(item.get("disposition") or "").strip().upper()
        reason = _normalized_text(item.get("reason"))
        if (
            claim_id not in aligned_states
            or claim_id in dispositions
            or disposition not in ALLOWED_DISPOSITIONS
            or not reason
            or set(item) != {"claim_id", "disposition", "reason"}
        ):
            raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
        dispositions[claim_id] = item
    if set(dispositions) != set(aligned_states):
        raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
    return dispositions


def _validate_bound_claim_ids(
    claim_ids: list[str],
    *,
    aligned_states: Mapping[str, Mapping[str, Any]],
    dispositions: Mapping[str, Mapping[str, Any]],
) -> None:
    if not claim_ids or any(claim_id not in aligned_states for claim_id in claim_ids):
        raise OperatorGateValidationError("INVALID_BINDING")
    if any(
        str(dispositions[claim_id].get("disposition") or "").strip().upper()
        not in {"SELECTED", "MEMBERSHIP_UNRESOLVED", "TEMPORAL_UNRESOLVED"}
        for claim_id in claim_ids
    ):
        raise OperatorGateValidationError("INVALID_BINDING")


def _execute_set(
    raw_members: Any,
    *,
    aligned_states: Mapping[str, Mapping[str, Any]],
    dispositions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw_members, list):
        raise OperatorGateValidationError("INVALID_SCHEMA")
    confirmed: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    referenced_claims: set[str] = set()
    unresolved_claims: set[str] = set()
    for raw in raw_members:
        if not isinstance(raw, Mapping) or set(raw) != {
            "member",
            "normalized_key",
            "claim_ids",
            "resolution",
            "normalization_reason",
        }:
            raise OperatorGateValidationError("INVALID_SCHEMA")
        member = _normalized_text(raw.get("member"))
        normalized_key = _normalized_text(raw.get("normalized_key")).casefold()
        claim_ids = _string_list(raw.get("claim_ids"))
        resolution = str(raw.get("resolution") or "").strip().upper()
        normalization_reason = _normalized_text(raw.get("normalization_reason"))
        if (
            not member
            or not normalized_key
            or resolution not in {"CONFIRMED", "UNRESOLVED"}
            or not normalization_reason
        ):
            raise OperatorGateValidationError("INVALID_SCHEMA")
        _validate_bound_claim_ids(
            claim_ids,
            aligned_states=aligned_states,
            dispositions=dispositions,
        )
        normalized = {
            "member": member,
            "normalized_key": normalized_key,
            "claim_ids": claim_ids,
            "normalization_reason": normalization_reason,
        }
        referenced_claims.update(claim_ids)
        if resolution == "UNRESOLVED":
            unresolved_claims.update(claim_ids)
            unresolved.append(normalized)
            continue
        existing = confirmed.setdefault(
            normalized_key,
            {
                "normalized_key": normalized_key,
                "surface_forms": [],
                "claim_ids": [],
                "normalization_reasons": [],
            },
        )
        existing["surface_forms"] = list(
            dict.fromkeys([*existing["surface_forms"], member])
        )
        existing["claim_ids"] = list(
            dict.fromkeys([*existing["claim_ids"], *claim_ids])
        )
        existing["normalization_reasons"] = list(
            dict.fromkeys(
                [*existing["normalization_reasons"], normalization_reason]
            )
        )
    selected_claims = {
        claim_id
        for claim_id, item in dispositions.items()
        if str(item.get("disposition") or "").strip().upper() == "SELECTED"
    }
    declared_unresolved = {
        claim_id
        for claim_id, item in dispositions.items()
        if str(item.get("disposition") or "").strip().upper()
        == "MEMBERSHIP_UNRESOLVED"
    }
    forbidden = {
        str(item.get("disposition") or "").strip().upper()
        for item in dispositions.values()
    } - {"SELECTED", "NOT_RELEVANT", "MEMBERSHIP_UNRESOLVED"}
    if (
        forbidden
        or not selected_claims.issubset(referenced_claims)
        or declared_unresolved != unresolved_claims
    ):
        raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
    return {
        "result_kind": "SET",
        "confirmed_members": list(confirmed.values()),
        "confirmed_member_count": len(confirmed),
        "unresolved_members": unresolved,
        "operand_resolution_complete": not unresolved,
    }


def _item_scopes(
    claim_ids: Sequence[str],
    *,
    aligned_states: Mapping[str, Mapping[str, Any]],
    source_scopes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for claim_id in claim_ids:
        for source_id in _string_list(aligned_states[claim_id].get("source_ids")):
            for raw_scope in source_scopes.get(source_id, ()):
                try:
                    start = float(raw_scope["start_sec"])
                    end = float(raw_scope["end_sec"])
                except (KeyError, TypeError, ValueError):
                    continue
                if start >= 0 and end > start:
                    scopes.append(
                        {
                            "source_id": source_id,
                            "start_sec": start,
                            "end_sec": end,
                        }
                    )
    unique = {
        (item["source_id"], item["start_sec"], item["end_sec"]): item
        for item in scopes
    }
    return sorted(unique.values(), key=lambda item: (item["start_sec"], item["end_sec"]))


def _execute_temporal(
    raw_items: Any,
    *,
    aligned_states: Mapping[str, Mapping[str, Any]],
    dispositions: Mapping[str, Mapping[str, Any]],
    source_scopes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if not isinstance(raw_items, list):
        raise OperatorGateValidationError("INVALID_SCHEMA")
    items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    referenced_claims: set[str] = set()
    seen_events: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping) or set(raw) != {"event", "claim_ids"}:
            raise OperatorGateValidationError("INVALID_SCHEMA")
        event = _normalized_text(raw.get("event"))
        event_key = event.casefold()
        claim_ids = _string_list(raw.get("claim_ids"))
        if not event or event_key in seen_events:
            raise OperatorGateValidationError("INVALID_SCHEMA")
        seen_events.add(event_key)
        _validate_bound_claim_ids(
            claim_ids,
            aligned_states=aligned_states,
            dispositions=dispositions,
        )
        scopes = _item_scopes(
            claim_ids,
            aligned_states=aligned_states,
            source_scopes=source_scopes,
        )
        item = {"event": event, "claim_ids": claim_ids, "source_scopes": scopes}
        referenced_claims.update(claim_ids)
        binding_unresolved = any(
            str(dispositions[claim_id].get("disposition") or "").strip().upper()
            == "TEMPORAL_UNRESOLVED"
            for claim_id in claim_ids
        )
        if binding_unresolved:
            unresolved.append({**item, "reason": "EVENT_BINDING_UNRESOLVED"})
        elif not scopes:
            raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
        else:
            item["observed_start_sec"] = min(scope["start_sec"] for scope in scopes)
            item["observed_end_sec"] = max(scope["end_sec"] for scope in scopes)
            items.append(item)
    selected_claims = {
        claim_id
        for claim_id, item in dispositions.items()
        if str(item.get("disposition") or "").strip().upper() == "SELECTED"
    }
    declared_unresolved = {
        claim_id
        for claim_id, item in dispositions.items()
        if str(item.get("disposition") or "").strip().upper()
        == "TEMPORAL_UNRESOLVED"
    }
    forbidden = {
        str(item.get("disposition") or "").strip().upper()
        for item in dispositions.values()
    } - {"SELECTED", "NOT_RELEVANT", "TEMPORAL_UNRESOLVED"}
    unresolved_references = {
        claim_id
        for item in unresolved
        for claim_id in item["claim_ids"]
        if str(dispositions[claim_id].get("disposition") or "").strip().upper()
        == "TEMPORAL_UNRESOLVED"
    }
    if (
        forbidden
        or not selected_claims.issubset(referenced_claims)
        or declared_unresolved != unresolved_references
    ):
        raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
    relations: list[dict[str, Any]] = []
    has_overlap = False
    for left, right in combinations(items, 2):
        if left["observed_end_sec"] <= right["observed_start_sec"]:
            relation = "BEFORE"
        elif right["observed_end_sec"] <= left["observed_start_sec"]:
            relation = "AFTER"
        else:
            relation = "OVERLAP"
            has_overlap = True
        relations.append(
            {
                "left_event": left["event"],
                "relation": relation,
                "right_event": right["event"],
            }
        )
    return {
        "result_kind": "TEMPORAL",
        "observed_items": items,
        "relations": relations,
        "unresolved_items": unresolved,
        "temporal_basis": "SOURCE_RECEIPT_VIDEO_ORDER",
        "operand_resolution_complete": not unresolved and not has_overlap,
    }


def execute_operator_gate_response(
    *,
    text: str,
    evidence_revision: int,
    evidence_states: Sequence[Mapping[str, Any]],
    source_scopes: Mapping[str, Sequence[Mapping[str, Any]]],
    sampling_coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = _parse_strict_json(text)
    route = str(parsed.get("route") or "").strip().upper()
    route_reason = _normalized_text(parsed.get("route_reason"))
    raw_operator_chain = parsed.get("operator_chain")
    if not isinstance(raw_operator_chain, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in raw_operator_chain
    ):
        raise OperatorGateValidationError("INVALID_OPERATOR_CHAIN")
    operator_chain = tuple(item.strip().upper() for item in raw_operator_chain)
    selection_rule = _normalized_text(parsed.get("selection_rule"))
    if route not in {"APPLY", "BYPASS", "UNHANDLED", "INPUT_CONFLICT"} or not route_reason:
        raise OperatorGateValidationError("INVALID_SCHEMA")
    aligned = _aligned_states(evidence_states)

    if route in {"BYPASS", "UNHANDLED"}:
        if operator_chain or parsed.get("claim_dispositions") != []:
            raise OperatorGateValidationError("INVALID_OPERATOR_CHAIN")
        if (
            selection_rule
            or parsed.get("members") != []
            or parsed.get("temporal_items") != []
            or parsed.get("conflicts") != []
        ):
            raise OperatorGateValidationError("INVALID_SCHEMA")
        status = route
        manifest = _claim_manifest(
            evidence_states=evidence_states,
            dispositions={},
            source_scopes=source_scopes,
            fallback_disposition="PRESERVED_NOT_PROCESSED",
        )
        result = None
        closure_complete = True
    else:
        if route == "APPLY" and operator_chain not in ALLOWED_OPERATOR_CHAINS:
            raise OperatorGateValidationError("INVALID_OPERATOR_CHAIN")
        if route == "INPUT_CONFLICT" and operator_chain:
            raise OperatorGateValidationError("INVALID_OPERATOR_CHAIN")
        if not selection_rule:
            raise OperatorGateValidationError("INVALID_SCHEMA")
        dispositions = _validate_dispositions(
            parsed.get("claim_dispositions"),
            aligned_states=aligned,
        )
        manifest = _claim_manifest(
            evidence_states=evidence_states,
            dispositions=dispositions,
            source_scopes=source_scopes,
        )
        closure_complete = True
        if route == "INPUT_CONFLICT":
            if parsed.get("members") != [] or parsed.get("temporal_items") != []:
                raise OperatorGateValidationError("INVALID_SCHEMA")
            conflicts = parsed.get("conflicts")
            if not isinstance(conflicts, list) or not conflicts:
                raise OperatorGateValidationError("INVALID_SCHEMA")
            conflict_claim_ids: set[str] = set()
            for conflict in conflicts:
                if not isinstance(conflict, Mapping) or set(conflict) != {
                    "claim_ids",
                    "reason",
                }:
                    raise OperatorGateValidationError("INVALID_SCHEMA")
                claim_ids = _string_list(conflict.get("claim_ids"))
                if not _normalized_text(conflict.get("reason")):
                    raise OperatorGateValidationError("INVALID_SCHEMA")
                _validate_bound_claim_ids(
                    claim_ids,
                    aligned_states=aligned,
                    dispositions=dispositions,
                )
                conflict_claim_ids.update(claim_ids)
            expected_conflict_claim_ids = {
                claim_id
                for claim_id, item in dispositions.items()
                if str(item.get("disposition") or "").strip().upper()
                != "NOT_RELEVANT"
            }
            if conflict_claim_ids != expected_conflict_claim_ids:
                raise OperatorGateValidationError("INVALID_SELECTION_MANIFEST")
            status = "INVALID"
            route_reason = "INPUT_CONFLICT"
            result = None
        elif operator_chain[-1] in {"SET", "COUNT"}:
            if parsed.get("temporal_items") != [] or parsed.get("conflicts") != []:
                raise OperatorGateValidationError("INVALID_SCHEMA")
            result = _execute_set(
                parsed.get("members"),
                aligned_states=aligned,
                dispositions=dispositions,
            )
            result["result_kind"] = operator_chain[-1]
            status = "APPLIED"
        else:
            if parsed.get("members") != [] or parsed.get("conflicts") != []:
                raise OperatorGateValidationError("INVALID_SCHEMA")
            result = _execute_temporal(
                parsed.get("temporal_items"),
                aligned_states=aligned,
                dispositions=dispositions,
                source_scopes=source_scopes,
            )
            status = "APPLIED"

    return {
        "schema_version": OPERATOR_GATE_SCHEMA_VERSION,
        "status": status,
        "reason_code": route_reason,
        "evidence_revision": int(evidence_revision),
        "operator_chain": list(operator_chain),
        "selection_rule": selection_rule,
        "claim_manifest": manifest,
        "projection_closure": {
            "complete": closure_complete,
            "latest_claim_count": len(_latest_states(evidence_states)),
            "eligible_aligned_claim_count": len(aligned),
            "accounted_aligned_claim_count": len(aligned) if closure_complete else 0,
        },
        "sampling_coverage": dict(sampling_coverage) if sampling_coverage else None,
        "calculation": result,
    }


def failed_operator_gate_packet(
    *,
    evidence_revision: int,
    evidence_states: Sequence[Mapping[str, Any]],
    source_scopes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    sampling_coverage: Mapping[str, Any] | None,
    status: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "schema_version": OPERATOR_GATE_SCHEMA_VERSION,
        "status": status,
        "reason_code": reason_code,
        "evidence_revision": int(evidence_revision),
        "operator_chain": [],
        "selection_rule": "",
        "claim_manifest": _claim_manifest(
            evidence_states=evidence_states,
            dispositions={},
            source_scopes=source_scopes or {},
            fallback_disposition="PRESERVED_AFTER_GATE_ERROR",
        ),
        "projection_closure": {
            "complete": False,
            "latest_claim_count": len(_latest_states(evidence_states)),
            "eligible_aligned_claim_count": len(_aligned_states(evidence_states)),
            "accounted_aligned_claim_count": 0,
        },
        "sampling_coverage": dict(sampling_coverage) if sampling_coverage else None,
        "calculation": None,
    }


def is_input_too_large_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    markers = (
        "context length",
        "maximum context",
        "max context",
        "too many tokens",
        "input too long",
        "request too large",
    )
    return any(marker in text for marker in markers)
