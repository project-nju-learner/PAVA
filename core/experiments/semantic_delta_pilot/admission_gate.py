from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


SourceAlignment = Literal["ALIGNED", "UNCERTAIN", "CONFLICTING"]
Permission = Literal["ANSWER_ELIGIBLE", "SEARCH_ONLY", "BLOCKED_BINDING"]

EVIDENCE_AUDIT_SCHEMA_VERSION = "evidence_audit_state_v2"
LEGACY_ALIGNMENT = {
    "SUPPORTED": "ALIGNED",
    "OPEN": "UNCERTAIN",
    "CONTRADICTED": "CONFLICTING",
}
ALIGNMENT_TO_LEGACY = {value: key for key, value in LEGACY_ALIGNMENT.items()}
SOURCE_ALIGNMENTS = frozenset(ALIGNMENT_TO_LEGACY)


def normalize_source_alignment(value: Any) -> SourceAlignment:
    """Normalize v1 Evidence verdicts and v2 alignments to the v2 vocabulary."""

    raw = str(value or "").strip().upper()
    normalized = LEGACY_ALIGNMENT.get(raw, raw)
    if normalized not in SOURCE_ALIGNMENTS:
        raise ValueError(f"unknown_source_alignment:{raw}")
    return cast(SourceAlignment, normalized)


def evidence_alignment_from_record(
    record: Mapping[str, Any],
    *,
    default: SourceAlignment = "UNCERTAIN",
) -> SourceAlignment:
    """Read a known Evidence record without interpreting Hypothesis states."""

    raw_values = [
        record[key]
        for key in ("source_alignment", "verdict", "status")
        if record.get(key) is not None
    ]
    if not raw_values:
        return default
    alignments = {normalize_source_alignment(value) for value in raw_values}
    if len(alignments) != 1:
        raise ValueError("conflicting_source_alignment_fields")
    return alignments.pop()


def _migrate_legacy_alignment_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    legacy = payload.pop("verdict", None)
    current = payload.get("source_alignment")
    if current is None and legacy is not None:
        payload["source_alignment"] = normalize_source_alignment(legacy)
    elif current is not None:
        payload["source_alignment"] = normalize_source_alignment(current)
        if (
            legacy is not None
            and normalize_source_alignment(legacy) != payload["source_alignment"]
        ):
            raise ValueError("conflicting_source_alignment_fields")
    return payload


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceRecord(StrictFrozenModel):
    source_id: str = Field(min_length=1)
    provenance_root_id: str = Field(min_length=1)
    source_kind: Literal["caption", "observer_claim", "transport_receipt", "asr_segment"]
    exact_caption: str = Field(min_length=1)
    locator: str = Field(min_length=1)


class DeltaPacket(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    delta_id: str = Field(min_length=1)
    delta_claim: str = Field(min_length=1)
    source_records: tuple[SourceRecord, ...] = Field(min_length=1)


class SourceQuote(StrictFrozenModel):
    source_id: str = Field(min_length=1)
    exact_quote: str = Field(min_length=1)


class AdmissionAssessment(StrictFrozenModel):
    """Model output with no authority to choose verdict or permission."""

    support_quotes: tuple[SourceQuote, ...]
    conflict_quotes: tuple[SourceQuote, ...]
    missing_premises: tuple[str, ...]


class AdmissionAlignment(StrictFrozenModel):
    """Deterministic reduction of a quote-grounded model assessment."""

    source_alignment: SourceAlignment
    permission: Permission
    source_span_ids: tuple[str, ...]
    support_quotes: tuple[SourceQuote, ...]
    conflict_quotes: tuple[SourceQuote, ...]
    missing_premises: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def accept_v1_verdict(cls, value: Any) -> Any:
        return _migrate_legacy_alignment_payload(value)

    @property
    def verdict(self) -> str:
        """Deprecated v1 read view for callers that have not migrated yet."""

        return ALIGNMENT_TO_LEGACY[self.source_alignment]


# Import compatibility for older experiments. New code uses AdmissionAlignment.
AdmissionVerdict = AdmissionAlignment


class AuditRecord(StrictFrozenModel):
    schema_version: str = EVIDENCE_AUDIT_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    delta_id: str = Field(min_length=1)
    source_alignment: SourceAlignment
    permission: Permission
    unique_provenance_roots: tuple[str, ...]
    admission_request_id: str = Field(min_length=1)
    source_span_ids: tuple[str, ...]
    support_quotes: tuple[SourceQuote, ...]
    conflict_quotes: tuple[SourceQuote, ...]
    missing_premises: tuple[str, ...]
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def accept_v1_verdict(cls, value: Any) -> Any:
        return _migrate_legacy_alignment_payload(value)

    @property
    def verdict(self) -> str:
        """Deprecated v1 read view for callers that have not migrated yet."""

        return ALIGNMENT_TO_LEGACY[self.source_alignment]


_PERMISSIONS: dict[str, Permission] = {
    "ALIGNED": "ANSWER_ELIGIBLE",
    "UNCERTAIN": "SEARCH_ONLY",
    "CONFLICTING": "BLOCKED_BINDING",
}


def _deduplicate_sources(packet: DeltaPacket) -> tuple[SourceRecord, ...]:
    by_root: dict[str, list[SourceRecord]] = {}
    for record in packet.source_records:
        by_root.setdefault(record.provenance_root_id, []).append(record)

    unique: list[SourceRecord] = []
    for root_id in sorted(by_root):
        records = by_root[root_id]
        contents = {
            (record.source_kind, record.exact_caption, record.locator)
            for record in records
        }
        if len(contents) != 1:
            raise ValueError(f"conflicting_source_records:{root_id}")
        unique.append(
            min(
                records,
                key=lambda record: (
                    record.source_id,
                    record.exact_caption,
                    record.locator,
                ),
            )
        )
    return tuple(unique)


def build_admission_request(packet: DeltaPacket) -> dict[str, Any]:
    sources = _deduplicate_sources(packet)
    visible_payload = {
        "delta_claim": packet.delta_claim,
        "source_records": [
            {
                "source_id": record.source_id,
                "provenance_root_id": record.provenance_root_id,
                "source_kind": record.source_kind,
                "exact_caption": record.exact_caption,
                "locator": record.locator,
            }
            for record in sources
        ],
    }
    response_schema = AdmissionAssessment.model_json_schema()
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Assess whether the candidate claim follows from the exact cited sources "
                    "without adding unstated identity, event, temporal, causal, purpose, "
                    "counting, or coverage assumptions. Do not output a verdict or permission. "
                    "Return exact support/conflict substrings with their source_id values and "
                    "list every missing premise. An observer_claim is a model-reported visual "
                    "statement, not independent proof that the pixels were interpreted faithfully; "
                    "a transport_receipt has no semantic authority. "
                    "Return only one JSON object with keys support_quotes, "
                    "conflict_quotes, and missing_premises. Each quote item uses "
                    "source_id and exact_quote; each of the three top-level values "
                    "is an array."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    visible_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "admission_verdict",
                "strict": True,
                "schema": response_schema,
            },
        },
    }


def parse_admission_response(
    response: AdmissionAssessment | Mapping[str, Any] | str | bytes,
) -> AdmissionAssessment:
    if isinstance(response, AdmissionAssessment):
        return response
    if isinstance(response, (str, bytes)):
        return AdmissionAssessment.model_validate_json(response)
    return AdmissionAssessment.model_validate(response)


def reduce_alignment(source_alignment: str) -> Permission:
    alignment = normalize_source_alignment(source_alignment)
    try:
        return _PERMISSIONS[alignment]
    except KeyError as exc:
        raise ValueError(f"unknown_source_alignment:{source_alignment}") from exc


def reduce_verdict(verdict: str) -> Permission:
    """Deprecated v1 compatibility wrapper."""

    return reduce_alignment(verdict)


def _validated_quotes(
    packet: DeltaPacket,
    quotes: tuple[SourceQuote, ...],
) -> tuple[SourceQuote, ...]:
    sources = {record.source_id: record for record in _deduplicate_sources(packet)}
    validated: list[SourceQuote] = []
    seen: set[tuple[str, str]] = set()
    for quote in quotes:
        source = sources.get(quote.source_id)
        if source is None:
            raise ValueError(f"unknown_source_span_id:{quote.source_id}")
        if source.source_kind == "transport_receipt":
            raise ValueError("transport_receipt_has_no_semantic_authority")
        exact_quote = quote.exact_quote.strip()
        if not exact_quote or exact_quote not in source.exact_caption:
            raise ValueError(f"source_quote_not_exact:{quote.source_id}")
        key = (quote.source_id, exact_quote)
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            SourceQuote(source_id=quote.source_id, exact_quote=exact_quote)
        )
    return tuple(validated)


def derive_admission_alignment(
    packet: DeltaPacket,
    assessment: AdmissionAssessment,
) -> AdmissionAlignment:
    """Derive the authoritative source alignment from validated evidence."""

    support_quotes = _validated_quotes(packet, assessment.support_quotes)
    conflict_quotes = _validated_quotes(packet, assessment.conflict_quotes)
    missing_premises = tuple(
        dict.fromkeys(item.strip() for item in assessment.missing_premises)
    )
    if any(not item for item in missing_premises):
        raise ValueError("missing_premise_must_be_nonblank")
    support_keys = {(item.source_id, item.exact_quote) for item in support_quotes}
    conflict_keys = {(item.source_id, item.exact_quote) for item in conflict_quotes}
    if support_keys & conflict_keys:
        raise ValueError("same_quote_cannot_support_and_conflict")
    source_span_ids = tuple(
        dict.fromkeys(
            quote.source_id for quote in support_quotes + conflict_quotes
        )
    )
    if conflict_quotes:
        source_alignment: SourceAlignment = "CONFLICTING"
    elif support_quotes and not missing_premises:
        source_alignment = "ALIGNED"
    else:
        source_alignment = "UNCERTAIN"
    return AdmissionAlignment(
        source_alignment=source_alignment,
        permission=reduce_alignment(source_alignment),
        source_span_ids=source_span_ids,
        support_quotes=support_quotes,
        conflict_quotes=conflict_quotes,
        missing_premises=missing_premises,
    )


def derive_admission_verdict(
    packet: DeltaPacket,
    assessment: AdmissionAssessment,
) -> AdmissionAlignment:
    """Deprecated v1 compatibility wrapper."""

    return derive_admission_alignment(packet, assessment)


def freeze_audit_record(
    packet: DeltaPacket,
    request: Mapping[str, Any],
    alignment: AdmissionAlignment,
    *,
    admission_request_id: str,
    created_at: datetime | None = None,
) -> AuditRecord:
    expected_request = build_admission_request(packet)
    if request != expected_request:
        raise ValueError("admission_request_does_not_match_packet")

    unique_sources = _deduplicate_sources(packet)
    if not str(admission_request_id).strip():
        raise ValueError("admission_request_id_required")
    if alignment.permission != reduce_alignment(alignment.source_alignment):
        raise ValueError("alignment_permission_mismatch")
    expected_alignment = derive_admission_alignment(
        packet,
        AdmissionAssessment(
            support_quotes=alignment.support_quotes,
            conflict_quotes=alignment.conflict_quotes,
            missing_premises=alignment.missing_premises,
        ),
    )
    if alignment != expected_alignment:
        raise ValueError("admission_alignment_not_deterministically_derived")

    frozen_at = created_at or datetime.now(timezone.utc)
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError("created_at_must_be_timezone_aware")

    return AuditRecord(
        case_id=packet.case_id,
        delta_id=packet.delta_id,
        source_alignment=alignment.source_alignment,
        permission=reduce_alignment(alignment.source_alignment),
        unique_provenance_roots=tuple(
            record.provenance_root_id for record in unique_sources
        ),
        admission_request_id=str(admission_request_id).strip(),
        source_span_ids=alignment.source_span_ids,
        support_quotes=alignment.support_quotes,
        conflict_quotes=alignment.conflict_quotes,
        missing_premises=alignment.missing_premises,
        created_at=frozen_at,
    )


class AppendOnlyAuditLog:
    def __init__(self) -> None:
        self._records: tuple[AuditRecord, ...] = ()

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return self._records

    def append(self, record: AuditRecord) -> None:
        if not isinstance(record, AuditRecord):
            raise TypeError("audit_record_required")
        self._records = self._records + (record,)
