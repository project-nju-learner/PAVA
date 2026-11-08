"""Receipt-only observation audit. No video decoding or model calls.

This boundary checks provenance and submitted scope, not visual truth.
Semantic reinspection must be an explicit Controller -> Observe request.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from google.genai import types

from avp.main import parse_json_response
from experiments.semantic_delta_pilot.admission_gate import (
    EVIDENCE_AUDIT_SCHEMA_VERSION,
    evidence_alignment_from_record,
)
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    MAX_VIDEO_FRAMES,
    SOURCE_SCOPE_TIME_TOLERANCE_SEC,
)

MIN_CLAIM_SOURCE_COVERAGE = 0.5


class EvidenceAuditRuntime:
    def __init__(self, *, audit_client: Any, run_dir: Path) -> None:
        # Compatibility argument only: Audit deliberately retains no media client.
        self.audit_client = None
        self.run_dir = Path(run_dir)
        self.audit_dir = self.run_dir / "coarse_mcq_audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.request_count = 0

    @staticmethod
    def _claims(evidence: Any) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for index, item in enumerate(getattr(evidence, "key_evidence", []) or [], 1):
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("description") or "").strip()
            if text:
                claim: Dict[str, Any] = {
                    "claim_id": f"observe_span_{index}",
                    "text": text,
                }
                if "uncertainty" in item:
                    claim["uncertainty"] = item["uncertainty"]
                try:
                    start = float(item["timestamp_start"])
                    end = float(item["timestamp_end"])
                except (KeyError, TypeError, ValueError):
                    claim["timestamp_error"] = "key_evidence_timestamp_required"
                else:
                    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                        claim["timestamp_error"] = "invalid_key_evidence_timestamp"
                    else:
                        claim["timestamp_start"] = start
                        claim["timestamp_end"] = end
                if (
                    str(item.get("scope_precision") or "").strip()
                    == "submitted_negative_observation"
                ):
                    claim["scope_precision"] = "submitted_negative_observation"
                claims.append(claim)
        return claims

    @staticmethod
    def _claim_source_coverage(
        *,
        claim_start: float,
        claim_end: float,
        sources: Sequence[Mapping[str, Any]],
    ) -> float:
        clipped = sorted(
            (
                max(claim_start, float(source["start_sec"])),
                min(claim_end, float(source["end_sec"])),
            )
            for source in sources
            if (
                float(source["start_sec"]) < claim_end
                and claim_start < float(source["end_sec"])
            )
        )
        covered = 0.0
        cursor = claim_start
        for start, end in clipped:
            start = max(start, cursor)
            if end > start:
                covered += end - start
                cursor = end
        return covered / (claim_end - claim_start)

    @staticmethod
    def _partition_claim_scopes(
        claims: Sequence[Mapping[str, Any]],
        sources: Sequence[Mapping[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        valid: List[Dict[str, Any]] = []
        open_results: List[Dict[str, Any]] = []
        for claim in claims:
            claim_id = str(claim["claim_id"])
            timestamp_error = str(claim.get("timestamp_error") or "").strip()
            if timestamp_error:
                open_results.append(
                    {
                        "claim_id": claim_id,
                        "source_alignment": "UNCERTAIN",
                        "source_ids": [],
                        "brief_reason": (
                            f"planner_only_unparsed_fact:{timestamp_error}"
                        ),
                    }
                )
                continue
            start = float(claim["timestamp_start"])
            end = float(claim["timestamp_end"])
            overlapping_sources = [
                source
                for source in sources
                if (
                    float(source["start_sec"]) < end
                    and start < float(source["end_sec"])
                )
            ]
            source_coverage = EvidenceAuditRuntime._claim_source_coverage(
                claim_start=start,
                claim_end=end,
                sources=overlapping_sources,
            )
            if source_coverage + 1e-12 < MIN_CLAIM_SOURCE_COVERAGE:
                open_results.append(
                    {
                        "claim_id": claim_id,
                        "source_alignment": "UNCERTAIN",
                        "source_ids": [],
                        "brief_reason": (
                            "planner_only_unparsed_fact:"
                            "key_evidence_outside_submitted_source_scope"
                        ),
                    }
                )
                continue
            if all(
                str(source.get("observation_profile") or "").strip().upper()
                == "COARSE"
                for source in overlapping_sources
            ):
                coarse_claim = dict(claim)
                coarse_claim["observation_profile"] = "COARSE"
                valid.append(coarse_claim)
                continue
            valid.append(dict(claim))
        return valid, open_results

    @staticmethod
    def _apply_coarse_policy(
        results: Sequence[Mapping[str, Any]],
        claims: Sequence[Mapping[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Discard junk and keep COARSE information as retrieval-only UNCERTAIN."""

        claims_by_id = {
            str(claim.get("claim_id") or ""): claim for claim in claims
        }
        kept: List[Dict[str, Any]] = []
        discarded: List[Dict[str, Any]] = []
        for result in results:
            item = dict(result)
            claim_id = str(item.get("claim_id") or "")
            claim = claims_by_id.get(claim_id, {})
            if item.get("observable_information") is False:
                discarded.append(
                    {
                        "claim_id": claim_id,
                        "audit_alignment": evidence_alignment_from_record(item),
                        "reason": "audit_marked_non_observable_information",
                    }
                )
                continue
            if str(claim.get("observation_profile") or "").upper() != "COARSE":
                kept.append(item)
                continue
            audit_alignment = evidence_alignment_from_record(item)
            tendency = (
                "SUPPORT"
                if audit_alignment == "ALIGNED"
                else (
                    "CONSTRAIN"
                    if audit_alignment == "CONFLICTING"
                    else "UNRESOLVED"
                )
            )
            start = float(claim["timestamp_start"])
            end = float(claim["timestamp_end"])
            audit_reason = str(item.get("brief_reason") or "").strip()
            item.update(
                {
                    "source_alignment": "UNCERTAIN",
                    "source_ids": [],
                    "coarse_tendency": tendency,
                    "audit_alignment": audit_alignment,
                    "brief_reason": (
                        f"COARSE_{tendency}_TENDENCY at [{start:.3f},{end:.3f}]; "
                        "keep UNCERTAIN and recheck this range with a denser local "
                        "observation."
                        + (f" Audit: {audit_reason}" if audit_reason else "")
                    ),
                }
            )
            kept.append(item)
        return kept, discarded

    @staticmethod
    def _source_records(evidence: Any, video_path: str) -> List[Dict[str, Any]]:
        receipt = (getattr(evidence, "model_call", {}) or {}).get(
            "observation_receipt"
        )
        if not isinstance(receipt, Mapping) or receipt.get("status") != "COMPLETED":
            raise ValueError("completed_observation_receipt_required_for_mcq_audit")
        units = receipt.get("source_units")
        if not isinstance(units, list) or not units:
            raise ValueError("observation_source_units_required_for_mcq_audit")
        expected_video = str(Path(video_path).resolve(strict=True))
        records: List[Dict[str, Any]] = []
        for index, unit in enumerate(units, 1):
            if not isinstance(unit, Mapping):
                raise ValueError("invalid_observation_source_unit")
            source_video = str(
                Path(str(unit.get("source_video_path") or "")).resolve(strict=True)
            )
            submitted_media = Path(
                str(unit.get("submitted_media_path") or "")
            ).resolve(strict=True)
            if source_video != expected_video or not submitted_media.is_file():
                raise ValueError("observation_source_identity_mismatch")
            start = float(unit["source_start_sec"])
            end = float(unit["source_end_sec"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("invalid_observation_source_scope")
            submitted_start = float(unit["submitted_start_sec"])
            submitted_end = float(unit["submitted_end_sec"])
            transport = unit.get("transport_evidence")
            if (
                not math.isfinite(submitted_start) or not math.isfinite(submitted_end)
                or submitted_media != Path(source_video)
                or abs(submitted_start - start)
                > SOURCE_SCOPE_TIME_TOLERANCE_SEC
                or abs(submitted_end - end)
                > SOURCE_SCOPE_TIME_TOLERANCE_SEC
                or not isinstance(transport, Mapping)
                or transport.get("kind")
                != "direct_bounded_ffmpeg_base64_jpeg_frames"
            ):
                raise ValueError("direct_original_video_scope_required_for_mcq_audit")
            submitted_fps = float(unit.get("submitted_fps") or unit.get("fps") or 0)
            if not math.isfinite(submitted_fps) or submitted_fps <= 0:
                raise ValueError("invalid_observation_source_fps")
            records.append(
                {
                    "source_id": f"obs_r{int(getattr(evidence, 'round_id', 0))}_u{index}",
                    "source_video_path": source_video,
                    "submitted_media_path": str(submitted_media),
                    "start_sec": start,
                    "end_sec": end,
                    "submitted_fps": submitted_fps,
                    "observation_profile": (
                        "COARSE"
                        if submitted_fps <= 0.1 + 1e-9
                        else None
                    ),
                }
            )
        return records

    @staticmethod
    def _normalize_observation_results(
        parsed: Mapping[str, Any],
        *,
        claims: Sequence[Mapping[str, Any]],
        sources: Sequence[Mapping[str, Any]],
        prior_evidence_states: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        allowed_sources = {str(item["source_id"]) for item in sources}
        expected_claims = {str(item["claim_id"]) for item in claims}
        allowed_prior_claims = {
            str(item.get("claim_id") or "").strip()
            for item in prior_evidence_states
            if str(item.get("claim_id") or "").strip()
            and item.get("latest") is not False
        }
        raw_items = parsed.get("observation_results")
        if not isinstance(raw_items, list):
            raise ValueError("claim_id_audit_results_required")
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                continue
            claim_id = str(raw.get("claim_id") or "").strip()
            try:
                source_alignment = evidence_alignment_from_record(raw)
            except ValueError:
                continue
            source_ids = list(
                dict.fromkeys(str(value).strip() for value in (raw.get("source_ids") or []))
            )
            if claim_id not in expected_claims or claim_id in seen:
                continue
            if any(source_id not in allowed_sources for source_id in source_ids):
                continue
            if source_alignment in {"ALIGNED", "CONFLICTING"} and not source_ids:
                continue
            raw_updates = raw.get("updates_claim_ids") or []
            if isinstance(raw_updates, str):
                raw_updates = [raw_updates]
            updates_claim_ids = (
                list(
                    dict.fromkeys(
                        str(value).strip()
                        for value in raw_updates
                        if str(value).strip() in allowed_prior_claims
                    )
                )
                if isinstance(raw_updates, (list, tuple))
                else []
            )
            normalized.append(
                {
                    "claim_id": claim_id,
                    "source_alignment": source_alignment,
                    "observable_information": (
                        raw.get("observable_information")
                        if isinstance(raw.get("observable_information"), bool)
                        else True
                    ),
                    "source_ids": source_ids,
                    "updates_claim_ids": updates_claim_ids,
                    "brief_reason": str(raw.get("brief_reason") or "").strip(),
                }
            )
            seen.add(claim_id)
        # Missing or malformed rows stay UNCERTAIN instead of killing the observation.
        for claim_id in sorted(expected_claims - seen):
            normalized.append(
                {
                    "claim_id": claim_id,
                    "source_alignment": "UNCERTAIN",
                    "observable_information": True,
                    "source_ids": [],
                    "updates_claim_ids": [],
                    "brief_reason": "Evidence Audit returned no usable source alignment for this claim.",
                }
            )
        return normalized

    def audit_observation_claims(
        self,
        *,
        query: str,
        evidence: Any,
        video_path: str,
        prior_evidence_states: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Check receipt/scope only; never infer facts or supersede prior claims."""

        sources = self._source_records(evidence, video_path)
        claims = self._claims(evidence)
        if not claims:
            fallback_text = str(
                getattr(evidence, "detailed_response", "") or ""
            ).strip()
            if fallback_text:
                claims = [
                    {
                        "claim_id": "unparsed_observation",
                        "text": fallback_text,
                        "timestamp_error": "no_parseable_key_evidence",
                    }
                ]
        valid_claims, open_results = self._partition_claim_scopes(claims, sources)
        observation_results: List[Dict[str, Any]] = []
        discarded_observation_results: List[Dict[str, Any]] = []
        for claim in valid_claims:
            start, end = float(claim["timestamp_start"]), float(claim["timestamp_end"])
            bound_sources = [source["source_id"] for source in sources
                if float(source["start_sec"]) < end and start < float(source["end_sec"])]
            observation_results.append({
                "claim_id": claim["claim_id"],
                "source_alignment": "ALIGNED",
                "source_ids": bound_sources,
                "updates_claim_ids": [],
                "binding_check": "PASSED",
                "observation_uncertainty": claim.get("uncertainty"),
                "semantic_verification": "NOT_PERFORMED",
                "brief_reason": "Qualified local Observe fact with checked receipt and temporal binding; "
                    "does not establish global coverage or answer sufficiency.",
            })
        observation_results.extend(open_results)
        result_order = {
            str(claim["claim_id"]): index for index, claim in enumerate(claims)
        }
        observation_results.sort(
            key=lambda item: result_order.get(str(item.get("claim_id") or ""), len(claims))
        )
        event = {
            "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
            "audit_mode": "receipt_only",
            "semantic_verification": "NOT_PERFORMED",
            "round_id": int(getattr(evidence, "round_id", 0)),
            "sources": sources,
            "claims": claims,
            "observation_results": observation_results,
            "discarded_observation_results": discarded_observation_results,
        }
        self.events.append(copy.deepcopy(event))
        (self.audit_dir / f"evidence_round_{event['round_id']}.json").write_text(
            json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return event

    def admit_observation(self, **kwargs: Any) -> Dict[str, Any]:
        evidence = kwargs["evidence"]
        if kwargs.get("option_candidates") is not None:
            raise ValueError("evidence_audit_must_not_receive_option_candidates")
        return {
            "status": "SOURCE_ADMISSION_COMPLETE",
            "round_id": int(getattr(evidence, "round_id", 0)),
        }

    def project_evidence_list(self, evidence_list: List[Any]) -> List[Any]:
        return evidence_list

    def project_blackboard(self, blackboard: Any) -> Any:
        return blackboard

    def observation_blackboard(self, blackboard: Any) -> Any:
        return blackboard

    def has_blocking_delta(self) -> bool:
        return False

    def can_dispatch_repair(self) -> bool:
        return False

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": "visual_evidence_audit",
            "request_count": self.request_count,
            "rounds": copy.deepcopy(self.events),
        }

    def close(self) -> None:
        self.audit_client.cleanup_uploaded_files()


# Compatibility for older experiment imports.  The class no longer owns any
# option or answer-hypothesis behavior.
CoarseMCQAuditRuntime = EvidenceAuditRuntime
