"""MCQ option hypotheses for the optional AVP Hypothesis Verifier path.

Options are answer candidates, never observations.  They therefore enter the
decision path as UNCHECKED hypotheses with no source authority.  Only an audit result
that cites observed sources may move a candidate to SUPPORTED or
CONTRADICTED.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


VALID_CANDIDATE_VERDICTS = {"UNCHECKED", "SUPPORTED", "OPEN", "CONTRADICTED"}


def _split_option(index: int, raw_option: str) -> tuple[str, str]:
    text = str(raw_option).strip()
    if not text:
        raise ValueError("MCQ options must be non-empty strings")
    if index >= 26:
        raise ValueError("At most 26 MCQ options are supported")

    fallback_label = chr(ord("A") + index)
    match = re.match(r"^([A-Za-z])\s*[\.:\)\-]\s*(.+)$", text)
    if not match:
        return fallback_label, text
    return match.group(1).upper(), match.group(2).strip()


def initialize_open_option_candidates(options: Sequence[str]) -> List[Dict[str, Any]]:
    """Create deterministic UNCHECKED hypotheses without source authority."""
    candidates: List[Dict[str, Any]] = []
    labels: set[str] = set()
    for index, raw_option in enumerate(options):
        label, claim = _split_option(index, raw_option)
        if label in labels:
            raise ValueError(f"Duplicate MCQ option label: {label}")
        labels.add(label)
        candidates.append(
            {
                "option": label,
                "claim": claim,
                "status": "UNCHECKED",
                "source_ids": [],
            }
        )
    return candidates


def validate_option_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    labels: set[str] = set()
    for candidate in candidates:
        if set(candidate) != {"option", "claim", "status", "source_ids"}:
            raise ValueError("Invalid MCQ candidate fields")
        label = str(candidate["option"]).strip().upper()
        claim = str(candidate["claim"]).strip()
        status = str(candidate["status"]).strip().upper()
        source_ids = candidate["source_ids"]
        if not label or label in labels:
            raise ValueError("MCQ candidate labels must be unique")
        if not claim:
            raise ValueError("MCQ candidate claims must be non-empty")
        if status not in VALID_CANDIDATE_VERDICTS:
            raise ValueError(f"Invalid MCQ candidate status: {status}")
        if not isinstance(source_ids, list) or any(
            not isinstance(source_id, str) or not source_id.strip()
            for source_id in source_ids
        ):
            raise ValueError("source_ids must be a list of non-empty strings")
        if status in {"SUPPORTED", "CONTRADICTED"} and not source_ids:
            raise ValueError(f"{status} requires at least one audited source")
        labels.add(label)


def apply_hypothesis_verifier_results(
    candidates: Sequence[Mapping[str, Any]],
    verifier_results: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply source-grounded hypothesis verdicts without auto-eliminating rivals.

    Each verifier result must contain ``option``, ``verdict`` and ``source_ids``.
    Absence from a clip is not converted to CONTRADICTED here; that semantic
    judgment belongs to the Hypothesis Verifier.  This reducer only enforces
    that decisive verdicts cite an actual audited source and leaves every
    unmentioned option intact.
    """
    validate_option_candidates(candidates)
    updated = copy.deepcopy(list(candidates))
    by_label = {str(item["option"]).upper(): item for item in updated}
    seen: set[str] = set()

    for result in verifier_results:
        required = {"option", "verdict", "source_ids"}
        if not required.issubset(result):
            raise ValueError("Candidate audit result is missing required fields")
        label = str(result["option"]).strip().upper()
        verdict = str(result["verdict"]).strip().upper()
        source_ids = list(dict.fromkeys(str(v).strip() for v in result["source_ids"]))
        if label not in by_label:
            raise ValueError(f"Audit result references unknown option: {label}")
        if label in seen:
            raise ValueError(f"Duplicate audit result for option: {label}")
        if verdict not in VALID_CANDIDATE_VERDICTS:
            raise ValueError(f"Invalid candidate audit verdict: {verdict}")
        if any(not source_id for source_id in source_ids):
            raise ValueError("Candidate audit source_ids must be non-empty strings")
        if verdict in {"SUPPORTED", "CONTRADICTED"} and not source_ids:
            raise ValueError(f"{verdict} requires at least one audited source")
        by_label[label]["status"] = verdict
        by_label[label]["source_ids"] = source_ids
        seen.add(label)

    validate_option_candidates(updated)
    return updated


def apply_option_candidate_audit(
    candidates: Sequence[Mapping[str, Any]],
    audit_results: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Backward-compatible alias for older experiment callers.

    New code must use :func:`apply_hypothesis_verifier_results`; option
    semantics are no longer owned by the visual Evidence Audit.
    """

    return apply_hypothesis_verifier_results(candidates, audit_results)


def format_option_candidate_context(
    candidates: Sequence[Mapping[str, Any]] | None,
    *,
    force_best_effort: bool = False,
) -> str:
    """Render a prompt section that cannot be mistaken for observed evidence."""
    if not candidates:
        return ""
    validate_option_candidates(candidates)
    lines = [
        "MCQ candidate hypotheses (NOT observed Evidence):",
        "Statuses describe support from cited video sources, not confidence or GT.",
    ]
    for candidate in candidates:
        source_ids = candidate["source_ids"]
        source_text = ", ".join(source_ids) if source_ids else "none"
        lines.append(
            f"- {candidate['option']} [{candidate['status']}]: "
            f"{candidate['claim']} (audited source_ids: {source_text})"
        )
    if not force_best_effort:
        lines.append(
            "UNCHECKED candidates may guide observation but cannot support an answer; "
            "do not treat absence from a partial clip as contradiction."
        )
    supported = [
        str(candidate["option"]).strip().upper()
        for candidate in candidates
        if str(candidate["status"]).strip().upper() == "SUPPORTED"
    ]
    if force_best_effort:
        lines.append(
            "Final-answer guidance: evidence collection has ended. The states above "
            "are diagnostics, not eligibility rules. Prefer SUPPORTED options and "
            "compare their cited evidence first. If none is SUPPORTED, weigh all "
            "supplied options using the question and available evidence, then choose "
            "the most likely one. "
            "Explain any uncertainty or conflicting evidence in your reasoning."
        )
    elif not supported:
        lines.append(
            "Answer authority: no option is SUPPORTED. Do not answer or guess; "
            "return to planning for more audited evidence."
        )
    elif len(supported) == 1:
        lines.append(
            "Answer authority: "
            f"{supported[0]} is the sole eligible option. Check and explain its "
            "audited support. Do not replace it with an OPEN or CONTRADICTED option. "
            "If a concrete counterexample may invalidate its support, report the "
            "specific evidence gap instead of selecting another option."
        )
    else:
        lines.append(
            "Answer authority: compare only within the SUPPORTED set "
            f"({', '.join(supported)}). OPEN and CONTRADICTED options are ineligible."
        )
    return "\n".join(lines)
