from __future__ import annotations

import dataclasses
from io import BytesIO
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import EventMatch, PerceptionEvent, PerceptionMap

MIN_OVERVIEW_FRAMES = 4
MAX_OVERVIEW_FRAMES_PER_REQUEST = 200
MAX_OVERVIEW_REQUESTS = 8


def _event_ref(event: PerceptionEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_index": event.event_index,
        "event_start_s": round(float(event.start_sec), 6),
        "event_end_s": round(float(event.end_sec), 6),
        "representative_timestamps": [
            round(float(timestamp), 6)
            for timestamp in event.representative_timestamps
        ],
    }


def _label_overview_jpeg(
    jpeg: bytes,
    *,
    frame_id: str,
    event_id: str,
    timestamp_sec: float,
) -> bytes:
    """Add an out-of-frame provenance header so the model can cite exact frames."""

    from PIL import Image, ImageDraw, ImageFont

    with Image.open(BytesIO(jpeg)) as source:
        image = source.convert("RGB")
    font_size = max(12, min(24, image.width // 16))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    header_text = (
        f"{frame_id}  {event_id}\n"
        f"source_timestamp={float(timestamp_sec):.3f}s"
    )
    line_box = font.getbbox("Ag")
    line_height = max(1, line_box[3] - line_box[1])
    header_height = max(28, line_height * 2 + 10)
    labeled = Image.new("RGB", (image.width, image.height + header_height), "black")
    labeled.paste(image, (0, header_height))
    ImageDraw.Draw(labeled).multiline_text(
        (6, 4),
        header_text,
        fill="white",
        font=font,
        spacing=2,
    )
    buffer = BytesIO()
    labeled.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def event_refs_for_ranges(
    perception_map: PerceptionMap,
    ranges: Iterable[tuple[float, float]],
) -> tuple[dict[str, Any], ...]:
    """Map grounded source ranges to every overlapping Event, chronologically."""

    normalized: list[tuple[float, float]] = []
    for raw_start, raw_end in ranges:
        start = float(raw_start)
        end = float(raw_end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        normalized.append((start, end))
    selected: list[dict[str, Any]] = []
    for event in perception_map.events:
        if any(start < event.end_sec and end > event.start_sec for start, end in normalized):
            selected.append(_event_ref(event))
    return tuple(selected)


def _requested_regions(plan: Any, duration_sec: float) -> tuple[tuple[float, float], ...]:
    watch = plan.watch
    if str(watch.load_mode) != "region" or not watch.regions:
        return ()
    normalized: list[tuple[float, float]] = []
    for raw_start, raw_end in watch.regions:
        start = max(0.0, float(raw_start))
        end = min(float(duration_sec), float(raw_end))
        if end > start:
            normalized.append((start, end))
    return tuple(normalized)


def _event_aligned_intersections(
    perception_map: PerceptionMap,
    requested_regions: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    intersections: list[tuple[float, float]] = []
    for event in perception_map.events:
        for requested_start, requested_end in requested_regions:
            start = max(event.start_sec, requested_start)
            end = min(event.end_sec, requested_end)
            if end > start:
                intersections.append((float(start), float(end)))
    intersections.sort()
    return tuple(intersections)


def _estimated_frames(match: EventMatch, fps: float) -> int:
    return max(1, int(math.ceil((match.end_sec - match.start_sec) * fps)))


def select_ranked_event_batch(
    matches: Sequence[EventMatch],
    *,
    fps: float,
    frame_budget: int,
    excluded_event_indexes: Iterable[int] = (),
) -> tuple[tuple[EventMatch, ...], int]:
    """Consume an ordered prefix that fits transport budget; never discard the tail."""

    rate = float(fps)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("fps must be finite and positive")
    if frame_budget < 1:
        raise ValueError("frame_budget must be positive")
    excluded = {int(index) for index in excluded_event_indexes}
    candidates = [match for match in matches if match.event_index not in excluded]
    if not candidates:
        raise RuntimeError("event_assist_candidates_exhausted")
    selected: list[EventMatch] = []
    frames = 0
    for match in candidates:
        cost = _estimated_frames(match, rate)
        if selected and frames + cost > frame_budget:
            break
        selected.append(match)
        frames += cost
        if frames >= frame_budget:
            break
    return tuple(selected), len(candidates) - len(selected)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("event_overview_malformed_output")


def _partition_overview_frames(
    frames: Sequence[dict[str, Any]],
    *,
    maximum: int = MAX_OVERVIEW_FRAMES_PER_REQUEST,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if len(frames) < MIN_OVERVIEW_FRAMES:
        raise ValueError("event_overview_requires_at_least_four_frames")
    if maximum < MIN_OVERVIEW_FRAMES:
        raise ValueError("event_overview_maximum_is_too_small")
    batches: list[tuple[dict[str, Any], ...]] = []
    start = 0
    while start < len(frames):
        remaining = len(frames) - start
        size = min(maximum, remaining)
        tail = remaining - size
        if 0 < tail < MIN_OVERVIEW_FRAMES:
            size -= MIN_OVERVIEW_FRAMES - tail
        batches.append(tuple(frames[start : start + size]))
        start += size
    if len(batches) > MAX_OVERVIEW_REQUESTS:
        raise ValueError("event_overview_exceeds_eight_requests")
    if any(
        not MIN_OVERVIEW_FRAMES <= len(batch) <= maximum
        for batch in batches
    ):
        raise RuntimeError("event_overview_partition_contract_invalid")
    return tuple(batches)


class EventAssistRuntime:
    """Default-off adapter between an AVP Plan and its existing Observer."""

    def __init__(
        self,
        *,
        perception_map: PerceptionMap,
        text_encoder: Callable[[str], np.ndarray],
        event_index_run_id: str,
        observation_frame_budget: int = 200,
        frame_jpegs: Mapping[float, bytes] | None = None,
    ) -> None:
        run_id = str(event_index_run_id).strip()
        if not run_id:
            raise ValueError("event_index_run_id is required")
        if observation_frame_budget < 1:
            raise ValueError("observation_frame_budget must be positive")
        self.perception_map = perception_map
        self.text_encoder = text_encoder
        self.event_index_run_id = run_id
        self.observation_frame_budget = int(observation_frame_budget)
        self.frame_jpegs = {
            float(timestamp): bytes(content)
            for timestamp, content in (frame_jpegs or {}).items()
        }
        self._overview_record: dict[str, Any] | None = None

    @property
    def overview_complete(self) -> bool:
        return self._overview_record is not None

    def _event_for_timestamp(self, timestamp: float) -> PerceptionEvent:
        for event in self.perception_map.events:
            if event.start_sec <= timestamp < event.end_sec:
                return event
        if self.perception_map.events and math.isclose(
            timestamp,
            self.perception_map.events[-1].end_sec,
            abs_tol=1e-6,
        ):
            return self.perception_map.events[-1]
        raise ValueError("overview_frame_is_outside_event_map")

    def _jpeg_for_timestamp(self, timestamp: float) -> bytes:
        exact = self.frame_jpegs.get(float(timestamp))
        if exact is not None:
            return exact
        nearby = [
            (abs(float(candidate) - float(timestamp)), content)
            for candidate, content in self.frame_jpegs.items()
        ]
        if not nearby:
            raise ValueError("event_overview_frame_jpegs_required")
        distance, content = min(nearby, key=lambda item: item[0])
        if distance > 1e-6:
            raise ValueError("event_overview_representative_jpeg_missing")
        return content

    def _overview_frames(self) -> tuple[dict[str, Any], ...]:
        scheduled: list[dict[str, Any]] = []
        used: set[float] = set()
        for event in self.perception_map.events:
            for timestamp in event.representative_timestamps:
                second = float(timestamp)
                if second in used:
                    continue
                used.add(second)
                scheduled.append(
                    {
                        "event_id": event.event_id,
                        "event_index": event.event_index,
                        "timestamp_sec": second,
                        "jpeg": self._jpeg_for_timestamp(second),
                    }
                )
        if len(scheduled) < MIN_OVERVIEW_FRAMES:
            for timestamp in self.perception_map.sample_timestamps:
                second = float(timestamp)
                if second in used or second not in self.frame_jpegs:
                    continue
                event = self._event_for_timestamp(second)
                used.add(second)
                scheduled.append(
                    {
                        "event_id": event.event_id,
                        "event_index": event.event_index,
                        "timestamp_sec": second,
                        "jpeg": self._jpeg_for_timestamp(second),
                    }
                )
                if len(scheduled) >= MIN_OVERVIEW_FRAMES:
                    break
        scheduled.sort(
            key=lambda item: (float(item["timestamp_sec"]), int(item["event_index"]))
        )
        for index, item in enumerate(scheduled, 1):
            item["frame_id"] = f"overview_frame_{index:04d}"
            item["jpeg"] = _label_overview_jpeg(
                bytes(item["jpeg"]),
                frame_id=str(item["frame_id"]),
                event_id=str(item["event_id"]),
                timestamp_sec=float(item["timestamp_sec"]),
            )
        event_ids = {item["event_id"] for item in scheduled}
        expected = {event.event_id for event in self.perception_map.events}
        if event_ids != expected:
            raise ValueError("event_overview_does_not_represent_every_event")
        return tuple(scheduled)

    @staticmethod
    def _overview_prompt(
        *,
        query: str,
        options: Sequence[str],
        batch: Sequence[dict[str, Any]],
        batch_index: int,
        batch_count: int,
    ) -> str:
        manifest = "\n".join(
            f"{item['frame_id']} | {item['event_id']} | {item['timestamp_sec']:.3f}s"
            for item in batch
        )
        option_text = "\n".join(f"- {option}" for option in options)
        return f"""You are viewing one chronological packet of representative frames from a complete Event map.
This is navigation only. Do not answer the multiple-choice question and do not claim that a representative frame fully proves an option.
Identify concrete visible clues and the Event ranges that deserve denser rereading. Every clue must cite only frame_id and event_id values from this packet.

Question:
{query}

Options:
{option_text}

Packet {batch_index}/{batch_count}; frames are attached in exactly this order:
{manifest}

Every attached frame has a visible black provenance header. Read that header
and cite its exact frame_id and event_id; do not infer an ID by counting frames.

Return one JSON object only:
{{
  "overview": "brief description of what this packet visibly contains",
  "clues": [
    {{
      "observation": "concrete visible navigation clue",
      "event_ids": ["event_000001"],
      "frame_ids": ["overview_frame_0001"],
      "reason_to_reinspect": "what denser rereading should verify"
    }}
  ]
}}
An empty clues list is allowed when nothing in this packet is relevant."""

    @staticmethod
    def _validated_packet_output(
        raw: str,
        *,
        batch: Sequence[dict[str, Any]],
        packet_index: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        payload = _parse_json_object(raw)
        overview = str(payload.get("overview") or "").strip()
        clues = payload.get("clues")
        if not overview or not isinstance(clues, list):
            raise ValueError("event_overview_output_contract_invalid")
        valid_events = {str(item["event_id"]) for item in batch}
        valid_frames = {str(item["frame_id"]) for item in batch}
        event_by_frame = {
            str(item["frame_id"]): str(item["event_id"])
            for item in batch
        }
        validated: list[dict[str, Any]] = []
        for clue in clues:
            if not isinstance(clue, dict):
                raise ValueError("event_overview_clue_must_be_object")
            observation = str(clue.get("observation") or "").strip()
            event_ids = clue.get("event_ids")
            frame_ids = clue.get("frame_ids")
            reason = str(clue.get("reason_to_reinspect") or "").strip()
            if (
                not observation
                or not reason
                or not isinstance(event_ids, list)
                or not event_ids
                or not isinstance(frame_ids, list)
                or not frame_ids
            ):
                raise ValueError("event_overview_clue_contract_invalid")
            normalized_events = [str(value) for value in event_ids]
            normalized_frames = [str(value) for value in frame_ids]
            if not set(normalized_events) <= valid_events:
                raise ValueError("event_overview_clue_has_unknown_event")
            if not set(normalized_frames) <= valid_frames:
                raise ValueError("event_overview_clue_has_unknown_frame")
            cited_frame_events = {
                event_by_frame[frame_id] for frame_id in normalized_frames
            }
            if not cited_frame_events <= set(normalized_events):
                raise ValueError("event_overview_clue_frame_event_mismatch")
            validated.append(
                {
                    "packet_index": int(packet_index),
                    "observation": observation,
                    "event_ids": normalized_events,
                    "frame_ids": normalized_frames,
                    "reason_to_reinspect": reason,
                }
            )
        return overview, validated

    def run_initial_overview(
        self,
        *,
        client: Any,
        query: str,
        options: Sequence[str],
    ) -> dict[str, Any]:
        if self._overview_record is not None:
            return dict(self._overview_record)
        infer = getattr(client, "infer_on_event_overview", None)
        if not callable(infer):
            raise RuntimeError("event_overview_client_not_supported")
        frames = self._overview_frames()
        batches = _partition_overview_frames(
            frames,
            maximum=min(
                self.observation_frame_budget,
                MAX_OVERVIEW_FRAMES_PER_REQUEST,
            ),
        )
        packet_overviews: list[dict[str, Any]] = []
        navigation_clues: list[dict[str, Any]] = []
        for packet_index, batch in enumerate(batches, 1):
            prompt = self._overview_prompt(
                query=query,
                options=options,
                batch=batch,
                batch_index=packet_index,
                batch_count=len(batches),
            )
            raw = infer(
                prompt=prompt,
                jpeg_frames=tuple(bytes(item["jpeg"]) for item in batch),
            )
            overview, clues = self._validated_packet_output(
                raw,
                batch=batch,
                packet_index=packet_index,
            )
            packet_overviews.append(
                {
                    "packet_index": packet_index,
                    "overview": overview,
                    "raw_response": str(raw),
                }
            )
            navigation_clues.extend(clues)
        public_frames = [
            {key: value for key, value in item.items() if key != "jpeg"}
            for item in frames
        ]
        self._overview_record = {
            "schema_version": "event_overview_navigation_v1",
            "mode": "NAVIGATION_ONLY",
            "event_index_run_id": self.event_index_run_id,
            "coverage_complete": True,
            "event_count": len(self.perception_map.events),
            "frame_count": len(public_frames),
            "physical_request_count": len(batches),
            "audit_executed": False,
            "answer_eligible": False,
            "visible_frame_provenance_labels": True,
            "frames": public_frames,
            "packet_overviews": packet_overviews,
            "navigation_clues": navigation_clues,
        }
        return dict(self._overview_record)

    def planning_context_text(self) -> str:
        if self._overview_record is None:
            raise RuntimeError("event_overview_has_not_completed")
        lines = [
            "EVENT OVERVIEW — NAVIGATION ONLY; NOT EVIDENCE.",
            "Choose explicit time region(s) for dense raw-video rereading. Do not answer from these clues.",
            "Event catalog:",
        ]
        for event in self.perception_map.events:
            reps = ", ".join(
                f"{timestamp:.3f}s" for timestamp in event.representative_timestamps
            )
            lines.append(
                f"- {event.event_id}: [{event.start_sec:.3f}, {event.end_sec:.3f}) reps={reps}"
            )
        lines.append("Navigation clues:")
        clues = self._overview_record.get("navigation_clues") or []
        if not clues:
            lines.append("- No question-relevant clue was identified; choose a justified Event range to inspect.")
        for clue in clues:
            event_ids = ", ".join(clue["event_ids"])
            lines.append(
                f"- {event_ids}: {clue['observation']} Reinspect to verify: {clue['reason_to_reinspect']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _need_key(plan: Any) -> str:
        need_id = str(getattr(plan, "need_id", "") or "").strip()
        if need_id:
            return need_id
        description = str(getattr(plan, "description", "") or "").strip()
        return description or str(getattr(plan, "query", "") or "").strip()

    def _already_observed(self, plan: Any, blackboard: Any) -> set[int]:
        current_need = self._need_key(plan)
        current_fps = float(plan.watch.fps)
        observed: set[int] = set()
        for evidence in getattr(blackboard, "evidences", ()):
            model_call = getattr(evidence, "model_call", None)
            if not isinstance(model_call, dict):
                continue
            metadata = model_call.get("event_assist")
            if not isinstance(metadata, dict) or metadata.get("need_key") != current_need:
                continue
            try:
                previous_fps = float(metadata["fps"])
            except (KeyError, TypeError, ValueError):
                continue
            if previous_fps + 1e-9 < current_fps:
                continue
            for event_index in metadata.get("selected_event_indexes", []):
                try:
                    observed.add(int(event_index))
                except (TypeError, ValueError):
                    continue
        return observed

    def _encode_need(self, text: str) -> np.ndarray:
        encoded = np.asarray(self.text_encoder(text), dtype=np.float32)
        if encoded.ndim == 2 and len(encoded) == 1:
            encoded = encoded[0]
        if encoded.ndim != 1:
            raise ValueError("text_encoder must return one embedding vector")
        return encoded

    def prepare_observation(
        self,
        *,
        plan: Any,
        blackboard: Any,
    ) -> tuple[Any, dict[str, Any]]:
        duration = float(blackboard.duration_sec)
        requested = _requested_regions(plan, duration)
        need_key = self._need_key(plan)
        if requested:
            observation_regions = _event_aligned_intersections(
                self.perception_map, requested
            )
            if not observation_regions:
                raise RuntimeError("event_assist_explicit_scope_has_no_event")
            selected_refs = event_refs_for_ranges(
                self.perception_map, observation_regions
            )
            ranking_basis = "explicit_scope_chronological"
            remaining_event_count = 0
        else:
            raise RuntimeError("event_overview_refinement_requires_explicit_region")

        watch = dataclasses.replace(
            plan.watch,
            load_mode="region",
            regions=list(observation_regions),
        )
        prepared_plan = dataclasses.replace(plan, watch=watch)
        metadata = {
            "schema_version": "event_assist_receipt_v1",
            "event_index_run_id": self.event_index_run_id,
            "ranking_basis": ranking_basis,
            "need_key": need_key,
            "need_id": getattr(plan, "need_id", None),
            "fps": float(plan.watch.fps),
            "selected_event_ids": [item["event_id"] for item in selected_refs],
            "selected_event_indexes": [
                int(item["event_index"]) for item in selected_refs
            ],
            "planned_source_events": list(selected_refs),
            "remaining_event_count": int(remaining_event_count),
        }
        return prepared_plan, metadata

    def annotate_observation(
        self,
        *,
        evidence: Any,
        preparation: dict[str, Any],
    ) -> None:
        actual_ranges: list[tuple[float, float]] = []
        for item in getattr(evidence, "frames_used", ()):
            if not isinstance(item, dict):
                continue
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                actual_ranges.append((start, end))
        actual_refs = event_refs_for_ranges(self.perception_map, actual_ranges)
        metadata = dict(preparation)
        metadata["source_events"] = list(actual_refs)
        model_call = getattr(evidence, "model_call", None)
        if not isinstance(model_call, dict):
            model_call = {}
            evidence.model_call = model_call
        model_call["event_assist"] = metadata
        receipt = model_call.get("observation_receipt")
        if isinstance(receipt, dict):
            receipt["event_index_run_id"] = self.event_index_run_id
            receipt["source_events"] = list(actual_refs)

        for item in getattr(evidence, "key_evidence", ()):
            if not isinstance(item, dict):
                continue
            try:
                start = float(item["timestamp_start"])
                end = float(item["timestamp_end"])
            except (KeyError, TypeError, ValueError):
                continue
            refs = event_refs_for_ranges(self.perception_map, ((start, end),))
            if refs:
                item["source_event_ids"] = [ref["event_id"] for ref in refs]
