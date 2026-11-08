"""Observable trace recorder for an unchanged AVP run.

The runner does not change AVP prompts, model payloads, model responses, or
control flow.  By default it also performs fixed ForceAnswer replays over
mechanically selected evidence subsets; --native-only disables every replay.
The output contains candidate phenomena only; it never assigns a pollution
verdict.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Iterator, Optional

from avp.config import load_config
from avp.internvl_adapter import InternVLClient
from avp.main import Controller, Evidence, Reflector, plan_from_dict
from avp.prompt import EVIDENCE_SCHEMA, parse_json_response, validate_against_schema
from avp.qwen_adapter import QwenClient
from avp.video_utils import set_metadata_source
from experiments.semantic_delta_pilot.avp_integration import (
    OpenAIJSONBackend,
    RepairController,
)


SCHEMA_VERSION = "avp_pollution_observable_v1"
ALLOWED_QA_FIELDS = {"video_id", "question_id", "question", "duration", "options"}
REQUIRED_QA_FIELDS = {"video_id", "question_id", "question"}
SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _read_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("QA input JSON must be a list")
        return value
    values: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on QA input line {line_number}: {exc}") from exc
    return values


def load_qa_only(path: Path) -> list[dict[str, Any]]:
    """Load the inference projection; options are allowed, answer labels are not."""
    rows = _read_json_or_jsonl(path)
    clean: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"QA input row {index} must be an object")
        keys = set(row)
        missing = REQUIRED_QA_FIELDS - keys
        extra = keys - ALLOWED_QA_FIELDS
        if missing:
            raise ValueError(f"QA input row {index} is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(
                f"Inference guard rejected row {index}; unapproved fields: {sorted(extra)}"
            )
        for name in REQUIRED_QA_FIELDS:
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"QA input row {index} field {name!r} must be a non-empty string")
        if "duration" in row and row["duration"] is not None:
            duration = _duration_value(row["duration"])
            if duration is None or duration <= 0:
                raise ValueError(f"QA input row {index} field 'duration' must be a positive scalar")
        if "options" in row:
            options = row["options"]
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 6
                or any(not isinstance(option, str) or not option.strip() for option in options)
            ):
                raise ValueError(
                    f"QA input row {index} field 'options' must contain 2 to 6 non-empty strings"
                )
        if not SAFE_VIDEO_ID.fullmatch(row["video_id"]):
            raise ValueError(f"QA input row {index} has unsafe video_id")
        if row["question_id"] in seen_question_ids:
            raise ValueError(f"Duplicate question_id: {row['question_id']}")
        seen_question_ids.add(row["question_id"])
        clean.append(dict(row))
    return clean


def format_mcq_query(question: str, options: Optional[list[str]]) -> str:
    """Use AVP's existing MCQ query format without exposing an answer label."""
    if not options:
        return question
    return question + "\n\nOptions:\n" + "\n".join(options)


def select_questions(
    rows: Iterable[dict[str, Any]], question_ids: list[str]
) -> list[dict[str, Any]]:
    if not question_ids:
        raise ValueError("At least one --question-id is required; refusing to run the full dataset")
    by_id = {row["question_id"]: row for row in rows}
    missing = [question_id for question_id in question_ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Unknown question_id values: {missing}")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("--question-id values must be unique")
    return [by_id[question_id] for question_id in question_ids]


def resolve_video(video_root: Path, video_id: str) -> Path:
    root = video_root.resolve(strict=True)
    path = (root / f"{video_id}.mp4").resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Resolved video escaped video root: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Video is not a non-empty file: {path}")
    return path


def _duration_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    text = str(value).strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _part_metadata(part: Any) -> dict[str, Any]:
    metadata = getattr(part, "video_metadata", None)
    if metadata is None:
        return {"fps": None, "start_offset_sec": None, "end_offset_sec": None}
    return {
        "fps": _duration_value(getattr(metadata, "fps", None)),
        "start_offset_sec": _duration_value(getattr(metadata, "start_offset", None)),
        "end_offset_sec": _duration_value(getattr(metadata, "end_offset", None)),
    }


def probe_media(path: Path) -> dict[str, Any]:
    """Read container/stream facts only; selected vLLM frame PTS remain unknown."""
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "byte_count": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_file():
        return record
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            record["probe_error"] = completed.stderr.strip() or f"ffprobe exit {completed.returncode}"
            return record
        parsed = json.loads(completed.stdout)
        stream = (parsed.get("streams") or [{}])[0]
        fmt = parsed.get("format") or {}
        record.update(
            {
                "codec": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "avg_frame_rate": stream.get("avg_frame_rate"),
                "r_frame_rate": stream.get("r_frame_rate"),
                "container_frame_count": stream.get("nb_frames"),
                "stream_duration_sec": _duration_value(stream.get("duration")),
                "container_duration_sec": _duration_value(fmt.get("duration")),
            }
        )
    except Exception as exc:  # logging must never change AVP success/failure
        record["probe_error"] = f"{type(exc).__name__}: {exc}"
    return record


def media_duration_sec(media_probe: dict[str, Any]) -> float:
    for name in ("container_duration_sec", "stream_duration_sec"):
        value = media_probe.get(name)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    raise ValueError(f"No positive video duration was established for {media_probe.get('path')}")


def write_qa_only_metadata_source(path: Path, video_path: Path, video_id: str, duration_sec: float) -> None:
    """Write the smallest metadata source AVP needs; it contains no QA labels."""
    write_json(
        path,
        [
            {
                "path": str(video_path),
                "duration": duration_sec,
                "video_id": video_id,
            }
        ],
    )


def _inline_data_summary(url: str) -> dict[str, Any]:
    header, _, encoded = url.partition(",")
    mime_type = header[5:].split(";", 1)[0] if header.startswith("data:") else None
    padding = len(encoded) - len(encoded.rstrip("="))
    byte_count = max(0, len(encoded) * 3 // 4 - padding)
    return {
        "transport": "inline_data",
        "mime_type": mime_type,
        "byte_count": byte_count,
        "binary_logged": False,
    }


def sanitize_qwen_payload(payload: dict[str, Any], transport: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy a request without ever copying base64 video data into the trace."""
    clean: dict[str, Any] = {}
    media: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key != "messages":
            clean[key] = value
            continue
        clean_messages: list[dict[str, Any]] = []
        for message in value:
            clean_message = {k: v for k, v in message.items() if k != "content"}
            content = message.get("content")
            if not isinstance(content, list):
                clean_message["content"] = content
                clean_messages.append(clean_message)
                continue
            clean_content: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "video_url":
                    clean_content.append(part)
                    continue
                url = str((part.get("video_url") or {}).get("url", ""))
                if url.startswith("data:"):
                    summary = _inline_data_summary(url)
                    clean_content.append({"type": "video_url", "video_url": summary})
                    media.append(dict(summary))
                    continue
                resolved = transport.media.resolve(url)
                summary = {
                    "transport": "loopback_url",
                    "url": url,
                    "resolved_path": str(resolved) if resolved is not None else None,
                }
                clean_content.append({"type": "video_url", "video_url": summary})
                media_record = dict(summary)
                if resolved is not None:
                    media_record["media_probe"] = probe_media(Path(resolved))
                media.append(media_record)
            clean_message["content"] = clean_content
            clean_messages.append(clean_message)
        clean["messages"] = clean_messages
    return clean, media


def classify_call(text: str, has_video: bool) -> str:
    if has_video:
        return "observe"
    stripped = text.lstrip()
    if stripped.startswith("You are an expert video analysis planner"):
        return "plan_initial"
    if stripped.startswith("You are replanning a video observation"):
        return "replan"
    if stripped.startswith("**Reflector prompt (evidence sufficiency checker)**"):
        return "reflect"
    if stripped.startswith("You are synthesizing the final answer"):
        return "force_answer"
    return "unknown"


class ObservableTraceRecorder:
    """Transparent wrappers around AVP's media creation and provider boundary."""

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self.records: list[dict[str, Any]] = []
        self.media_creations: list[dict[str, Any]] = []
        self.plan_returns: list[dict[str, Any]] = []
        self.observe_returns: list[dict[str, Any]] = []
        self._parts: dict[int, dict[str, Any]] = {}
        self._call_count = 0
        self._observe_count = 0
        self._active_record: Optional[dict[str, Any]] = None
        self._mode = "native"
        self._replay_id: Optional[str] = None
        self._replay_evidence_rounds: Optional[list[int]] = None
        self._client: Any = None
        self._transport: Any = None
        self._original_create_video_part: Any = None
        self._original_plan: Any = None
        self._original_infer_on_video: Any = None
        self._original_generate_content: Any = None
        self._original_post_chat_completion: Any = None

    def attach(self, client: Any) -> None:
        if self._client is not None:
            raise RuntimeError("Recorder is already attached")
        if client.client is None:
            raise RuntimeError("Qwen client must be initialized before attaching recorder")
        self._client = client
        self._transport = client.client
        self._original_create_video_part = client.create_video_part
        self._original_plan = client.plan
        self._original_infer_on_video = client.infer_on_video
        self._original_generate_content = self._transport.models.generate_content
        self._original_post_chat_completion = self._transport.post_chat_completion
        client.create_video_part = self._create_video_part
        client.plan = self._plan
        client.infer_on_video = self._infer_on_video
        self._transport.models.generate_content = self._generate_content
        self._transport.post_chat_completion = self._post_chat_completion

    def detach(self) -> None:
        if self._client is None:
            return
        self._client.create_video_part = self._original_create_video_part
        self._client.plan = self._original_plan
        self._client.infer_on_video = self._original_infer_on_video
        self._transport.models.generate_content = self._original_generate_content
        self._transport.post_chat_completion = self._original_post_chat_completion
        self._client = None
        self._transport = None

    @contextmanager
    def replay_phase(self, replay_id: str, evidence_rounds: list[int]) -> Iterator[None]:
        old = (self._mode, self._replay_id, self._replay_evidence_rounds)
        self._mode = "counterfactual_replay"
        self._replay_id = replay_id
        self._replay_evidence_rounds = list(evidence_rounds)
        try:
            yield
        finally:
            self._mode, self._replay_id, self._replay_evidence_rounds = old

    def _create_video_part(self, *args: Any, **kwargs: Any) -> Any:
        names = [
            "video_path",
            "fps",
            "start_offset",
            "end_offset",
            "media_resolution",
            "duration_sec",
        ]
        inputs = dict(kwargs)
        for name, value in zip(names, args):
            inputs.setdefault(name, value)
        started = time.monotonic()
        record: dict[str, Any] = {
            "media_creation_id": f"media_creation_{len(self.media_creations) + 1}",
            "requested": {name: inputs.get(name) for name in names},
            "status": "running",
        }
        source = inputs.get("video_path")
        if source:
            try:
                record["source_media"] = probe_media(Path(str(source)).resolve())
            except Exception as exc:
                record["source_media"] = {"path": str(source), "probe_error": f"{type(exc).__name__}: {exc}"}
        try:
            part = self._original_create_video_part(*args, **kwargs)
            record["returned_video_metadata"] = _part_metadata(part)
            file_data = getattr(part, "file_data", None)
            inline_data = getattr(part, "inline_data", None)
            if file_data is not None and getattr(file_data, "file_uri", None):
                record["returned_media"] = {
                    "kind": "file_data",
                    "uri": str(file_data.file_uri),
                    "mime_type": str(getattr(file_data, "mime_type", "") or ""),
                }
            elif inline_data is not None and getattr(inline_data, "data", None) is not None:
                record["returned_media"] = {
                    "kind": "inline_data",
                    "mime_type": str(getattr(inline_data, "mime_type", "") or ""),
                    "byte_count": len(bytes(inline_data.data)),
                    "binary_logged": False,
                }
            else:
                record["returned_media"] = {"kind": "unknown"}
            record["status"] = "completed"
            self._parts[id(part)] = record
            return part
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["elapsed_sec"] = round(time.monotonic() - started, 6)
            self.media_creations.append(record)

    def _plan(self, *args: Any, **kwargs: Any) -> Any:
        """Record the PlanSpec AVP actually returns, including fallback plans."""
        prior = kwargs.get("prior")
        if prior is None and len(args) >= 3:
            prior = args[2]
        event: dict[str, Any] = {
            "kind": "replan" if prior is not None and getattr(prior, "evidences", []) else "plan_initial",
            "status": "running",
        }
        started = time.monotonic()
        try:
            plan = self._original_plan(*args, **kwargs)
            event["returned_plan"] = dataclasses.asdict(plan)
            event["status"] = "completed"
            return plan
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            event["elapsed_sec"] = round(time.monotonic() - started, 6)
            matching = [
                record
                for record in self.records
                if record["mode"] == self._mode and record["stage"] == event["kind"]
            ]
            if matching:
                event["provider_call_id"] = matching[-1]["call_id"]
                event["round"] = matching[-1]["round"]
            self.plan_returns.append(event)

    def _infer_on_video(self, *args: Any, **kwargs: Any) -> Any:
        """Record Observer's effective request and the normalized returned Evidence."""
        names = [
            "video_path",
            "duration_sec",
            "sub_query",
            "context",
            "start_sec",
            "end_sec",
            "watch_cfg",
            "step_id",
            "original_query",
        ]
        inputs = dict(kwargs)
        for name, value in zip(names, args):
            inputs.setdefault(name, value)
        watch = inputs.get("watch_cfg")
        event: dict[str, Any] = {
            "status": "running",
            "effective_request": {
                "video_path": inputs.get("video_path"),
                "duration_sec": inputs.get("duration_sec"),
                "start_sec": inputs.get("start_sec"),
                "end_sec": inputs.get("end_sec"),
                "watch": dataclasses.asdict(watch) if dataclasses.is_dataclass(watch) else None,
                "sub_query": inputs.get("sub_query"),
                "accumulated_context": inputs.get("context"),
                "original_query": inputs.get("original_query"),
            },
        }
        started = time.monotonic()
        try:
            evidence = self._original_infer_on_video(*args, **kwargs)
            event["normalized_evidence"] = dataclasses.asdict(evidence)
            matching = [
                record
                for record in self.records
                if record["mode"] == self._mode and record["stage"] == "observe"
            ]
            if matching:
                raw_response = matching[-1].get("raw_response", "")
                parsed = matching[-1].get("parsed_response")
                if parsed and validate_against_schema(parsed, EVIDENCE_SCHEMA):
                    event["parser_path"] = "schema_valid_structured_json"
                else:
                    extracted = self._client._extract_json_field(raw_response, "detailed_response")
                    event["parser_path"] = (
                        "field_regex_fallback"
                        if extracted and extracted != raw_response
                        else "timestamp_regex_fallback"
                    )
            event["status"] = "completed"
            return evidence
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            event["elapsed_sec"] = round(time.monotonic() - started, 6)
            matching = [
                record
                for record in self.records
                if record["mode"] == self._mode and record["stage"] == "observe"
            ]
            if matching:
                event["provider_call_id"] = matching[-1]["call_id"]
                event["round"] = matching[-1]["round"]
            self.observe_returns.append(event)

    def _snapshot_contents(self, contents: Any) -> tuple[list[dict[str, Any]], str, bool]:
        items = contents if isinstance(contents, (list, tuple)) else [contents]
        snapshot: list[dict[str, Any]] = []
        texts: list[str] = []
        has_video = False
        for item in items:
            if isinstance(item, str):
                snapshot.append({"kind": "text", "text": item})
                texts.append(item)
                continue
            text = getattr(item, "text", None)
            if text is not None:
                text = str(text)
                snapshot.append({"kind": "text", "text": text})
                texts.append(text)
                continue
            has_video = True
            entry: dict[str, Any] = {
                "kind": "video",
                "video_metadata": _part_metadata(item),
            }
            creation = self._parts.get(id(item))
            if creation is not None:
                entry["media_creation_id"] = creation["media_creation_id"]
            file_data = getattr(item, "file_data", None)
            inline_data = getattr(item, "inline_data", None)
            if file_data is not None and getattr(file_data, "file_uri", None):
                uri = str(file_data.file_uri)
                resolved = self._transport.media.resolve(uri)
                entry["source"] = {
                    "kind": "file_data",
                    "uri": uri,
                    "mime_type": str(getattr(file_data, "mime_type", "") or ""),
                    "resolved_path": str(resolved) if resolved is not None else None,
                }
            elif inline_data is not None and getattr(inline_data, "data", None) is not None:
                entry["source"] = {
                    "kind": "inline_data",
                    "mime_type": str(getattr(inline_data, "mime_type", "") or ""),
                    "byte_count": len(bytes(inline_data.data)),
                    "binary_logged": False,
                }
            else:
                entry["source"] = {"kind": "unknown"}
            snapshot.append(entry)
        return snapshot, "\n".join(texts), has_video

    def _stage_round(self, stage: str) -> Optional[int]:
        if self._mode == "counterfactual_replay":
            return None
        if stage == "observe":
            self._observe_count += 1
            return self._observe_count
        if stage in {"reflect", "force_answer"}:
            return self._observe_count or None
        if stage == "plan_initial":
            return 1
        if stage == "replan":
            return self._observe_count + 1
        return None

    def _generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self._call_count += 1
        snapshot, prompt, has_video = self._snapshot_contents(contents)
        stage = classify_call(prompt, has_video)
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "call_id": f"call_{self._call_count:03d}",
            "call_ordinal": self._call_count,
            "mode": self._mode,
            "replay_id": self._replay_id,
            "replay_evidence_rounds": self._replay_evidence_rounds,
            "stage": stage,
            "round": self._stage_round(stage),
            "started_at": utc_now(),
            "model": model,
            "gemini_request": {
                "contents": snapshot,
                "config_media_resolution": str(getattr(config, "media_resolution", "") or "") if config is not None else None,
            },
            "status": "running",
        }
        started = time.monotonic()
        if self._active_record is not None:
            raise RuntimeError("Nested generate_content calls are not supported by the recorder")
        self._active_record = record
        try:
            response = self._original_generate_content(model=model, contents=contents, config=config)
            response_text = getattr(response, "text", str(response))
            record["raw_response"] = response_text
            record["parsed_response"] = parse_json_response(response_text)
            record["status"] = "completed"
            return response
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["finished_at"] = utc_now()
            record["elapsed_sec"] = round(time.monotonic() - started, 6)
            self._active_record = None
            self.records.append(record)
            append_jsonl(self.trace_path, record)

    def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        record = self._active_record
        if record is not None:
            clean, media = sanitize_qwen_payload(payload, self._transport)
            record["qwen_request"] = clean
            record["transport_media"] = media
            record["vllm_selected_frame_pts"] = {
                "status": "unknown",
                "reason": "The OpenAI-compatible endpoint does not expose selected frame PTS.",
            }
        started = time.monotonic()
        try:
            response = self._original_post_chat_completion(payload)
            if record is not None:
                record["qwen_raw_response"] = response
            return response
        finally:
            if record is not None:
                record["qwen_request_elapsed_sec"] = round(time.monotonic() - started, 6)


def replay_subsets(round_count: int) -> list[dict[str, Any]]:
    if round_count <= 0:
        return []
    full = tuple(range(1, round_count + 1))
    labels: dict[tuple[int, ...], list[str]] = {full: ["full_evidence", "prefix"]}
    order: list[tuple[int, ...]] = [full]

    for end in range(1, round_count):
        subset = tuple(range(1, end + 1))
        if subset not in labels:
            labels[subset] = []
            order.append(subset)
        labels[subset].append("prefix")

    if round_count > 1:
        for omitted in range(1, round_count + 1):
            subset = tuple(value for value in full if value != omitted)
            if subset not in labels:
                labels[subset] = []
                order.append(subset)
            labels[subset].append(f"leave_out_round_{omitted}")

    return [
        {
            "replay_id": "evidence_" + "_".join(str(value) for value in subset),
            "evidence_rounds": list(subset),
            "selection_kinds": labels[subset],
        }
        for subset in order
    ]


def _answer_view(final_answer: Optional[dict[str, Any]]) -> dict[str, Any]:
    value = final_answer or {}
    text = value.get("selected_option_text") or value.get("reasoning") or ""
    normalized = " ".join(str(text).split()).casefold()
    confidence = value.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {
        "answer_text": str(text),
        "normalized_answer_text": normalized,
        "confidence": confidence,
    }


def run_counterfactual_replays(
    recorder: ObservableTraceRecorder,
    controller: Controller,
    native_result: dict[str, Any],
    query: str,
    options: Optional[list[str]],
) -> list[dict[str, Any]]:
    evidences = list(controller.bb.evidences)
    prompt_ordered_evidences = controller.bb.get_evidence_list()
    subsets = replay_subsets(len(evidences))
    results: list[dict[str, Any]] = []
    plan = plan_from_dict(native_result["plan"])
    for spec in subsets:
        selected_rounds = set(spec["evidence_rounds"])
        selected = [
            evidence
            for evidence in prompt_ordered_evidences
            if evidence.round_id in selected_rounds
        ]
        replay: dict[str, Any] = dict(spec)
        replay["status"] = "running"
        started = time.monotonic()
        try:
            with recorder.replay_phase(spec["replay_id"], spec["evidence_rounds"]):
                reflection = Reflector(controller.client).reflect(
                    query=query,
                    plan=plan,
                    evidence_list=selected,
                    interaction_history=[],
                    video_path=controller.bb.video_path,
                    duration_sec=controller.bb.duration_sec,
                    is_last_round=True,
                    options=options,
                )
            replay["reflection"] = reflection
            replay["answer"] = _answer_view(reflection.get("final_answer"))
            replay["status"] = "completed"
        except Exception as exc:
            replay["status"] = "failed"
            replay["error"] = {"type": type(exc).__name__, "message": str(exc)}
        replay["elapsed_sec"] = round(time.monotonic() - started, 6)
        results.append(replay)

    baseline = next(
        (item for item in results if "full_evidence" in item["selection_kinds"] and item["status"] == "completed"),
        None,
    )
    if baseline is not None:
        base_answer = baseline["answer"]
        for item in results:
            if item["status"] != "completed":
                continue
            answer = item["answer"]
            item["comparison_to_full_evidence"] = {
                "answer_text_changed": answer["normalized_answer_text"] != base_answer["normalized_answer_text"],
                "confidence_delta": (
                    answer["confidence"] - base_answer["confidence"]
                    if answer["confidence"] is not None and base_answer["confidence"] is not None
                    else None
                ),
            }
    return results


def _within_any_range(start: float, end: float, ranges: list[dict[str, Any]]) -> bool:
    # AVP deliberately normalizes evidence intervals to full seconds with
    # floor(start)/ceil(end).  Compare like with like so an ordinary final
    # fractional second is not mislabeled as an out-of-range candidate.
    return any(
        math.floor(float(item.get("start", 0.0))) - 1e-6 <= start
        and end <= math.ceil(float(item.get("end", 0.0))) + 1e-6
        for item in ranges
    )


def build_observable_chain(
    sample: dict[str, Any],
    video_path: Path,
    controller: Controller,
    recorder: ObservableTraceRecorder,
    native_result: dict[str, Any],
    replays: list[dict[str, Any]],
) -> dict[str, Any]:
    native_records = [record for record in recorder.records if record["mode"] == "native"]
    plan_by_round: dict[int, dict[str, Any]] = {}
    for record in native_records:
        if record["stage"] not in {"plan_initial", "replan"} or record["round"] is None:
            continue
        parsed = record.get("parsed_response") or {}
        steps = parsed.get("steps") if isinstance(parsed, dict) else None
        plan_by_round[int(record["round"])] = {
            "call_id": record["call_id"],
            "raw_response": record.get("raw_response"),
            "parsed_step": steps[0] if isinstance(steps, list) and steps else None,
        }
    for event in recorder.plan_returns:
        if event.get("status") != "completed" or event.get("round") is None:
            continue
        entry = plan_by_round.setdefault(int(event["round"]), {})
        entry["avp_returned_plan"] = event.get("returned_plan")

    candidate_events: list[dict[str, Any]] = []
    normalization_events: list[dict[str, Any]] = []
    exact_evidence_seen: dict[tuple[Any, ...], str] = {}
    rounds: list[dict[str, Any]] = []
    for evidence in controller.bb.evidences:
        evidence_dict = dataclasses.asdict(evidence)
        observe = next(
            (
                record
                for record in native_records
                if record["stage"] == "observe" and record["round"] == evidence.round_id
            ),
            None,
        )
        observe_return = next(
            (
                event
                for event in recorder.observe_returns
                if event.get("round") == evidence.round_id
            ),
            None,
        )
        reflect = next(
            (
                record
                for record in native_records
                if record["stage"] in {"reflect", "force_answer"}
                and record["round"] == evidence.round_id
            ),
            None,
        )
        evidence_items: list[dict[str, Any]] = []
        raw_parsed = observe.get("parsed_response") if observe else None
        raw_key_evidence = raw_parsed.get("key_evidence") if isinstance(raw_parsed, dict) else None
        if isinstance(raw_key_evidence, list) and raw_key_evidence != evidence.key_evidence:
            normalization_events.append(
                {
                    "kind": "avp_full_second_normalization",
                    "round": evidence.round_id,
                    "raw_key_evidence": raw_key_evidence,
                    "normalized_key_evidence": evidence.key_evidence,
                }
            )
        for index, item in enumerate(evidence.key_evidence, 1):
            evidence_id = f"round_{evidence.round_id}_evidence_{index}"
            start = item.get("timestamp_start") if isinstance(item, dict) else None
            end = item.get("timestamp_end") if isinstance(item, dict) else None
            description = item.get("description", "") if isinstance(item, dict) else ""
            in_range = None
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                in_range = _within_any_range(float(start), float(end), evidence.frames_used)
                if not in_range:
                    candidate_events.append(
                        {
                            "kind": "evidence_timestamp_outside_requested_ranges",
                            "evidence_id": evidence_id,
                            "round": evidence.round_id,
                            "timestamp_start": start,
                            "timestamp_end": end,
                            "requested_ranges": evidence.frames_used,
                        }
                    )
            duplicate_key = (start, end, " ".join(str(description).split()).casefold())
            duplicate_of = exact_evidence_seen.get(duplicate_key)
            if duplicate_of is not None:
                candidate_events.append(
                    {
                        "kind": "exact_duplicate_evidence",
                        "evidence_id": evidence_id,
                        "duplicate_of": duplicate_of,
                    }
                )
            else:
                exact_evidence_seen[duplicate_key] = evidence_id
            evidence_items.append(
                {
                    "evidence_id": evidence_id,
                    "timestamp_start": start,
                    "timestamp_end": end,
                    "description": description,
                    "inside_requested_ranges": in_range,
                    "duplicate_of": duplicate_of,
                    "source_observe_call_id": observe["call_id"] if observe else None,
                }
            )
        rounds.append(
            {
                "round": evidence.round_id,
                "plan": plan_by_round.get(evidence.round_id),
                "observe_call_id": observe["call_id"] if observe else None,
                "requested_ranges": evidence.frames_used,
                "observer_effective_request": observe_return.get("effective_request") if observe_return else None,
                "parser_path": observe_return.get("parser_path") if observe_return else None,
                "model_call_metadata": evidence.model_call,
                "observer_raw_response": observe.get("raw_response") if observe else None,
                "parsed_evidence": evidence_dict,
                "downstream_propagation": {
                    "included_by_blackboard_summary": ["detailed_response", "key_evidence"],
                    "not_included_by_blackboard_summary": ["reasoning"],
                },
                "evidence_items": evidence_items,
                "reflect_or_force_answer_call_id": reflect["call_id"] if reflect else None,
                "reflect_or_force_answer_raw_response": reflect.get("raw_response") if reflect else None,
            }
        )

    parsed_steps = [value.get("parsed_step") for _, value in sorted(plan_by_round.items())]
    for index in range(1, len(parsed_steps)):
        if parsed_steps[index] is not None and parsed_steps[index] == parsed_steps[index - 1]:
            candidate_events.append(
                {
                    "kind": "exact_duplicate_observation_plan",
                    "round": index + 1,
                    "duplicate_of_round": index,
                }
            )

    for replay in replays:
        comparison = replay.get("comparison_to_full_evidence")
        if comparison and comparison.get("answer_text_changed"):
            candidate_events.append(
                {
                    "kind": "answer_text_changed_under_evidence_subset",
                    "replay_id": replay["replay_id"],
                    "evidence_rounds": replay["evidence_rounds"],
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "interpretation": {
            "verdict": None,
            "rule": "Candidate events are mechanical diagnostics, not pollution labels.",
            "not_establishable_is_error": False,
        },
        "sample": {
            "video_id": sample["video_id"],
            "question_id": sample["question_id"],
            "question": sample["question"],
            "video_path": str(video_path),
            "duration_sec": controller.bb.duration_sec,
            "inference_fields": sorted(set(sample)),
            "options_loaded": bool(sample.get("options")),
            "ground_truth_loaded": False,
        },
        "rounds": rounds,
        "media_creations": recorder.media_creations,
        "native_final": native_result["final"],
        "counterfactual_replays": replays,
        "normalization_events": normalization_events,
        "candidate_events": candidate_events,
        "known_limits": [
            "Model-provided reasoning is observable output, not hidden chain-of-thought.",
            "The vLLM endpoint does not expose the final selected frame PTS; that lineage remains unknown.",
            "Timestamp range checks establish request-range consistency only, not visual truth.",
            "Each ForceAnswer evidence subset is sampled once; differences are sensitivity candidates, not stochasticity-controlled causal effects.",
            "No answer, solution, or ground truth was loaded during native inference or replay.",
            "Semantic groundedness requires a later blinded review: SUPPORTED, CONTRADICTED, NOT_ESTABLISHABLE, or NON_FACTUAL.",
        ],
    }


def reclassify_saved_chain(sample_dir: Path) -> dict[str, Any]:
    """Apply the v0 full-second rule to saved derived analysis, without model calls."""
    chain_path = sample_dir / "observable_chain.json"
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    normalization_events = list(chain.get("normalization_events") or [])
    retained: list[dict[str, Any]] = []
    for event in chain.get("candidate_events") or []:
        if event.get("kind") == "parser_or_normalizer_changed_key_evidence":
            normalized = dict(event)
            normalized["kind"] = "avp_full_second_normalization"
            normalization_events.append(normalized)
            continue
        if event.get("kind") == "evidence_timestamp_outside_requested_ranges":
            start = event.get("timestamp_start")
            end = event.get("timestamp_end")
            if (
                isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and _within_any_range(float(start), float(end), event.get("requested_ranges") or [])
            ):
                continue
        retained.append(event)
    chain["normalization_events"] = normalization_events
    chain["candidate_events"] = retained
    write_json(chain_path, chain)

    status_path = sample_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_event_count"] = len(retained)
    write_json(status_path, status)

    manifest_path = sample_dir.parent.parent / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("samples") or []:
            if item.get("question_id") == chain["sample"]["question_id"]:
                item["candidate_event_count"] = len(retained)
        write_json(manifest_path, manifest)
    return chain


def intervention_schedule(repetitions: int) -> list[dict[str, Any]]:
    """A fixed Latin-square order for full/drop/corrected ForceAnswer controls."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    conditions = ["full_original_evidence", "drop_contradicted_item", "grounded_correction"]
    schedule: list[dict[str, Any]] = []
    for replicate in range(1, repetitions + 1):
        offset = (replicate - 1) % len(conditions)
        for position in range(len(conditions)):
            condition = conditions[(offset + position) % len(conditions)]
            schedule.append(
                {
                    "replicate": replicate,
                    "position_in_replicate": position + 1,
                    "condition": condition,
                }
            )
    return schedule


def _load_saved_evidence(path: Path) -> Evidence:
    raw = json.loads(path.read_text(encoding="utf-8"))
    field_names = {field.name for field in dataclasses.fields(Evidence)}
    values = {name: value for name, value in raw.items() if name in field_names}
    return Evidence(**values)


def _summarize_intervention(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_condition.setdefault(result["condition"], []).append(result)
    summary: dict[str, Any] = {"conditions": {}}
    stable_text_by_condition: dict[str, Optional[str]] = {}
    for condition, items in by_condition.items():
        completed = [item for item in items if item.get("status") == "completed"]
        answer_texts = [item["answer"]["normalized_answer_text"] for item in completed]
        confidences = [
            item["answer"]["confidence"]
            for item in completed
            if item["answer"]["confidence"] is not None
        ]
        stable = len(completed) == len(items) and len(set(answer_texts)) == 1
        confidence_stable = (
            len(completed) == len(items)
            and len(confidences) == len(items)
            and len(set(confidences)) == 1
        )
        stable_text_by_condition[condition] = answer_texts[0] if stable and answer_texts else None
        summary["conditions"][condition] = {
            "requested_repetitions": len(items),
            "completed_repetitions": len(completed),
            "exact_answer_text_stable": stable,
            "confidence_stable": confidence_stable,
            "normalized_answer_texts": answer_texts,
            "confidence_values": confidences,
            "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
        }

    full = summary["conditions"].get("full_original_evidence") or {}
    full_text = stable_text_by_condition.get("full_original_evidence")
    comparisons: dict[str, Any] = {}
    for condition in ("drop_contradicted_item", "grounded_correction"):
        condition_summary = summary["conditions"].get(condition) or {}
        condition_text = stable_text_by_condition.get(condition)
        full_confidence_stable = bool(full.get("confidence_stable"))
        condition_confidence_stable = bool(condition_summary.get("confidence_stable"))
        if not full_confidence_stable or not condition_confidence_stable:
            comparisons[condition] = {
                "status": "INCONCLUSIVE_CONFIDENCE_STOCHASTIC_OR_FAILED",
                "answer_text_changed": None,
                "confidence_delta": None,
            }
            continue
        full_confidence = full.get("mean_confidence")
        condition_confidence = condition_summary.get("mean_confidence")
        comparisons[condition] = {
            "status": "STABLE_CONFIDENCE_SENSITIVITY",
            "answer_text_changed": (
                condition_text != full_text
                if condition_text is not None and full_text is not None
                else None
            ),
            "answer_text_comparison_status": (
                "EXACT_TEXT_STABLE"
                if condition_text is not None and full_text is not None
                else "REQUIRES_SEPARATE_SEMANTIC_REVIEW"
            ),
            "confidence_delta": round(
                condition_confidence - full_confidence
                if condition_confidence is not None and full_confidence is not None
                else 0.0,
                6,
            ),
        }
    summary["comparisons_to_full_original_evidence"] = comparisons
    summary["pollution_verdict"] = None
    summary["interpretation"] = (
        "Stable changes establish ForceAnswer sensitivity to the evidence intervention only. "
        "They do not establish that the answer became correct; options and ground truth remain sealed."
    )
    return summary


def resummarize_saved_intervention(output_dir: Path) -> dict[str, Any]:
    """Recompute derived intervention statistics without any model request."""
    output_dir = output_dir.resolve(strict=True)
    results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = _summarize_intervention(results)
    write_json(output_dir / "summary.json", summary)
    return summary


def run_labeled_intervention(
    sample_dir: Path,
    config_path: Path,
    output_dir: Path,
    repetitions: int = 3,
) -> int:
    """Run sealed, text-only ForceAnswer controls after a blinded evidence label."""
    sample_dir = sample_dir.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Intervention output already exists: {output_dir}")

    qa_input = json.loads((sample_dir / "qa_only_input.json").read_text(encoding="utf-8"))
    labels_path = sample_dir / "blind_review" / "evidence_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    review_inputs = labels.get("review_inputs") or {}
    if review_inputs.get("options_visible") or review_inputs.get("ground_truth_visible"):
        raise ValueError("Intervention requires a blind evidence label with sealed options and ground truth")
    contradicted = [
        item
        for item in labels.get("evidence_reviews") or []
        if item.get("overall_label") == "CONTRADICTED"
    ]
    if len(contradicted) != 1:
        raise ValueError("This minimal intervention requires exactly one CONTRADICTED evidence item")
    correction = contradicted[0].get("grounded_correction_for_intervention") or {}
    correction_text = str(correction.get("detailed_response", "")).strip()
    if not correction_text or correction.get("key_evidence") not in ([], None):
        raise ValueError("A text-only grounded correction with empty key_evidence is required")

    original = _load_saved_evidence(sample_dir / "native" / "evidence" / "round_1" / "evidence.json")
    corrected = Evidence(
        detailed_response=correction_text,
        key_evidence=[],
        reasoning=correction_text,
        frames_used=[],
        model_call={
            "producer": "blind_review_grounded_correction",
            "source_locator": correction.get("source_locator"),
        },
        timestamp=utc_now(),
        round_id=original.round_id,
    )
    conditions = {
        "full_original_evidence": [original],
        "drop_contradicted_item": [],
        "grounded_correction": [corrected],
    }
    schedule = intervention_schedule(repetitions)
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "avp_pollution_intervention_v0",
        "status": "running",
        "started_at": utc_now(),
        "sample_dir": str(sample_dir),
        "question_id": qa_input["question_id"],
        "question": qa_input["question"],
        "options_or_ground_truth_loaded": False,
        "method": "Original Reflector.reflect(..., is_last_round=True, options=None)",
        "conditions": {
            "full_original_evidence": "Unchanged saved AVP Evidence object.",
            "drop_contradicted_item": "No evidence supplied; same ForceAnswer method.",
            "grounded_correction": correction,
        },
        "schedule": schedule,
        "request_budget": {
            "planned_text_only_requests": len(schedule),
            "automatic_retry": False,
            "cache_reuse_expected": False,
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    native_meta = json.loads((sample_dir / "native" / "meta.json").read_text(encoding="utf-8"))
    plan = plan_from_dict(json.loads((sample_dir / "native" / "plan.initial.json").read_text(encoding="utf-8")))
    client = make_client(config_path)
    recorder = ObservableTraceRecorder(output_dir / "trace.jsonl")
    results: list[dict[str, Any]] = []
    try:
        recorder.attach(client)
        for item in schedule:
            result = dict(item)
            result["status"] = "running"
            started = time.monotonic()
            replay_id = f"{item['condition']}__replicate_{item['replicate']}"
            try:
                with recorder.replay_phase(replay_id, [e.round_id for e in conditions[item["condition"]]]):
                    reflection = Reflector(client).reflect(
                        query=qa_input["question"],
                        plan=plan,
                        evidence_list=conditions[item["condition"]],
                        interaction_history=[],
                        video_path=qa_input["resolved_video_path"],
                        duration_sec=float(native_meta["duration_sec"]),
                        is_last_round=True,
                        options=None,
                    )
                result["reflection"] = reflection
                result["answer"] = _answer_view(reflection.get("final_answer"))
                result["status"] = "completed"
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["elapsed_sec"] = round(time.monotonic() - started, 6)
            results.append(result)
            append_jsonl(output_dir / "results.jsonl", result)
    finally:
        recorder.detach()
        client.cleanup_uploaded_files()

    summary = _summarize_intervention(results)
    write_json(output_dir / "summary.json", summary)
    manifest["status"] = "completed" if all(item["status"] == "completed" for item in results) else "completed_with_failures"
    manifest["finished_at"] = utc_now()
    manifest["actual_request_count"] = len(recorder.records)
    write_json(output_dir / "manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 1


def make_client(
    config_path: Path,
    *,
    temperature: float = 0.0,
    seed: int = 20260829,
) -> QwenClient:
    cfg = load_config(str(config_path))
    if cfg.backend == "qwen":
        client_type = QwenClient
    elif cfg.backend == "internvl":
        client_type = InternVLClient
    else:
        raise ValueError(
            "Pollution audit currently requires a local OpenAI-compatible transport"
        )
    client_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "plan_replan_model": cfg.get_plan_replan_model(),
        "execute_model": cfg.get_execute_model(),
        "project": cfg.project,
        "location": cfg.get_random_location(),
        "api_key": (cfg.api_key or "").strip() or None,
        "max_frame_low": cfg.max_frame_low,
        "max_frame_medium": cfg.max_frame_medium,
        "max_frame_high": cfg.max_frame_high,
        "confidence_threshold": cfg.confidence_threshold,
        "debug": cfg.debug,
        "base_url": cfg.base_url,
    }
    if cfg.backend == "qwen":
        client_kwargs["generation_overrides"] = {
            "temperature": float(temperature),
            "seed": int(seed),
        }
    client = client_type(
        **client_kwargs,
    )
    client.initialize_client()
    return client


def run_sample(
    sample: dict[str, Any],
    video_path: Path,
    sample_dir: Path,
    config_path: Path,
    max_rounds: int,
    native_only: bool = False,
    admission_mode: str = "off",
    temperature: float = 0.0,
    seed: int = 20260829,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=False)
    options = sample.get("options") or None
    query = format_mcq_query(sample["question"], options)
    source_media_probe = probe_media(video_path)
    derived_duration_sec = media_duration_sec(source_media_probe)
    metadata_source = sample_dir / "qa_only_video_metadata.json"
    write_qa_only_metadata_source(
        metadata_source,
        video_path=video_path,
        video_id=sample["video_id"],
        duration_sec=derived_duration_sec,
    )
    set_metadata_source(str(metadata_source))
    write_json(
        sample_dir / "qa_only_input.json",
        {
            "video_id": sample["video_id"],
            "question_id": sample["question_id"],
            "question": sample["question"],
            "options": options,
            "input_duration": sample.get("duration"),
            "derived_duration_sec": derived_duration_sec,
            "resolved_video_path": str(video_path),
            "source_media_probe": source_media_probe,
        },
    )
    client = make_client(
        config_path,
        temperature=temperature,
        seed=seed,
    )
    admission_runtime: Optional[RepairController] = None
    audit_backend: Optional[OpenAIJSONBackend] = None
    if admission_mode == "repair":
        cfg = load_config(str(config_path))
        audit_backend = OpenAIJSONBackend(
            base_url=cfg.base_url,
            model=cfg.get_plan_replan_model(),
            trace_path=sample_dir / "native" / "semantic_delta" / "model_trace.jsonl",
            temperature=temperature,
            seed=seed,
            max_requests=5,
        )
        admission_runtime = RepairController(
            run_dir=sample_dir / "native",
            backend=audit_backend,
            raw_qwen_only=True,
        )
    recorder = ObservableTraceRecorder(sample_dir / "trace.jsonl")
    controller: Optional[Controller] = None
    native_result: Optional[dict[str, Any]] = None
    replays: list[dict[str, Any]] = []
    status: dict[str, Any] = {"status": "running", "started_at": utc_now()}
    write_json(sample_dir / "status.json", status)
    try:
        recorder.attach(client)
        controller = Controller(
            run_dir=str(sample_dir / "native"),
            video_path=str(video_path),
            client=client,
            options=options,
            admission_runtime=admission_runtime,
        )
        if not controller.bb.duration_sec or abs(controller.bb.duration_sec - derived_duration_sec) > 0.05:
            raise RuntimeError(
                "Controller duration disagrees with the clean run-local metadata source: "
                f"controller={controller.bb.duration_sec}, derived={derived_duration_sec}"
            )
        native_result = controller.run(query=query, max_rounds=max_rounds)
        if not native_only:
            replays = run_counterfactual_replays(
                recorder=recorder,
                controller=controller,
                native_result=native_result,
                query=query,
                options=options,
            )
            write_json(sample_dir / "counterfactual_replays.json", replays)
        chain = build_observable_chain(
            sample=sample,
            video_path=video_path,
            controller=controller,
            recorder=recorder,
            native_result=native_result,
            replays=replays,
        )
        write_json(sample_dir / "observable_chain.json", chain)
        native_request_count = len(
            [record for record in recorder.records if record["mode"] == "native"]
        )
        replay_request_count = len(
            [
                record
                for record in recorder.records
                if record["mode"] == "counterfactual_replay"
            ]
        )
        admission_request_count = (
            audit_backend.request_count if audit_backend is not None else 0
        )
        total_model_request_count = (
            native_request_count + replay_request_count + admission_request_count
        )
        native_max = 3 * (
            max_rounds + (1 if admission_mode == "repair" else 0)
        )
        replay_max = 0 if native_only else 2 * max_rounds - 1
        admission_max = 5 if admission_mode == "repair" else 0
        if native_request_count > native_max:
            raise RuntimeError("avp_native_request_budget_exceeded")
        if replay_request_count > replay_max:
            raise RuntimeError("counterfactual_replay_request_budget_exceeded")
        if admission_request_count > admission_max:
            raise RuntimeError("admission_request_budget_exceeded")
        if (
            admission_runtime is not None
            and admission_runtime.repair_observations > 1
        ):
            raise RuntimeError("repair_observation_budget_exceeded")
        if total_model_request_count > native_max + replay_max + admission_max:
            raise RuntimeError("total_model_request_budget_exceeded")
        status.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "native_round_count": len(controller.bb.evidences),
                "native_request_count": native_request_count,
                "replay_request_count": replay_request_count,
                "admission_request_count": admission_request_count,
                "total_model_request_count": total_model_request_count,
                "repair_observation_count": (
                    admission_runtime.repair_observations
                    if admission_runtime is not None
                    else 0
                ),
                "admission_summary": (
                    admission_runtime.summary()
                    if admission_runtime is not None
                    else None
                ),
                "candidate_event_count": len(chain["candidate_events"]),
            }
        )
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "automatic_retry": False,
            }
        )
    finally:
        native_request_count = len(
            [record for record in recorder.records if record["mode"] == "native"]
        )
        replay_request_count = len(
            [
                record
                for record in recorder.records
                if record["mode"] == "counterfactual_replay"
            ]
        )
        admission_request_count = (
            audit_backend.request_count if audit_backend is not None else 0
        )
        status.setdefault("native_request_count", native_request_count)
        status.setdefault("replay_request_count", replay_request_count)
        status.setdefault("admission_request_count", admission_request_count)
        status.setdefault(
            "total_model_request_count",
            native_request_count + replay_request_count + admission_request_count,
        )
        status.setdefault(
            "repair_observation_count",
            admission_runtime.repair_observations
            if admission_runtime is not None
            else 0,
        )
        status.setdefault(
            "admission_summary",
            admission_runtime.summary() if admission_runtime is not None else None,
        )
        recorder.detach()
        try:
            client.cleanup_uploaded_files()
        except Exception as exc:
            status["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(sample_dir / "status.json", status)
    return status


def run_experiment(args: argparse.Namespace) -> int:
    questions_path = Path(args.questions).resolve(strict=True)
    video_root = Path(args.video_root).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Output directory already exists; refusing to resume or overwrite: {out}")
    rows = load_qa_only(questions_path)
    selected = select_questions(rows, args.question_id)
    options_presence = [bool(row.get("options")) for row in selected]
    if args.require_question_only and any(options_presence):
        raise ValueError("Question-only run rejected a selected row containing options")
    if any(options_presence) and not all(options_presence):
        raise ValueError("A run must not mix MCQ rows with question-only rows")
    options_loaded = all(options_presence)
    cfg = load_config(str(config_path))
    if args.admission_mode == "repair" and cfg.backend != "qwen":
        raise ValueError("Raw admission acceptance requires the Qwen transport receipt")
    out.mkdir(parents=True)
    replay_max = 0 if args.native_only else 2 * args.max_rounds - 1
    native_round_max = args.max_rounds + (
        1 if args.admission_mode == "repair" else 0
    )
    admission_max = 5 if args.admission_mode == "repair" else 0
    request_budget = {
        "per_sample_native_max": 3 * native_round_max,
        "per_sample_total_max": 3 * native_round_max + replay_max,
        "per_sample_avp_native_max": 3 * native_round_max,
        "per_sample_replay_max": replay_max,
        "per_sample_admission_max": admission_max,
        "per_sample_total_model_max": (
            3 * native_round_max + replay_max + admission_max
        ),
        "automatic_retry": False,
        "cache_reuse_expected": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "questions_path": str(questions_path),
        "video_root": str(video_root),
        "config_path": str(config_path),
        "selected_question_ids": args.question_id,
        "pair_id": args.pair_id,
        "max_rounds": args.max_rounds,
        "repair_extension_round_max": (
            1 if args.admission_mode == "repair" else 0
        ),
        "native_only": bool(args.native_only),
        "question_only_required": bool(args.require_question_only),
        "admission_mode": args.admission_mode,
        "sampling": {"temperature": args.temperature, "seed": args.seed},
        "model_config": {
            "backend": cfg.backend,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "plan_replan_model": cfg.get_plan_replan_model(),
            "execute_model": cfg.get_execute_model(),
        },
        "admission_model_config": (
            {
                "enable_thinking": False,
                "max_tokens": 1024,
                "max_requests": admission_max,
                "automatic_retry": False,
            }
            if args.admission_mode == "repair" else None
        ),
        "request_budget": request_budget,
        "model_inputs": (
            "AVP native prompts, question text, official answer options, and requested "
            "video/derived clips; no ground truth."
            if options_loaded
            else "AVP native prompts, question text, and requested video/derived clips; no ground truth."
        ),
        "options_loaded": options_loaded,
        "ground_truth_loaded": False,
        "gold_span_loaded": False,
        "raw_fidelity_boundary": (
            "The transport receipt proves that vLLM fetched the submitted media bytes over "
            "local HTTP and returned a response. AVP records the declared raw source range "
            "and submitted-media lineage, but selected frame PTS remain unknown. Admission "
            "judges semantic deltas against Observer text and does not independently prove "
            "that the Observer text is visually faithful."
        ),
        "samples": [],
    }
    write_json(out / "run_manifest.json", manifest)
    any_failed = False
    for sample in selected:
        video_path = resolve_video(video_root, sample["video_id"])
        sample_dir = out / "samples" / f"{sample['video_id']}__{sample['question_id']}"
        status = run_sample(
            sample,
            video_path,
            sample_dir,
            config_path,
            args.max_rounds,
            native_only=args.native_only,
            admission_mode=args.admission_mode,
            temperature=args.temperature,
            seed=args.seed,
        )
        manifest["samples"].append(
            {
                "video_id": sample["video_id"],
                "question_id": sample["question_id"],
                "question": sample["question"],
                "sample_dir": str(sample_dir),
                **status,
            }
        )
        write_json(out / "run_manifest.json", manifest)
        any_failed = any_failed or status["status"] != "completed"
    manifest["status"] = "completed_with_failures" if any_failed else "completed"
    manifest["finished_at"] = utc_now()
    manifest["ground_truth_loaded"] = False
    manifest["gold_span_loaded"] = False
    manifest["actual_avp_native_requests"] = sum(
        int(sample.get("native_request_count", 0)) for sample in manifest["samples"]
    )
    manifest["actual_replay_requests"] = sum(
        int(sample.get("replay_request_count", 0)) for sample in manifest["samples"]
    )
    manifest["actual_admission_requests"] = sum(
        int(sample.get("admission_request_count", 0))
        for sample in manifest["samples"]
    )
    manifest["actual_model_requests"] = sum(
        int(sample.get("total_model_request_count", 0))
        for sample in manifest["samples"]
    )
    write_json(out / "run_manifest.json", manifest)
    return 1 if any_failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", help="Inference JSON or JSONL; options are optional, GT fields are rejected")
    parser.add_argument("--video-root", help="Directory containing <video_id>.mp4")
    parser.add_argument("--config", required=True, help="AVP Qwen config JSON")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--question-id", action="append", help="Question ID; repeat to run a fixed cohort")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--admission-mode",
        choices=("off", "repair"),
        default="off",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--pair-id")
    parser.add_argument(
        "--require-question-only",
        action="store_true",
        help="Reject any selected inference row containing answer options",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Run unchanged AVP inference and record it; do not issue any replay or intervention request",
    )
    parser.add_argument("--intervention-sample-dir", help="Completed blind-reviewed sample directory")
    parser.add_argument("--intervention-repetitions", type=int, default=3)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.intervention_sample_dir:
        if args.admission_mode != "off" or args.require_question_only:
            raise ValueError(
                "Intervention mode cannot enable admission or question-only acceptance"
            )
        if args.questions or args.video_root or args.question_id:
            raise ValueError("Intervention mode cannot accept baseline question selection arguments")
        return run_labeled_intervention(
            sample_dir=Path(args.intervention_sample_dir),
            config_path=Path(args.config),
            output_dir=Path(args.out),
            repetitions=args.intervention_repetitions,
        )
    if not args.questions or not args.video_root or not args.question_id:
        raise ValueError("Baseline mode requires --questions, --video-root, and at least one --question-id")
    if args.max_rounds <= 0:
        raise ValueError("--max-rounds must be positive")
    if args.admission_mode == "repair" and not args.native_only:
        raise ValueError("Repair acceptance requires --native-only; replays are forbidden")
    if args.admission_mode == "repair" and not args.require_question_only:
        raise ValueError("Repair acceptance requires --require-question-only")
    if args.require_question_only and not str(args.pair_id or "").strip():
        raise ValueError("Question-only acceptance requires --pair-id")
    if args.pair_id and (not args.native_only or not args.require_question_only):
        raise ValueError(
            "Paired raw acceptance requires --native-only and --require-question-only"
        )
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
