"""Run an isolated query-driven frame-selection experiment.

The script builds one complete 1 fps OpenCLIP index per source video, uses one
text-only model call per question to propose visual retrieval queries, sends at
most 32 selected JPEG frames to one visual observation call, and gives only the
grounded observation facts to one text-only answer call. It never reads ground
truth and does not enter the AVP Controller, Evidence Audit, or Verifier.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avp.perception_tools import (
    DynamicPerceptionConfig,
    OpenClipBackend,
    build_global_perception,
)
from avp.perception_tools.query_driven_frames import (
    QueryFramePick,
    select_query_frames,
)
from experiments.pollution_audit import write_json
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    DashScopeFlashTransport,
)
from scripts.run_perception_shadow import (
    decode_scheduled_frames,
    encode_frame_samples,
    load_inference_input,
    sample_timestamps,
)


MODEL = "qwen3.8-flash"
MIN_PROVIDER_FRAMES = 4
MAX_EXPERIMENT_FRAMES = 32


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


QUERY_FORMAT = response_format(
    "query_driven_visual_queries_v1",
    {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {"type": "string", "minLength": 1},
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--global-fps", type=float, default=1.0)
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--frame-budget", type=int, default=32)
    parser.add_argument("--minimum-time-gap-sec", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.global_fps <= 0.0 or not math.isfinite(args.global_fps):
        raise ValueError("global_fps_must_be_finite_and_positive")
    if args.max_edge < 64:
        raise ValueError("max_edge_must_be_at_least_64")
    if args.embedding_batch_size < 1:
        raise ValueError("embedding_batch_size_must_be_positive")
    if not MIN_PROVIDER_FRAMES <= args.frame_budget <= MAX_EXPERIMENT_FRAMES:
        raise ValueError("frame_budget_must_be_between_4_and_32")
    if args.minimum_time_gap_sec < 0.0:
        raise ValueError("minimum_time_gap_sec_must_be_non_negative")


def parse_json_value(raw: str) -> Any:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def parse_json_object(raw: str) -> dict[str, Any]:
    parsed = parse_json_value(raw)
    if not isinstance(parsed, dict):
        raise ValueError("provider_response_must_be_a_json_object")
    return parsed


def call_json(
    transport: DashScopeFlashTransport,
    *,
    stage: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    enable_thinking: bool,
    temperature: float,
    seed: int,
    max_tokens: int,
    request_trace: list[dict[str, Any]],
    submitted_frame_count: int = 0,
    bare_array_key: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "request_index": len(request_trace) + 1,
        "stage": stage,
        "enable_thinking": enable_thinking,
        "submitted_frame_count": submitted_frame_count,
        "started_at": utc_now(),
    }
    request_trace.append(record)
    started = time.monotonic()
    try:
        raw = transport.post_chat_completion(
            {
                "model": MODEL,
                "messages": messages,
                "response_format": schema,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": enable_thinking,
            }
        )
        record["raw_response"] = raw
        if bare_array_key is None:
            parsed = parse_json_object(raw)
        else:
            parsed_value = parse_json_value(raw)
            if isinstance(parsed_value, dict):
                parsed = parsed_value
            elif isinstance(parsed_value, list):
                parsed = {bare_array_key: parsed_value}
                record["normalization_warnings"] = [
                    f"wrapped_bare_array_as_{bare_array_key}"
                ]
            else:
                raise ValueError("provider_response_must_be_a_json_object")
    except Exception as exc:
        record.update(
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "finished_at": utc_now(),
            }
        )
        raise
    record.update(
        {
            "status": "COMPLETED",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "finished_at": utc_now(),
        }
    )
    return parsed


def option_block(item: Mapping[str, Any]) -> str:
    return "\n".join(str(option) for option in item["options"])


def generate_queries(
    transport: DashScopeFlashTransport,
    item: Mapping[str, Any],
    args: argparse.Namespace,
    request_trace: list[dict[str, Any]],
) -> tuple[str, ...]:
    parsed = call_json(
        transport,
        stage="QUERY_GENERATION",
        messages=[
            {
                "role": "system",
                "content": (
                    "Create 2 to 4 short, concrete English visual-search phrases for "
                    "a complete 1 fps video-frame index. Each phrase must describe "
                    "visible content that would distinguish the answer options, such "
                    "as exact on-screen wording, objects, actions, or ordered list "
                    "items. Prefer distinct, instance-specific targets over generic "
                    "title screens. For questions asking how many kinds, instances, "
                    "or viewpoints occur, search for the individual visible instances "
                    "rather than treating the numeric answer choices as visible target "
                    "values; include plausible instances not literally named by the "
                    "choices when the question requires exhaustive counting. Do not "
                    "answer the question and do not invent timestamps."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item['question']}\n"
                    f"Official options:\n{option_block(item)}"
                ),
            },
        ],
        schema=QUERY_FORMAT,
        enable_thinking=False,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=512,
        request_trace=request_trace,
    )
    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list) or not 2 <= len(raw_queries) <= 4:
        raise ValueError("query_generation_returned_invalid_query_count")
    queries = tuple(str(value).strip() for value in raw_queries)
    if any(not value for value in queries) or len(set(queries)) != len(queries):
        raise ValueError("query_generation_returned_empty_or_duplicate_queries")
    return queries


def event_index_for_timestamp(perception_map: Any, timestamp_sec: float) -> int:
    for event in perception_map.events:
        if event.start_sec <= timestamp_sec < event.end_sec:
            return int(event.event_index)
    if perception_map.events and math.isclose(
        timestamp_sec, perception_map.events[-1].end_sec, abs_tol=1e-6
    ):
        return int(perception_map.events[-1].event_index)
    raise ValueError("sample_timestamp_outside_perception_events")


def encode_labeled_jpeg(image_rgb: np.ndarray, frame_id: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    source = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    font_size = max(18, min(34, source.width // 18))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    box = font.getbbox(frame_id)
    header_height = max(34, box[3] - box[1] + 12)
    labeled = Image.new("RGB", (source.width, source.height + header_height), "black")
    labeled.paste(source, (0, header_height))
    ImageDraw.Draw(labeled).text((8, 5), frame_id, fill="white", font=font)
    buffer = BytesIO()
    labeled.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def observation_format(frame_ids: Sequence[str]) -> dict[str, Any]:
    return response_format(
        "query_driven_visual_observations_v1",
        {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {"type": "string", "enum": list(frame_ids)},
                            },
                            "fact": {"type": "string", "minLength": 1},
                        },
                        "required": ["frame_ids", "fact"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    )


def observe_frames(
    transport: DashScopeFlashTransport,
    item: Mapping[str, Any],
    queries: Sequence[str],
    selected: Sequence[dict[str, Any]],
    jpegs: Sequence[bytes],
    args: argparse.Namespace,
    request_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame_ids = [str(frame["frame_id"]) for frame in selected]
    frame_uris = [
        "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        for jpeg in jpegs
    ]
    parsed = call_json(
        transport,
        stage="VISUAL_OBSERVATION",
        messages=[
            {
                "role": "system",
                "content": (
                    "Inspect the supplied frames as one chronological sequence. "
                    "Report only directly visible facts relevant to the question. "
                    "Preserve exact on-screen wording, numbers, list order, and object "
                    "or action identity when visible. Cite only frame IDs printed in "
                    "the images. Do not infer timestamps, fill unseen gaps, or answer."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Question: {item['question']}\n"
                            f"Official options:\n{option_block(item)}\n"
                            "Retrieval intents:\n- "
                            + "\n- ".join(queries)
                        ),
                    },
                    {"type": "video", "video": frame_uris, "fps": 1.0},
                ],
            },
        ],
        schema=observation_format(frame_ids),
        enable_thinking=False,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=2048,
        request_trace=request_trace,
        submitted_frame_count=len(jpegs),
    )
    facts = parsed.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("visual_observation_returned_no_facts")
    allowed = set(frame_ids)
    grounded: list[dict[str, Any]] = []
    frame_by_id = {str(frame["frame_id"]): frame for frame in selected}
    for index, fact in enumerate(facts, 1):
        if not isinstance(fact, dict):
            raise ValueError("visual_observation_fact_must_be_an_object")
        cited = fact.get("frame_ids")
        text = str(fact.get("fact") or "").strip()
        if (
            not isinstance(cited, list)
            or not cited
            or any(str(frame_id) not in allowed for frame_id in cited)
            or not text
        ):
            raise ValueError("visual_observation_fact_is_not_frame_grounded")
        locators = [
            {
                "frame_id": str(frame_id),
                "timestamp_sec": frame_by_id[str(frame_id)]["timestamp_sec"],
                "event_index": frame_by_id[str(frame_id)]["event_index"],
            }
            for frame_id in cited
        ]
        grounded.append(
            {
                "fact_id": f"O{index:03d}",
                "fact": text,
                "frame_locators": locators,
                "earliest_timestamp_sec": min(
                    float(locator["timestamp_sec"]) for locator in locators
                ),
            }
        )
    grounded.sort(key=lambda value: (value["earliest_timestamp_sec"], value["fact_id"]))
    return grounded


def answer_format(option_ids: Sequence[str], fact_ids: Sequence[str]) -> dict[str, Any]:
    return response_format(
        "query_driven_grounded_answer_v1",
        {
            "type": "object",
            "properties": {
                "selected_option_id": {"type": "string", "enum": list(option_ids)},
                "reason": {"type": "string", "minLength": 1},
                "fact_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": list(fact_ids)},
                },
            },
            "required": ["selected_option_id", "reason", "fact_ids"],
            "additionalProperties": False,
        },
    )


def answer_from_facts(
    transport: DashScopeFlashTransport,
    item: Mapping[str, Any],
    facts: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    request_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    option_ids = tuple(chr(ord("A") + index) for index in range(len(item["options"])))
    fact_ids = tuple(str(fact["fact_id"]) for fact in facts)
    parsed = call_json(
        transport,
        stage="ANSWER",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the multiple-choice question using only the ordered, "
                    "frame-grounded observations. You must select exactly one option. "
                    "Use the timestamps only to preserve ordering and binding; do not "
                    "invent missing visual facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item['question']}\n"
                    f"Official options:\n{option_block(item)}\n"
                    "Chronological observations:\n"
                    + json.dumps(list(facts), ensure_ascii=False, indent=2)
                ),
            },
        ],
        schema=answer_format(option_ids, fact_ids),
        enable_thinking=True,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=2048,
        request_trace=request_trace,
    )
    selected = str(parsed.get("selected_option_id") or "").strip().upper()
    if selected not in option_ids:
        raise ValueError("answer_returned_unknown_option")
    cited = parsed.get("fact_ids")
    if not isinstance(cited, list) or not cited or any(str(value) not in fact_ids for value in cited):
        raise ValueError("answer_returned_unknown_or_empty_fact_ids")
    return {
        "selected_option_id": selected,
        "selected_option_text": item["options"][option_ids.index(selected)],
        "reason": str(parsed.get("reason") or "").strip(),
        "fact_ids": [str(value) for value in cited],
    }


def run_question(
    transport: DashScopeFlashTransport,
    item: Mapping[str, Any],
    frames: Sequence[Any],
    perception_map: Any,
    backend: OpenClipBackend,
    args: argparse.Namespace,
) -> dict[str, Any]:
    request_trace: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        queries = generate_queries(transport, item, args, request_trace)
        query_embeddings = backend.encode_text(queries)
        frame_embeddings = np.stack([frame.embedding for frame in frames])
        event_indexes = [
            event_index_for_timestamp(perception_map, frame.timestamp_sec)
            for frame in frames
        ]
        picks = select_query_frames(
            timestamps=[frame.timestamp_sec for frame in frames],
            frame_embeddings=frame_embeddings,
            event_indexes=event_indexes,
            query_embeddings=query_embeddings,
            frame_budget=args.frame_budget,
            minimum_time_gap_sec=args.minimum_time_gap_sec,
        )
        if not MIN_PROVIDER_FRAMES <= len(picks) <= args.frame_budget:
            raise RuntimeError("selected_frame_count_outside_experiment_contract")
        selected: list[dict[str, Any]] = []
        jpegs: list[bytes] = []
        for chronological_index, pick in enumerate(picks, 1):
            frame_id = f"F{chronological_index:04d}"
            record = pick.to_dict()
            record.update(
                {
                    "frame_id": frame_id,
                    "query": queries[pick.query_index],
                }
            )
            selected.append(record)
            jpegs.append(encode_labeled_jpeg(frames[pick.frame_index].image_rgb, frame_id))
        facts = observe_frames(
            transport, item, queries, selected, jpegs, args, request_trace
        )
        answer = answer_from_facts(transport, item, facts, args, request_trace)
    except Exception as exc:
        return {
            "question_id": item["question_id"],
            "video_id": item["video_id"],
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "request_count": len(request_trace),
            "requests": request_trace,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "ground_truth_loaded": False,
        }
    return {
        "question_id": item["question_id"],
        "video_id": item["video_id"],
        "status": "COMPLETED",
        "question": item["question"],
        "options": list(item["options"]),
        "queries": list(queries),
        "retrieval": {
            "scored_frame_count": len(frames),
            "scored_event_count": len(perception_map.events),
            "fixed_top_k_events": False,
            "frame_budget": args.frame_budget,
            "selected_frame_count": len(selected),
            "selected_frames": selected,
        },
        "observations": facts,
        "answer": answer,
        "request_count": len(request_trace),
        "requests": request_trace,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "ground_truth_loaded": False,
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL_required")
    if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        raise RuntimeError("DASHSCOPE_API_KEY_required")
    output_root = Path(args.out).resolve()
    if output_root.exists():
        raise FileExistsError(f"output_already_exists:{output_root}")
    output_root.mkdir(parents=True)
    inputs = [load_inference_input(path) for path in args.input_json]
    if len({item["question_id"] for item in inputs}) != len(inputs):
        raise ValueError("question_ids_must_be_unique")
    video_keys = {
        (item["video_id"], item["video_path"], item["duration_sec"])
        for item in inputs
    }
    if len(video_keys) != 1:
        raise ValueError("this_reuse_experiment_requires_one_shared_video")

    started_at = utc_now()
    started = time.monotonic()
    item = inputs[0]
    backend = OpenClipBackend(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    timestamps = sample_timestamps(0.0, item["duration_sec"], args.global_fps)
    images, decoder = decode_scheduled_frames(
        item["video_path"], timestamps, max_edge=args.max_edge
    )
    frames = encode_frame_samples(
        backend,
        timestamps,
        images,
        batch_size=args.embedding_batch_size,
    )
    perception_map = build_global_perception(
        item["duration_sec"],
        frames,
        DynamicPerceptionConfig(global_sample_interval_sec=1.0 / args.global_fps),
    )
    if not perception_map.coverage_complete_at_declared_cadence:
        raise RuntimeError(
            "event_map_incomplete:" + ",".join(perception_map.coverage_violations)
        )
    write_json(
        output_root / "perception_map.json",
        {
            "video_id": item["video_id"],
            "video_path": item["video_path"],
            "duration_sec": item["duration_sec"],
            "ground_truth_loaded": False,
            "decoder": decoder,
            "perception_map": perception_map.to_dict(),
        },
    )

    transport = DashScopeFlashTransport(
        base_url=base_url,
        generation_overrides={},
    )
    samples: list[dict[str, Any]] = []
    try:
        for question in inputs:
            sample = run_question(
                transport, question, frames, perception_map, backend, args
            )
            question_dir = output_root / "questions" / question["question_id"]
            write_json(question_dir / "result.json", sample)
            samples.append(sample)
    finally:
        transport.close()

    status = "COMPLETED" if all(sample["status"] == "COMPLETED" for sample in samples) else "COMPLETED_WITH_FAILURES"
    manifest = {
        "schema_version": "query_driven_frame_shadow_v1",
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": MODEL,
        "ground_truth_loaded": False,
        "options_visible": True,
        "cache_reuse_expected": False,
        "automatic_retry": False,
        "shared_video_index": True,
        "index": {
            "sampling_fps": args.global_fps,
            "sample_count": len(frames),
            "event_count": len(perception_map.events),
            "coverage_complete": True,
            "max_edge": args.max_edge,
        },
        "visual_frame_budget_per_question": args.frame_budget,
        "maximum_provider_requests_per_question": 3,
        "request_count": sum(sample["request_count"] for sample in samples),
        "samples": [
            {
                "question_id": sample["question_id"],
                "status": sample["status"],
                "selected_option_id": (sample.get("answer") or {}).get("selected_option_id"),
                "result": str(
                    output_root / "questions" / sample["question_id"] / "result.json"
                ),
            }
            for sample in samples
        ],
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
