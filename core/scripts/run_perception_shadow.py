from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np


def _load_isolated_perception_tools() -> Any:
    """Load the extracted package without importing AVP's eager root package."""

    package_dir = Path(__file__).resolve().parents[1] / "avp" / "perception_tools"
    package_name = "_avp_perception_tools_shadow"
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load isolated perception_tools package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


_perception_tools = _load_isolated_perception_tools()
DynamicPerceptionConfig = _perception_tools.DynamicPerceptionConfig
FrameSample = _perception_tools.FrameSample
OpenClipBackend = _perception_tools.OpenClipBackend
build_global_perception = _perception_tools.build_global_perception
rank_events = _perception_tools.rank_events
refine_interval = _perception_tools.refine_interval
select_ranked_event_batch = _perception_tools.select_ranked_event_batch


FORBIDDEN_GROUND_TRUTH_KEYS = {
    "answer",
    "correct_answer",
    "ground_truth",
    "gt",
    "label",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_timestamps(start_sec: float, end_sec: float, fps: float) -> tuple[float, ...]:
    start = float(start_sec)
    end = float(end_sec)
    rate = float(fps)
    if not math.isfinite(start) or not math.isfinite(end) or not 0.0 <= start < end:
        raise ValueError("sample range must satisfy 0 <= start_sec < end_sec")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("fps must be finite and positive")
    interval = 1.0 / rate
    count = max(1, int(math.ceil((end - start) / interval)))
    values = tuple(
        start + interval * (index + 0.5)
        for index in range(count)
        if start + interval * (index + 0.5) < end
    )
    return values or ((start + end) / 2.0,)


def load_inference_input(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("inference input must be a JSON object")
    if payload.get("ground_truth_loaded") is not False:
        raise ValueError("ground_truth_loaded must be explicitly false")
    leaked = sorted(key for key in FORBIDDEN_GROUND_TRUTH_KEYS if key in payload)
    if leaked:
        raise ValueError("ground-truth fields are forbidden: " + ", ".join(leaked))
    required = (
        "question_id",
        "video_id",
        "question",
        "options",
        "video_path",
        "duration_sec",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("missing inference fields: " + ", ".join(missing))
    if not isinstance(payload["options"], list) or not payload["options"]:
        raise ValueError("options must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in payload["options"]):
        raise ValueError("every option must be a non-empty string")
    duration = float(payload["duration_sec"])
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    video_path = Path(str(payload["video_path"]))
    if not video_path.is_file():
        raise ValueError(f"video_path does not exist: {video_path}")
    return {
        "source_path": str(source.resolve()),
        "question_id": str(payload["question_id"]),
        "video_id": str(payload["video_id"]),
        "question": str(payload["question"]),
        "options": [str(item) for item in payload["options"]],
        "video_path": str(video_path.resolve()),
        "duration_sec": duration,
        "ground_truth_loaded": False,
    }


def query_text(item: dict[str, Any]) -> str:
    options = "\n".join(str(option) for option in item["options"])
    return f"Question: {item['question']}\nOptions:\n{options}"


def _resize_rgb(image_rgb: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale >= 1.0:
        return np.ascontiguousarray(image_rgb)
    resized = cv2.resize(
        image_rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def decode_scheduled_frames(
    video_path: str | Path,
    timestamps: Sequence[float],
    *,
    max_edge: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    schedule = tuple(float(value) for value in timestamps)
    if not schedule or tuple(sorted(schedule)) != schedule:
        raise ValueError("timestamps must be a non-empty ordered sequence")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise RuntimeError("video reports an invalid frame rate")
    seek_start = max(0.0, schedule[0] - 1.0)
    capture.set(cv2.CAP_PROP_POS_MSEC, seek_start * 1000.0)
    selected: list[np.ndarray] = []
    decoded = 0
    target_index = 0
    try:
        while target_index < len(schedule):
            ok, image_bgr = capture.read()
            if not ok:
                break
            decoded += 1
            absolute_frame = max(0.0, float(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1.0)
            actual_sec = absolute_frame / fps
            if actual_sec + (0.5 / fps) < schedule[target_index]:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = _resize_rgb(image_rgb, max_edge)
            while target_index < len(schedule) and actual_sec + (0.5 / fps) >= schedule[target_index]:
                selected.append(image_rgb.copy())
                target_index += 1
    finally:
        capture.release()
    if target_index != len(schedule):
        raise RuntimeError(
            f"decoder stopped before scheduled timestamp {schedule[target_index]:.6f}s"
        )
    return selected, {
        "decoder": "opencv_sequential_seek",
        "source_fps": fps,
        "scheduled_frames": len(schedule),
        "decoded_source_frames": decoded,
        "max_edge": int(max_edge),
    }


def encode_frame_samples(
    backend: OpenClipBackend,
    timestamps: Sequence[float],
    images: Sequence[np.ndarray],
    *,
    batch_size: int,
) -> tuple[FrameSample, ...]:
    if len(timestamps) != len(images):
        raise ValueError("timestamps and images must have the same length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batches.append(backend.encode_images(images[start : start + batch_size]))
    embeddings = np.concatenate(batches, axis=0)
    return tuple(
        FrameSample(timestamp_sec=float(timestamp), image_rgb=image, embedding=embedding)
        for timestamp, image, embedding in zip(timestamps, images, embeddings)
    )


def _nearest_sample(frames: Sequence[FrameSample], timestamp: float) -> FrameSample:
    return min(frames, key=lambda frame: abs(frame.timestamp_sec - float(timestamp)))


def _tile(image_rgb: np.ndarray, label: str, *, width: int, height: int) -> Image.Image:
    from PIL import Image, ImageDraw

    source = Image.fromarray(image_rgb, mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    source.thumbnail((width, height), resampling.LANCZOS)
    tile = Image.new("RGB", (width, height + 42), "white")
    left = (width - source.width) // 2
    top = (height - source.height) // 2
    tile.paste(source, (left, top))
    draw = ImageDraw.Draw(tile)
    draw.text((5, height + 4), label[:84], fill="black")
    return tile


def write_contact_sheet(
    path: Path,
    question_id: str,
    tiles: Sequence[tuple[np.ndarray, str]],
) -> None:
    from PIL import Image, ImageDraw

    tile_width = 320
    tile_height = 180
    columns = 4
    rows = max(1, int(math.ceil(len(tiles) / columns)))
    header_height = 34
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (tile_height + 42)),
        "white",
    )
    ImageDraw.Draw(sheet).text(
        (8, 8), f"Perception shadow contact sheet: {question_id}", fill="black"
    )
    for index, (image_rgb, label) in enumerate(tiles):
        column = index % columns
        row = index // columns
        sheet.paste(
            _tile(image_rgb, label, width=tile_width, height=tile_height),
            (column * tile_width, header_height + row * (tile_height + 42)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=90)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def gpu_memory_stats(backend: OpenClipBackend) -> dict[str, int] | None:
    torch = getattr(backend, "_torch", None)
    device = getattr(backend, "device", None)
    if torch is None or not isinstance(device, str) or not device.startswith("cuda"):
        return None
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_shadow(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output_root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = [load_inference_input(path) for path in args.input_json]
    config = DynamicPerceptionConfig(
        global_sample_interval_sec=1.0 / args.global_fps,
        local_sample_interval_sec=1.0 / args.local_fps,
    )
    backend = OpenClipBackend(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for item in inputs:
        key = (item["video_id"], item["video_path"], item["duration_sec"])
        groups.setdefault(key, []).append(item)

    started_at = utc_now()
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    for (video_id, video_path, duration_sec), video_items in groups.items():
        video_started = time.monotonic()
        global_times = sample_timestamps(0.0, duration_sec, args.global_fps)
        global_images, global_decode = decode_scheduled_frames(
            video_path, global_times, max_edge=args.max_edge
        )
        global_frames = encode_frame_samples(
            backend,
            global_times,
            global_images,
            batch_size=args.batch_size,
        )
        perception_map = build_global_perception(duration_sec, global_frames, config)
        video_dir = output_root / "videos" / video_id
        write_json(
            video_dir / "perception_map.json",
            {
                "video_id": video_id,
                "video_path": video_path,
                "duration_sec": duration_sec,
                "ground_truth_loaded": False,
                "decoder": global_decode,
                "perception_map": perception_map.to_dict(),
            },
        )
        for item in video_items:
            query_started = time.monotonic()
            query_embedding = backend.encode_text(query_text(item))[0]
            ranked_matches = rank_events(
                perception_map,
                query_embedding,
            )
            matches, remaining_event_count = select_ranked_event_batch(
                ranked_matches,
                fps=args.local_fps,
                frame_budget=args.observation_frame_budget,
            )
            refinements: list[dict[str, Any]] = []
            contact_tiles: list[tuple[np.ndarray, str]] = []
            for rank, match in enumerate(matches, start=1):
                best = _nearest_sample(global_frames, match.best_timestamp_sec)
                contact_tiles.append(
                    (
                        best.image_rgb,
                        f"R{rank} best {best.timestamp_sec:.1f}s | {match.start_sec:.1f}-{match.end_sec:.1f}s",
                    )
                )
                local_times = sample_timestamps(
                    match.start_sec, match.end_sec, args.local_fps
                )
                local_images, local_decode = decode_scheduled_frames(
                    video_path, local_times, max_edge=args.max_edge
                )
                local_frames = encode_frame_samples(
                    backend,
                    local_times,
                    local_images,
                    batch_size=args.batch_size,
                )
                children = refine_interval(
                    match.start_sec, match.end_sec, local_frames, config
                )
                child_payloads: list[dict[str, Any]] = []
                for child in children:
                    child_payloads.append(child.to_dict())
                    representative = child.representative_timestamps[
                        len(child.representative_timestamps) // 2
                    ]
                    sample = _nearest_sample(local_frames, representative)
                    contact_tiles.append(
                        (
                            sample.image_rgb,
                            f"R{rank} child {child.event_index} {child.start_sec:.1f}-{child.end_sec:.1f}s",
                        )
                    )
                refinements.append(
                    {
                        "rank": rank,
                        "source_match": match.to_dict(),
                        "decoder": local_decode,
                        "children": child_payloads,
                    }
                )
            question_dir = output_root / "questions" / item["question_id"]
            result = {
                "question_id": item["question_id"],
                "video_id": video_id,
                "source_input": item["source_path"],
                "question": item["question"],
                "options": item["options"],
                "query_text": query_text(item),
                "ground_truth_loaded": False,
                "qwen_requests": 0,
                "api_requests": 0,
                "ranked_event_count": len(ranked_matches),
                "remaining_event_count": remaining_event_count,
                "global_matches": [match.to_dict() for match in matches],
                "refinements": refinements,
                "elapsed_sec": round(time.monotonic() - query_started, 3),
                "contact_sheet": str(question_dir / "contact_sheet.jpg"),
            }
            write_json(question_dir / "result.json", result)
            write_contact_sheet(
                question_dir / "contact_sheet.jpg",
                item["question_id"],
                contact_tiles,
            )
            results.append(result)
        video_summaries.append(
            {
                "video_id": video_id,
                "question_ids": [item["question_id"] for item in video_items],
                "duration_sec": duration_sec,
                "global_sample_count": len(global_frames),
                "event_count": len(perception_map.events),
                "coverage_complete": perception_map.coverage_complete_at_declared_cadence,
                "elapsed_sec": round(time.monotonic() - video_started, 3),
            }
        )
    manifest = {
        "schema_version": "perception_shadow_run_v1",
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "ground_truth_loaded": False,
        "qwen_requests": 0,
        "api_requests": 0,
        "model": {
            "kind": "local_openclip",
            "model_name": args.model_name,
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "device": backend.device,
            "gpu_memory": gpu_memory_stats(backend),
        },
        "config": {
            "global_fps": args.global_fps,
            "local_fps": args.local_fps,
            "observation_frame_budget": args.observation_frame_budget,
            "batch_size": args.batch_size,
            "max_edge": args.max_edge,
        },
        "videos": video_summaries,
        "questions": [
            {
                "question_id": item["question_id"],
                "video_id": item["video_id"],
                "result": str(output_root / "questions" / item["question_id"] / "result.json"),
            }
            for item in inputs
        ],
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated dynamic-perception shadow retrieval without AVP or Qwen."
    )
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--global-fps", type=float, default=1.0)
    parser.add_argument("--local-fps", type=float, default=2.0)
    parser.add_argument("--observation-frame-budget", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-edge", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    try:
        manifest = run_shadow(args)
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(
            output_root / "manifest.json",
            {
                "schema_version": "perception_shadow_run_v1",
                "status": "failed",
                "finished_at": utc_now(),
                "ground_truth_loaded": False,
                "qwen_requests": 0,
                "api_requests": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
