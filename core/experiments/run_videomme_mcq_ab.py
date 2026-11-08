"""Run the paired Video-MME MCQ bare-AVP or full Audit/Repair arm.

The inference input may contain question text and official options only.  This
runner never resolves or reads the evaluation sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from avp.main import Controller
from avp.video_utils import set_metadata_source
from experiments.pollution_audit import (
    format_mcq_query,
    load_qa_only,
    media_duration_sec,
    probe_media,
    resolve_video,
    write_json,
    write_qa_only_metadata_source,
)
from experiments.semantic_delta_pilot.avp_integration import (
    OpenAIJSONBackend,
    RepairController,
)
from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import (
    make_dashscope_flash_client,
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def count_requests(client: Any) -> Callable[[], int]:
    counter = {"value": 0}
    original = client.client.post_chat_completion
    partition_counter = getattr(client, "partition_request_count", None)

    def counted(payload: Any) -> str:
        counter["value"] += 1
        return original(payload)

    client.client.post_chat_completion = counted
    return lambda: int(counter["value"]) + (
        int(partition_counter()) if callable(partition_counter) else 0
    )


def classify_sample_status(final: dict[str, Any]) -> str:
    if str(final.get("status") or "").strip().upper() == "FAILED":
        return "failed"
    return "completed" if str(final.get("selected_option") or "").strip() else "unresolved"


def count_sample_statuses(samples: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for sample in samples if sample.get("status") == status)
        for status in ("completed", "unresolved", "failed")
    }


def classify_run_status(sample_status_counts: dict[str, int]) -> str:
    return (
        "completed_with_failures"
        if int(sample_status_counts.get("failed") or 0) > 0
        else "completed"
    )


def run_one(
    *,
    row: dict[str, Any],
    video_path: Path,
    sample_dir: Path,
    mode: str,
    base_url: str,
    max_rounds: int | None,
    max_repair_observations: int | None,
    temperature: float,
    seed: int,
    operator_gate_mode: str = "off",
    event_assist_runtime: Any | None = None,
    event_prefill_map: str | None = None,
    event_scan_map: str | None = None,
    search_need_checkpoint: str | None = None,
    use_asr: bool = False,
    asr_roots: list[str] | None = None,
    asr_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if use_asr and (mode != "repair" or event_prefill_map is not None or event_assist_runtime is not None):
        raise ValueError("asr_requires_repair_without_prefill_or_event_assist")
    if use_asr and not asr_roots:
        raise ValueError("use_asr_requires_asr_root")
    if use_asr and operator_gate_mode != "off":
        raise ValueError("asr_utterance_times_not_supported_by_operator_gate")
    if search_need_checkpoint and not event_scan_map:
        raise ValueError("search_need_routing_requires_event_scan_map")
    if event_scan_map is not None and (mode != "repair" or event_assist_runtime is not None or event_prefill_map is not None):
        raise ValueError("event_scan_requires_unassisted_repair_without_prefill")
    if event_prefill_map is not None and (mode != "repair" or event_assist_runtime is not None):
        raise ValueError("event_prefill_requires_unassisted_repair_mode")
    sample_dir.mkdir(parents=True, exist_ok=False)
    media_probe = probe_media(video_path)
    duration = media_duration_sec(media_probe)
    metadata_path = sample_dir / "qa_only_video_metadata.json"
    write_qa_only_metadata_source(
        metadata_path,
        video_path=video_path,
        video_id=row["video_id"],
        duration_sec=duration,
    )
    set_metadata_source(str(metadata_path))
    write_json(
        sample_dir / "inference_input.json",
        {
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "video_path": str(video_path),
            "duration_sec": duration,
            "ground_truth_loaded": False,
            "use_asr": use_asr,
        },
    )

    avp_client = make_dashscope_flash_client(
        base_url=base_url,
        model=os.environ.get("PAVA_MODEL", "qwen3.8-flash"),
        temperature=temperature,
        seed=seed,
    )
    avp_request_count = count_requests(avp_client)
    runtime: RepairController | None = None
    audit_backend: OpenAIJSONBackend | None = None
    evidence_audit_client: Any | None = None
    if mode == "repair":
        audit_backend = OpenAIJSONBackend(
            base_url=base_url,
            model=os.environ.get("PAVA_MODEL", "qwen3.8-flash"),
            trace_path=sample_dir / "native" / "semantic_delta" / "model_calls.jsonl",
            temperature=temperature,
            seed=seed,
        )
        evidence_audit_client = make_dashscope_flash_client(
            base_url=base_url,
            model=os.environ.get("PAVA_MODEL", "qwen3.8-flash"),
            temperature=temperature,
            seed=seed,
        )
        runtime = RepairController(
            run_dir=sample_dir / "native",
            backend=audit_backend,
            evidence_audit_client=evidence_audit_client,
            raw_qwen_only=False,
            max_repair_observations=max_repair_observations,
            operator_gate_mode=operator_gate_mode,
        )

    started = time.monotonic()
    asr = None
    asr_input_status = "disabled"
    try:
        if use_asr:
            from experiments.semantic_delta_pilot.asr_mode import ASRContext, find_transcript
            try:
                transcript = find_transcript(asr_roots, row["video_id"], video_path)
            except FileNotFoundError:
                transcript = None
            asr_input_status = "missing_text"
            if transcript is not None:
                asr = ASRContext(transcript, video_path=video_path, backend=audit_backend,
                    cache_dir=asr_cache_dir or sample_dir.parent / "asr_cache",
                    run_dir=sample_dir / "native")
                runtime.asr_context = asr
                runtime.asr_background = asr.initialize()
                asr_input_status = "text_available" if asr.seed_text() else "empty_text"
        scan = None
        if event_scan_map is not None:
            from experiments.event_partition_scan import install_event_partition_scan

            scan = install_event_partition_scan(avp_client, runtime, map_path=event_scan_map,
                video_id=row["video_id"], video_path=video_path, duration=duration,
                run_dir=sample_dir / "native")
        if search_need_checkpoint or use_asr:
            from experiments.event_partition_scan import install_search_need_routing
            text_encoder = None
            if search_need_checkpoint:
                from avp.perception_tools import OpenClipBackend
                encoder = OpenClipBackend(model_name="ViT-B-32", checkpoint_path=search_need_checkpoint, device="cpu")
                text_encoder = encoder.encode_text
            install_search_need_routing(avp_client, runtime, scan, text_encoder=text_encoder,
                asr=asr, duration=duration, run_dir=sample_dir / "native")
            # This isolated variant replaces the text-only seed with the tested
            # uniform250 input, keeping the existing Reflector and continuation.
            seed_duration = duration
            if event_scan_map:
                scan_metadata = json.loads(Path(event_scan_map).read_text())
                seed_duration = float(scan_metadata.get("index_duration_sec") or duration)
            runtime.uniform_seed_content, runtime.uniform_seed_manifest = (
                avp_client.client.prepare_uniform_seed(video_path, seed_duration,
                    sample_dir.parent / "uniform_frames" / row["video_id"]))
            runtime.asr_decision_client = avp_client
            write_json(sample_dir / "uniform_seed.json", runtime.uniform_seed_manifest)
        controller = Controller(
            run_dir=str(sample_dir / "native"),
            video_path=str(video_path),
            client=avp_client,
            options=row["options"],
            admission_runtime=runtime,
            event_assist_runtime=event_assist_runtime,
        )
        run_kwargs = {}
        if event_prefill_map is not None:
            from experiments.event_1fps_prefill import prepare_event_prefill

            run_kwargs["initial_audited_observation"] = prepare_event_prefill(
                controller, row=row, map_path=event_prefill_map,
            )
        result = controller.run(
            query=format_mcq_query(row["question"], row["options"]),
            max_rounds=max_rounds,
            **run_kwargs,
        )
        final = result.get("final") if isinstance(result, dict) else None
        if not isinstance(final, dict):
            raise ValueError("controller_final_answer_missing")
        runtime_summary = runtime.summary() if runtime else {}
        status = {
            "status": classify_sample_status(final),
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "selected_option": final.get("selected_option"),
            "selected_option_text": final.get("selected_option_text"),
            "result": result,
            "avp_request_count": avp_request_count(),
            "audit_request_count": audit_backend.request_count if audit_backend else 0,
            "evidence_audit_request_count": (
                runtime_summary.get("evidence_audit_requests", 0)
            ),
            "operator_gate_mode": operator_gate_mode,
            "operator_gate_status": (
                (runtime_summary.get("operator_gate_packet") or {}).get("status")
            ),
            "operator_gate_request_count": runtime_summary.get(
                "operator_gate_request_count", 0
            ),
            "repair_observations": runtime.repair_observations if runtime else 0,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "ground_truth_loaded": False,
        }
    except Exception as exc:
        runtime_summary = runtime.summary() if runtime else {}
        status = {
            "status": "failed",
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "avp_request_count": avp_request_count(),
            "audit_request_count": audit_backend.request_count if audit_backend else 0,
            "evidence_audit_request_count": (
                runtime_summary.get("evidence_audit_requests", 0)
            ),
            "operator_gate_mode": operator_gate_mode,
            "operator_gate_status": (
                (runtime_summary.get("operator_gate_packet") or {}).get("status")
            ),
            "operator_gate_request_count": runtime_summary.get(
                "operator_gate_request_count", 0
            ),
            "repair_observations": runtime.repair_observations if runtime else 0,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "ground_truth_loaded": False,
        }
    finally:
        status["use_asr"] = use_asr
        if use_asr:
            status["asr"] = {"enabled": True, "roots": list(asr_roots),
                "input_status": asr_input_status,
                "initialization": asr.initialization if asr is not None else None,
                "source": asr.source if asr is not None else None,
                "reads": asr.reads if asr is not None else [], "recognition_requests": 0,
                "request_accounting": "Initialization/search requests are included in audit_request_count; initialization is video preparation cost.",
                "ground_truth_loaded": False}
        if event_scan_map is not None:
            status["event_scan_map"] = str(event_scan_map)
            status["event_scan_fps_policy"] = "fixed_0.1_0.5_1_short_video_1fps"
        if search_need_checkpoint:
            status["search_need_routing"] = {"enabled": True, "checkpoint_path": str(search_need_checkpoint),
                "candidate_limit": 5, "initial_need": "verifier_generated_neutral_need",
                "requery_need": "verifier_missing_proposition", "observe_options_visible": False}
        if event_prefill_map is not None:
            status["event_prefill_map"] = str(event_prefill_map)
            status["event_prefill_artifact"] = str(sample_dir / "native" / "event_prefill.json")
            status["prefill_options_visible_to_observer"] = False
            status["prefill_question_visible_to_observer"] = True
        write_json(sample_dir / "result.json", status)
        avp_client.cleanup_uploaded_files()
        if evidence_audit_client is not None:
            evidence_audit_client.cleanup_uploaded_files()
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bare", "repair"), required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-repair-observations", type=int, default=10)
    parser.add_argument("--max-video-duration-sec", type=float, default=2400.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--use_asr", "--use-asr", action="store_true",
                        help="Opt in to ASR full-text understanding and need/Audit acquisition (repair only).")
    parser.add_argument("--asr-root", "--asr_root", action="append", default=None,
                        help="Existing ASR run/asr directory or transcript.json; repeat to reuse multiple batches. Completed manifests only; no transcription.")
    parser.add_argument("--event-prefill-map", action="append", default=None,
                        help="Opt in to audited Event 1fps prefill; one map per selected video (repair only).")
    parser.add_argument("--search-need-checkpoint", default=None,
                        help="Opt in to neutral search-need Event retrieval with this local ViT-B-32 checkpoint; requires event-scan-map.")
    parser.add_argument("--event-scan-map", action="append", default=None,
                        help="Split planner-requested whole-video scans by Event at requested FPS; no prefill (repair only).")
    parser.add_argument(
        "--operator-gate-mode",
        choices=("off", "shadow", "active"),
        default="off",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_rounds <= 0:
        raise ValueError("max_rounds_must_be_positive")
    if args.max_repair_observations <= 0:
        raise ValueError("max_repair_observations_must_be_positive")
    if args.max_video_duration_sec <= 0:
        raise ValueError("max_video_duration_sec_must_be_positive")
    if args.mode != "repair" and args.operator_gate_mode != "off":
        raise ValueError("operator_gate_requires_repair_mode")
    if args.event_prefill_map and args.mode != "repair":
        raise ValueError("event_prefill_requires_repair_mode")
    if args.event_scan_map and (args.mode != "repair" or args.event_prefill_map):
        raise ValueError("event_scan_requires_repair_without_prefill")
    if args.search_need_checkpoint and not args.event_scan_map:
        raise ValueError("search_need_routing_requires_event_scan_map")
    if args.use_asr and (args.mode != "repair" or args.event_prefill_map):
        raise ValueError("asr_requires_repair_without_prefill")
    if args.use_asr and not args.asr_root:
        raise ValueError("use_asr_requires_asr_root")
    if args.asr_root and not args.use_asr:
        raise ValueError("asr_root_requires_use_asr")
    if args.use_asr and args.operator_gate_mode != "off":
        raise ValueError("asr_utterance_times_not_supported_by_operator_gate")
    base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL_required")
    if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        raise RuntimeError("DASHSCOPE_API_KEY_required")

    questions_path = Path(args.questions).resolve(strict=True)
    video_root = Path(args.video_root).resolve(strict=True)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"output_already_exists:{out}")
    rows = load_qa_only(questions_path)
    if any(not row.get("options") for row in rows):
        raise ValueError("all_selected_rows_must_have_official_options")
    if args.question_id:
        wanted = list(args.question_id)
        by_id = {row["question_id"]: row for row in rows}
        missing = [question_id for question_id in wanted if question_id not in by_id]
        if missing:
            raise ValueError(f"unknown_question_ids:{missing}")
        rows = [by_id[question_id] for question_id in wanted]
    duration_by_video: dict[str, float] = {}
    eligible_rows: list[dict[str, Any]] = []
    for row in rows:
        video_id = row["video_id"]
        duration = duration_by_video.get(video_id)
        if duration is None:
            duration = media_duration_sec(
                probe_media(resolve_video(video_root, video_id))
            )
            duration_by_video[video_id] = duration
        if duration <= args.max_video_duration_sec:
            eligible_rows.append(row)
    rows = eligible_rows
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit_must_be_positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no_questions_selected")

    prefill_maps = {}
    scan_maps = {}
    if args.event_prefill_map or args.event_scan_map:
        from scripts.run_hierarchical_event_tree_shadow import load_map_artifact

        selected_maps = prefill_maps if args.event_prefill_map else scan_maps
        for path in (args.event_prefill_map or args.event_scan_map):
            video_id, source, duration, _ = load_map_artifact(path)
            if video_id in selected_maps:
                raise ValueError("duplicate_event_prefill_video")
            if Path(source).resolve() != resolve_video(video_root, video_id).resolve():
                raise ValueError("event_prefill_map_video_mismatch")
            selected_maps[video_id] = str(Path(path).resolve())
        if set(selected_maps) != {str(row["video_id"]) for row in rows}:
            raise ValueError("event_prefill_maps_must_match_selected_videos")

    out.mkdir(parents=True)
    manifest = {
        "status": "running",
        "started_at": utc_now(),
        "mode": args.mode,
        "model": "qwen3.8-flash",
        "video_transport": "dashscope_base64_jpeg_frames",
        "video_fps": 0.1,
        "max_video_frames": 250,
        "max_video_duration_sec": args.max_video_duration_sec,
        "questions_path": str(questions_path),
        "video_root": str(video_root),
        "question_ids": [row["question_id"] for row in rows],
        "bare_max_rounds": args.max_rounds,
        "max_repair_observations": args.max_repair_observations,
        "temperature": args.temperature,
        "seed": args.seed,
        "options_loaded": True,
        "ground_truth_loaded": False,
        "gold_span_loaded": False,
        "cache_reuse_expected": False,
        "automatic_retry": False,
        "observe_frozen": False,
        "repair_controller_enabled": args.mode == "repair",
        "use_asr": args.use_asr,
        "asr_roots": args.asr_root if args.use_asr else [],
        "asr_understanding_cache": str(out / "asr_cache") if args.use_asr else None,
        "asr_transcription_enabled": False,
        "operator_gate_mode": args.operator_gate_mode,
        "operator_gate_ground_truth_visible": False,
        "sample_status_counts": {
            "completed": 0,
            "unresolved": 0,
            "failed": 0,
        },
        "samples": [],
    }
    if prefill_maps:
        manifest["event_prefill_maps"] = prefill_maps
        manifest["event_prefill_fps"] = 1.0
        manifest["prefill_options_visible_to_observer"] = False
        manifest["prefill_question_visible_to_observer"] = True
    if scan_maps:
        manifest["event_scan_maps"] = scan_maps
        manifest["event_scan_fps_policy"] = "fixed_0.1_0.5_1_short_video_1fps"
    write_json(out / "manifest.json", manifest)
    for index, row in enumerate(rows, 1):
        print(
            f"START {args.mode} {index}/{len(rows)} "
            f"video={row['video_id']} question={row['question_id']}",
            flush=True,
        )
        sample_dir = out / "samples" / (
            f"{row['video_id']}__{row['question_id']}"
        )
        status = run_one(
            row=row,
            video_path=resolve_video(video_root, row["video_id"]),
            sample_dir=sample_dir,
            mode=args.mode,
            base_url=base_url,
            max_rounds=args.max_rounds,
            max_repair_observations=args.max_repair_observations,
            temperature=args.temperature,
            seed=args.seed,
            operator_gate_mode=args.operator_gate_mode,
            **({"use_asr": True, "asr_roots": args.asr_root,
                "asr_cache_dir": out / "asr_cache"} if args.use_asr else {}),
            **({"event_prefill_map": prefill_maps[str(row["video_id"])]} if prefill_maps else {}),
            **({"event_scan_map": scan_maps[str(row["video_id"])]} if scan_maps else {}),
            **({"search_need_checkpoint": args.search_need_checkpoint} if args.search_need_checkpoint else {}),
        )
        manifest["samples"].append(status)
        manifest["sample_status_counts"] = count_sample_statuses(manifest["samples"])
        write_json(out / "manifest.json", manifest)
        print(
            f"END {args.mode} {index}/{len(rows)} "
            f"status={status['status']} selected={status.get('selected_option')} "
            f"avp_requests={status['avp_request_count']} "
            f"audit_requests={status['audit_request_count']} "
            f"elapsed={status['elapsed_sec']}s",
            flush=True,
        )
    any_failed = manifest["sample_status_counts"]["failed"] > 0
    manifest["status"] = classify_run_status(manifest["sample_status_counts"])
    manifest["finished_at"] = utc_now()
    manifest["ground_truth_loaded"] = False
    write_json(out / "manifest.json", manifest)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
