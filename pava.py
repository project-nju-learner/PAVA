"""Portable preparation and bounded execution of the frozen observe-only PAVA."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import urlsplit

from common import DATASETS, QA_KEYS, ROOT, Recorder, now, read, save

sys.path.insert(0, str(ROOT / "core"))


def configure(path):
    cfg = read(path)
    cfg["base_url"] = os.environ.get("OPENAI_BASE_URL") or cfg["base_url"]
    cfg["model"] = os.environ.get("PAVA_MODEL") or cfg["model"]
    url = urlsplit(cfg["base_url"])
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an HTTP(S) base_url without credentials or query parameters")
    if any(k in cfg for k in ("api_key", "token", "password")):
        raise ValueError("Credentials belong in the environment, not config JSON")
    if cfg["input_format"] not in ("openai_images", "dashscope_video"):
        raise ValueError("input_format must be openai_images or dashscope_video")
    if not 4 <= cfg["initial_frames"] <= 250:
        raise ValueError("initial_frames must be 4..250")
    if type(cfg["qa_concurrency"]) is not int or cfg["qa_concurrency"] < 1:
        raise ValueError("qa_concurrency must be a positive integer")
    if cfg.get("automatic_service_retries", 0) != 0:
        raise ValueError("Automatic model-service retries are disabled")
    for key in ("max_rounds", "max_requests_per_question", "max_images_per_question", "question_timeout_seconds", "default_max_output_tokens"):
        if not isinstance(cfg[key], (int, float)) or cfg[key] <= 0:
            raise ValueError("Invalid budget: " + key)
    os.environ["PAVA_MODEL"] = cfg["model"]
    os.environ["PAVA_INITIAL_FRAMES"] = str(cfg["initial_frames"])
    os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
    return cfg


def select(rows, per_dataset=None):
    if not rows or any(r["dataset"] not in DATASETS for r in rows):
        raise ValueError("Roster is empty or contains an unknown dataset")
    groups = [[r for r in rows if r["dataset"] == d] for d in DATASETS]
    if per_dataset is not None:
        if per_dataset < 1 or any(len(g) < per_dataset for g in groups):
            raise ValueError("Not enough prepared questions in every dataset")
        groups = [g[:per_dataset] for g in groups]
    # zip(*groups) would drop questions from the larger datasets in the 597 roster.
    return [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]


def safe_id(value):
    if not isinstance(value, str) or Path(value).name != value or value in ("", ".", "..") or "\\" in value:
        raise ValueError("Invalid question/video identifier")
    return value


def imported_transcript(path, video_id):
    original = read(path)
    def check(value):
        if isinstance(value, dict):
            if value.get("ground_truth_loaded") is True or any(k in value for k in ("answer", "answers", "label", "labels", "correct_option", "ground_truth")):
                raise ValueError("Imported transcript contains evaluation information")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(original)
    if not isinstance(original.get("segments"), list) or original.get("video_id", video_id) != video_id:
        raise ValueError("Imported transcript requires matching video_id and a segments list")
    return dict(segments=original["segments"], producer=original.get("producer", "unknown"),
                imported_from=str(path.resolve()), original_metadata={k:v for k,v in original.items() if k != "segments"})


def validate_materials(record, work):
    vid = record["video_id"]
    if not Path(record["map_path"]).is_file():
        raise FileNotFoundError("Prepared index missing: " + vid)
    folder = work / "materials/uniform_frames" / vid
    binding = read(folder / "manifest.json")
    if (binding["video_path"], binding["source_size"], binding["source_mtime_ns"], binding["frame_count"]) != (record["video_path"], record["video_bytes"], record["video_mtime_ns"], record["initial_frames"]):
        raise ValueError("Prepared initial frames do not match video: " + vid)
    if any(not (folder / f"frame_{i+1:03d}.jpg").is_file() for i in range(record["initial_frames"])):
        raise FileNotFoundError("Prepared initial frames missing: " + vid)
    if record["use_asr"]:
        from experiments.semantic_delta_pilot.asr_mode import ASRContext
        ASRContext(work / "materials/transcripts" / vid / "transcript.json", video_path=record["video_path"],
                   backend=None, cache_dir=work / "validation_unused", run_dir=work)


def prepare(args):
    cfg = configure(args.config)
    if not (args.transcripts_dir or args.asr_model or args.no_asr):
        raise ValueError("Choose --transcripts-dir, --asr-model, or explicitly --no-asr")
    if args.no_asr and (args.transcripts_dir or args.asr_model):
        raise ValueError("--no-asr cannot be combined with transcript/ASR inputs")
    rows = select(read(args.roster)["samples"], args.per_dataset)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = args.openclip_checkpoint.resolve(strict=True)
    settings = dict(initial_frames=cfg["initial_frames"], checkpoint=str(checkpoint),
                    roster=str(args.roster.resolve()), asr_model=str(args.asr_model.resolve()) if args.asr_model else None,
                    transcripts_dir=str(args.transcripts_dir.resolve()) if args.transcripts_dir else None,
                    no_asr=args.no_asr, index_device=args.device)
    with (work / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / "setup.json").exists() and read(work / "setup.json") != settings:
            raise ValueError("Preparation settings changed; use a new work directory")
        save(work / "setup.json", settings)
        prepare_rows(args, cfg, rows, work, checkpoint)


def prepare_rows(args, cfg, rows, work, checkpoint):
    import cv2
    import torch
    from experiments.pollution_audit import probe_media, media_duration_sec
    from scripts.run_perception_shadow import OpenClipBackend, DynamicPerceptionConfig, build_global_perception, sample_timestamps, decode_scheduled_frames, encode_frame_samples
    from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import DashScopeFlashTransport
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    backend = asr_model = None
    outputs = []
    materials = work / "materials"
    started = time.perf_counter()
    for row in rows:
        qid, vid = safe_id(row["question_id"]), safe_id(row["video_id"])
        name = safe_id(row["video_filename"])
        video = (args.video_root / row["dataset"] / "videos" / name).resolve(strict=True)
        stat = video.stat()
        record_path = materials / "preparation" / (vid + ".json")
        if record_path.exists():
            record = read(record_path)
            if (record["video_path"], record["video_bytes"], record["video_mtime_ns"]) != (str(video), stat.st_size, stat.st_mtime_ns):
                raise ValueError("Prepared media changed: " + vid)
            if record["initial_frames"] != cfg["initial_frames"]:
                raise ValueError("Prepared frame count changed")
        else:
            t0 = time.perf_counter()
            index_load_seconds = asr_load_seconds = 0.0
            probe = probe_media(video)
            duration = media_duration_sec(probe)
            if backend is None:
                load_start = time.perf_counter()
                backend = OpenClipBackend(model_name="ViT-B-32", checkpoint_path=checkpoint, device=args.device)
                backend._load()
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                index_load_seconds = time.perf_counter()-load_start
            index_start = time.perf_counter()
            stamps = sample_timestamps(0, duration, 1.0)
            capture = cv2.VideoCapture(str(video))
            fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
            capture.release()
            if fps <= 0 or count <= 0:
                raise ValueError("Invalid video frame metadata: " + vid)
            stamps = tuple(dict.fromkeys(min(t, (count-1)/fps) for t in stamps))
            decoded, decoder = decode_scheduled_frames(str(video), stamps, max_edge=256)
            decode_end = time.perf_counter()
            frames = encode_frame_samples(backend, stamps, decoded, batch_size=32)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            encode_end = time.perf_counter()
            perception = build_global_perception(duration, frames, DynamicPerceptionConfig())
            if not perception.coverage_complete_at_declared_cadence:
                raise ValueError("Index coverage incomplete: " + vid)
            map_path = materials / "maps" / (vid + ".json")
            save(map_path, dict(video_id=vid, video_path=str(video), duration_sec=duration,
                 index_duration_sec=duration, ground_truth_loaded=False, decoder=decoder, perception_map=perception.to_dict()))
            index_end = time.perf_counter()
            del decoded, frames, perception
            client = DashScopeFlashTransport(base_url=cfg["base_url"])
            frame_start = time.perf_counter()
            try:
                _, receipt = client.prepare_uniform_seed(video, duration, materials / "uniform_frames" / vid)
            finally:
                client.close()
            frame_end = time.perf_counter()
            transcript_start = time.perf_counter()
            imported = args.transcripts_dir / (vid + ".json") if args.transcripts_dir else None
            transcript = None
            text_status = "disabled" if args.no_asr else "unavailable"
            if imported and imported.is_file():
                transcript = imported_transcript(imported, vid)
                text_status = "imported_transcript"
            elif args.asr_model and any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
                if asr_model is None:
                    from faster_whisper import WhisperModel
                    load_start = time.perf_counter()
                    asr_model = WhisperModel(str(args.asr_model.resolve(strict=True)), device=args.asr_device, compute_type="float16" if args.asr_device == "cuda" else "int8", local_files_only=True)
                    asr_load_seconds = time.perf_counter()-load_start
                with tempfile.TemporaryDirectory(prefix="pava_audio_") as temp:
                    wav = Path(temp) / "audio.wav"
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn", "-ar", "16000", "-ac", "1", str(wav)], check=True)
                    segments, info = asr_model.transcribe(str(wav), beam_size=5)
                    transcript = dict(segments=[dict(id=s.id, start=s.start, end=s.end, text=s.text) for s in segments], producer="faster-whisper", model=str(args.asr_model.resolve()), language=info.language)
                text_status = "new_local_asr"
            elif not args.no_asr and not any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
                text_status = "no_audio_and_no_imported_transcript"
            elif args.transcripts_dir and not args.asr_model:
                raise FileNotFoundError("Transcript missing for audio video: " + vid)
            use_asr = bool(transcript and transcript["segments"])
            if transcript is not None:
                transcript.update(video_id=vid, source_video_path=str(video), source_size=stat.st_size,
                    source_mtime_ns=stat.st_mtime_ns, source_duration_sec=duration, origin_run=str(work), ground_truth_loaded=False)
                target = materials / "transcripts" / vid
                save(target / "transcript.json", transcript)
                save(target / "manifest.json", dict(status="completed", source_video_path=str(video), segments=len(transcript["segments"]), producer=transcript["producer"]))
            record = dict(video_id=vid, dataset=row["dataset"], video_path=str(video), video_bytes=stat.st_size,
                video_mtime_ns=stat.st_mtime_ns, duration_sec=duration, map_path=str(map_path), use_asr=use_asr,
                transcript_status=text_status, initial_frames=cfg["initial_frames"], index_frames=len(stamps),
                index_decode_seconds=decode_end-index_start, index_encode_seconds=encode_end-decode_end,
                index_total_seconds=index_end-index_start, initial_frames_seconds=frame_end-frame_start,
                transcript_preparation_seconds=time.perf_counter()-transcript_start,
                index_model_load_seconds=index_load_seconds, asr_model_load_seconds=asr_load_seconds,
                preparation_seconds=time.perf_counter()-t0, ground_truth_loaded=False)
            validate_materials(record, work)
            save(record_path, record)
        validate_materials(record, work)
        outputs.append({**{k:row[k] for k in (*QA_KEYS, "dataset")}, **{k:record[k] for k in ("video_path", "map_path", "use_asr", "duration_sec")}})
        print("PREPARED", len(outputs), len(rows), qid, record["transcript_status"], flush=True)
    save(work / "inputs.json", outputs)
    save(work / "prepared.json", dict(questions=len(outputs), videos=len({r['video_id'] for r in outputs}), initial_frames=cfg["initial_frames"], elapsed_seconds=time.perf_counter()-started, model_api_requests=0))


def worker(run, qid):
    cfg = configure(run / "config.json")
    work = Path(read(run / "started.json")["work"])
    row = next(r for r in read(run / "inputs.json") if r["question_id"] == qid)
    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.suffix == ".parquet" or path.name in ("answers.json", "labels.json", "evaluation_sources.json") or "evaluation_after_inference" in path.name:
                raise PermissionError("Evaluation labels are not available during inference")
    sys.addaudithook(guard)
    from experiments.semantic_delta_pilot import dashscope_flash_video_adapter as transport
    from experiments.semantic_delta_pilot import avp_integration as integration
    from experiments.run_videomme_mcq_ab import run_one
    from transport import recorded_opener
    out = run / "questions" / qid
    out.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(out / "http", cfg, qid, run / "service_stop.json")
    original_init = transport.DashScopeFlashTransport.__init__
    original_opener = original_init.__kwdefaults__["request_opener"]
    def recorded_init(self, *args, **kwargs):
        kwargs["request_opener"] = recorded_opener(recorder, original_opener)
        return original_init(self, *args, **kwargs)
    transport.DashScopeFlashTransport.__init__ = recorded_init
    integration.urlopen = recorded_opener(recorder, integration.urlopen)
    save(out / "started.json", dict(at=now(), pid=os.getpid(), options_visible=True, ground_truth_loaded=False))
    started = time.perf_counter()
    try:
        result = run_one(row={k:row[k] for k in QA_KEYS}, video_path=Path(row["video_path"]),
            sample_dir=run / "samples" / qid, mode="repair", base_url=cfg["base_url"],
            max_rounds=cfg["max_rounds"], max_repair_observations=cfg["max_repairs"],
            temperature=cfg["temperature"], seed=cfg["inference_seed"], operator_gate_mode="off",
            event_scan_map=row["map_path"], search_need_checkpoint=read(work / "setup.json")["checkpoint"],
            use_asr=row["use_asr"], asr_roots=[str(work / "materials/transcripts")], asr_cache_dir=run / "asr_understanding" / qid)
    except Exception as error:
        result = dict(status="failed", error_type=type(error).__name__, error=recorder.clean(error), traceback=recorder.clean(traceback.format_exc()))
    save(out / "result.json", dict(question_id=qid, dataset=row["dataset"], video_id=row["video_id"],
        result=result, usage=recorder.summary(), online_qa_seconds=time.perf_counter()-started,
        options_visible=True, ground_truth_loaded=False, model_outputs_reused=False))


def terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_batch(args):
    cfg = configure(args.config)
    work = args.work.resolve(strict=True)
    if read(work / "prepared.json")["initial_frames"] != cfg["initial_frames"]:
        raise ValueError("Run frame count differs from prepared frames")
    rows = select(read(work / "inputs.json"), args.per_dataset)
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    for row in rows:
        record = read(work / "materials/preparation" / (safe_id(row["video_id"]) + ".json"))
        video = Path(record["video_path"])
        if (video.stat().st_size, video.stat().st_mtime_ns) != (record["video_bytes"], record["video_mtime_ns"]):
            raise ValueError("Video changed after preparation: " + row["video_id"])
        validate_materials(record, work)
    if args.concurrency is not None:
        if type(args.concurrency) is not int or args.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        cfg["qa_concurrency"] = args.concurrency
    print(json.dumps(dict(questions=len(rows), counts=dict(Counter(r["dataset"] for r in rows)), model=cfg["model"],
        endpoint=cfg["base_url"], initial_frames=cfg["initial_frames"], concurrency=cfg["qa_concurrency"],
        max_requests=len(rows)*cfg["max_requests_per_question"], max_image_submissions=len(rows)*cfg["max_images_per_question"],
        payload="questions, options, available transcripts, selected video frames", automatic_retries=0,
        reused="video/index/transcript/initial-frame artifacts only", ground_truth_loaded=False), indent=2), flush=True)
    if not args.execute:
        print("DRY RUN: no model requests")
        return
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.run_id):
        raise ValueError("Supply a new readable --run-id")
    run = work / "runs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(run / "config.json", cfg)
        save(run / "inputs.json", rows)
        save(run / "started.json", dict(at=now(), work=str(work), pid=os.getpid(), ground_truth_loaded=False))
        (run / "samples").mkdir()
        (run / "samples/uniform_frames").symlink_to(work / "materials/uniform_frames", target_is_directory=True)
        (run / "worker_logs").mkdir()
        active, results, next_row = {}, [], 0
        started = time.perf_counter()
        try:
            while active or next_row < len(rows):
                stopped = (run / "service_stop.json").exists()
                while not stopped and len(active) < cfg["qa_concurrency"] and next_row < len(rows):
                    row = rows[next_row]
                    qid = safe_id(row["question_id"])
                    with (run / "worker_logs" / (qid + ".log")).open("w") as log:
                        process = subprocess.Popen([sys.executable, "-B", "-u", str(ROOT / "pava.py"), "_worker", str(run), qid], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[qid] = (row, process, time.perf_counter())
                    next_row += 1
                    stopped = (run / "service_stop.json").exists()
                for qid, (row, process, begin) in list(active.items()):
                    timeout = process.poll() is None and time.perf_counter()-begin > cfg["question_timeout_seconds"]
                    if timeout:
                        terminate(process)
                    if process.poll() is None:
                        continue
                    result_path = run / "questions" / qid / "result.json"
                    if result_path.is_file():
                        item = read(result_path)
                    else:
                        usage_path = result_path.parent / "http/usage.json"
                        item = dict(question_id=qid, dataset=row["dataset"], video_id=row["video_id"],
                            result=dict(status="failed", error_type="QuestionTimeout" if timeout else "WorkerError", exit_code=process.returncode),
                            usage=read(usage_path) if usage_path.exists() else None, online_qa_seconds=None, ground_truth_loaded=False)
                    item["worker_wall_seconds"] = time.perf_counter()-begin
                    save(result_path, item)
                    results.append(item)
                    save(run / "results.json", results)
                    del active[qid]
                    print("FINISHED", len(results), len(rows), qid, item["result"]["status"], flush=True)
                save(run / "state.json", dict(planned=len(rows), completed=len(results), started=next_row, active=list(active), service_stopped=stopped))
                if stopped and not active:
                    break
                if active:
                    time.sleep(.5)
        except BaseException:
            for _, process, _ in active.values():
                terminate(process)
            raise
        save(run / "finished.json", dict(planned=len(rows), completed=len(results), all_workers_exited=True,
            phase="completed" if len(results) == len(rows) else "stopped", batch_wall_seconds=time.perf_counter()-started, ground_truth_loaded=False))
    print("Saved", run, flush=True)


def main():
    if sys.platform != "linux":
        raise RuntimeError("Run project code on Ubuntu Linux")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build local media artifacts; no remote model calls")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--roster", type=Path, default=ROOT / "data/roster597.json")
    prep.add_argument("--video-root", type=Path, required=True)
    prep.add_argument("--work", type=Path, required=True)
    prep.add_argument("--openclip-checkpoint", type=Path, required=True)
    prep.add_argument("--device", default="cuda:0")
    prep.add_argument("--asr-model", type=Path)
    prep.add_argument("--asr-device", choices=("cuda", "cpu"), default="cuda")
    prep.add_argument("--transcripts-dir", type=Path)
    prep.add_argument("--no-asr", action="store_true")
    prep.add_argument("--per-dataset", type=int)
    run = sub.add_parser("run", help="Default dry-run; --execute enables model requests")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--per-dataset", type=int)
    run.add_argument("--concurrency", type=int)
    run.add_argument("--run-id")
    run.add_argument("--execute", action="store_true")
    worker_parser = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("run", type=Path)
    worker_parser.add_argument("qid")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run_batch(args)
    else:
        worker(args.run, args.qid)


if __name__ == "__main__":
    main()
