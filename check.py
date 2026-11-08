"""Offline release checks. No model or network requests; run on Ubuntu."""
import ast
from collections import Counter
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from common import ROOT, DATASETS, QA_KEYS, Recorder, BudgetExit, read, save
import pava
from transport import local_payload, recorded_opener


def no_network(*args, **kwargs):
    raise RuntimeError("Offline check: network is forbidden")


socket.socket.connect = socket.socket.connect_ex = socket.create_connection = no_network


def expect(error_type, function):
    try:
        function()
    except error_type:
        return
    raise AssertionError("Expected " + error_type.__name__)


def source_and_roster():
    for file in ROOT.rglob("*.py"):
        ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
    for name in ("experiments.run_videomme_mcq_ab", "scripts.run_hierarchical_event_tree_shadow", "avp.perception_tools.openclip_backend"):
        module = importlib.import_module(name)
        assert Path(module.__file__).is_relative_to(ROOT / "core")
    roster = read(ROOT / "data/roster597.json")
    rows = pava.select(roster["samples"])
    assert len(rows) == len({r["question_id"] for r in rows}) == 597
    assert Counter(r["dataset"] for r in rows) == dict(zip(DATASETS, (180, 150, 140, 127)))
    assert len({r["video_id"] for r in rows}) == 465
    assert Counter(r["dataset"] for r in pava.select(rows, 5)) == {d:5 for d in DATASETS}
    assert pava.select(rows) == rows
    assert roster["seed"] == 42
    pilot = read(ROOT / "data/roster80.json")
    assert pilot["samples"] == pava.select(roster["samples"], 20)
    import download_videos
    selected, question_count = download_videos.selected_rows(ROOT / "data/roster80.json")
    assert question_count == 80 and len(selected) == len({(r["dataset"], r["video_id"]) for r in pilot["samples"]})
    assert download_videos.selected_rows()[1] == 597
    assert len(download_videos.selected_rows()[0]) == 465
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        with patch.object(download_videos, "complete", return_value=True), contextlib.redirect_stdout(io.StringIO()):
            download_videos.run(folder, selected, question_count)
        manifest = read(folder / "video_manifest.json")
        assert manifest["question_count"] == 80 and manifest["video_count"] == 75
        assert len(manifest["files"]) == 75
        save(folder / "bad.json", dict(samples=[dict(question_id="bad",dataset="MLVU",video_filename="not_in_plan.mp4")]))
        expect(download_videos.DownloadError, lambda: download_videos.selected_rows(folder / "bad.json"))
    forbidden = {"answer", "label", "correct_option", "ground_truth"}
    assert all(not forbidden.intersection(r) for r in rows)
    expect(ValueError, lambda: pava.select(rows, 181))
    expect(ValueError, lambda: pava.safe_id("../escape"))


def transport_and_usage():
    cfg = read(ROOT / "config.example.json")
    payload = dict(model=cfg["model"], enable_thinking=False, messages=[dict(role="user", content=[dict(type="video", video=["data:image/jpeg;base64,eA=="]*2, fps=2)])])
    images = local_payload(payload, cfg)
    assert images["chat_template_kwargs"] == {"enable_thinking":False}
    assert images["messages"][0]["content"][2]["text"].endswith("0.500 seconds.")
    assert Recorder.image_count(images) == 2 and payload["messages"][0]["content"][0]["type"] == "video"
    native = local_payload(payload, {**cfg, "input_format":"dashscope_video"})
    assert native["enable_thinking"] is False and native["messages"][0]["content"][0]["type"] == "video"
    events = [dict(id="fixture", model=cfg["model"], choices=[dict(index=0, delta=dict(content="A"), finish_reason=None)]),
              dict(choices=[dict(index=0, delta={}, finish_reason="stop")]),
              dict(choices=[], usage=dict(prompt_tokens=100, completion_tokens=5, total_tokens=105))]
    stream = b"".join(("data: " + json.dumps(e) + "\n\n").encode() for e in events) + b"data: [DONE]\n\n"
    req = Request(cfg["base_url"] + "/chat/completions", data=json.dumps(payload).encode())
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        config = {**cfg, "max_requests_per_question":1}
        recorder = Recorder(folder / "ok", config, "fixture", folder / "stop.json")
        sent = []
        def fake(request, *a, **kw):
            sent.append(json.loads(request.data))
            return io.BytesIO(stream)
        call = recorded_opener(recorder, fake)
        response = json.loads(call(req).read())
        assert response["choices"][0]["message"]["content"] == "A"
        assert recorder.summary()["total"]["known_prompt_tokens"] == 100
        assert recorder.calls[0]["ttft_seconds"] is not None
        assert recorder.calls[0]["context_tokens"] == 100
        expect(BudgetExit, lambda: call(req))
        assert len(sent) == 1
        partial = Recorder(folder / "partial", cfg, "fixture", folder / "stop.json")
        partial.wrap(lambda *a, **k: io.BytesIO(b'{"usage":{"prompt_tokens":7}}'))(req)
        assert partial.summary()["total"]["known_prompt_tokens"] == 7
        assert partial.summary()["total"]["calls_without_usage"] == 1
        restricted = Recorder(folder / "images", {**cfg, "max_images_per_question":1}, "fixture", folder / "stop.json")
        expect(BudgetExit, lambda: restricted.wrap(fake)(req))
        assert not restricted.calls
        failed = Recorder(folder / "429", cfg, "fixture", folder / "stop.json")
        def fail(*a, **kw):
            raise HTTPError(req.full_url, 429, "fixture", {}, io.BytesIO(b'{"error":"fixture"}'))
        expect(HTTPError, lambda: failed.wrap(fail)(req))
        assert read(folder / "stop.json")["http_status"] == 429
        expect(RuntimeError, lambda: call(req))
        assert len(sent) == 1
        incomplete = stream.replace(b"data: [DONE]\n\n", b"")
        from transport import read_stream
        expect(RuntimeError, lambda: read_stream(lambda *a, **kw: io.BytesIO(incomplete), req))


def frame_count_and_transcripts():
    from experiments.semantic_delta_pilot.dashscope_flash_video_adapter import DashScopeFlashTransport
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        video = root / "fixture.mp4"
        video.write_bytes(b"offline fixture")
        cfg = read(ROOT / "config.example.json")
        client = DashScopeFlashTransport(base_url=cfg["base_url"])
        try:
            for count in (128, 250):
                with patch.dict(os.environ, {"PAVA_INITIAL_FRAMES":str(count)}), patch.object(client, "_frame_extractor", return_value=[b"x"]*count) as extract:
                    _, first = client.prepare_uniform_seed(video, 100, root / str(count))
                    _, second = client.prepare_uniform_seed(video, 100, root / str(count))
                    assert first["frame_count"] == count and second["frames_reused"]
                    assert extract.call_count == 1
        finally:
            client.close()
        file = root / "transcript.json"
        save(file, dict(segments=[dict(id=0, start=0, end=1, text="fixture")]))
        assert pava.imported_transcript(file, "fixture")["producer"] == "unknown"
        save(file, dict(segments=[], ground_truth_loaded=True))
        expect(ValueError, lambda: pava.imported_transcript(file, "fixture"))
        save(file, dict(segments=[], metadata=dict(answer="A")))
        expect(ValueError, lambda: pava.imported_transcript(file, "fixture"))


def reports():
    import report
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        run = work / "runs/fixture"
        save(run / "started.json", dict(work=str(work)))
        (run / "run.lock").touch()
        inputs = [dict(question_id="q1", video_id="v", dataset="MLVU"), dict(question_id="q2", video_id="v", dataset="MLVU")]
        save(run / "inputs.json", inputs)
        save(run / "config.json", read(ROOT / "config.example.json"))
        save(work / "materials/preparation/v.json", dict(preparation_seconds=30))
        items = [dict(**row, result=dict(status="completed" if i == 0 else "failed", selected_option="A"),
            usage=dict(total=dict(known_prompt_tokens=100, known_completion_tokens=10, requests=2, images=4, calls_without_usage=i)),
            online_qa_seconds=20, worker_wall_seconds=21) for i,row in enumerate(inputs)]
        save(run / "results.json", items)
        save(run / "finished.json", dict(all_workers_exited=True, phase="completed", completed=2, batch_wall_seconds=45))
        labels = work / "post_inference_fixture.json"
        save(labels, dict(q1="A", q2="A"))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = report.summarize(run, labels)
        assert summary["evaluation"]["overall"]["accuracy"] == .5
        assert summary["overall"]["metrics"]["total_tokens"]["total"] == 220
        assert summary["overall"]["calls_without_usage"] == 1
        assert summary["preprocessing"]["measured_video_preparation_seconds"] == 30
        assert summary["accounted_preparation_plus_online_seconds_per_question"] == 35
        save(run / "finished.json", dict(all_workers_exited=True, phase="stopped", completed=2, batch_wall_seconds=45))
        expect(ValueError, lambda: report.summarize(run, labels))


def worker_fixture():
    from experiments.semantic_delta_pilot import avp_integration as integration
    from experiments import run_videomme_mcq_ab as native
    cfg = {**read(ROOT / "config.example.json"), "stream":False}
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        run = work / "runs/fixture"
        save(run / "config.json", cfg)
        save(run / "started.json", dict(work=str(work)))
        save(work / "setup.json", dict(checkpoint="offline"))
        row = dict(question_id="fixture", video_id="v", dataset="MLVU", question="Offline fixture", options=["A. One", "B. Two"], video_path="fixture.mp4", map_path="map.json", use_asr=False)
        save(run / "inputs.json", [row])
        def fake(request, *a, **kw):
            return io.BytesIO(b'{"choices":[{"message":{"content":"B"}}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}')
        def one(**kwargs):
            assert set(kwargs["row"]) == set(QA_KEYS)
            assert kwargs["mode"] == "repair" and kwargs["max_rounds"] == 8
            assert kwargs["asr_cache_dir"].is_relative_to(run)
            integration.urlopen(Request(cfg["base_url"] + "/chat/completions", data=json.dumps(dict(model=cfg["model"], messages=[])).encode())).read()
            return dict(status="completed", selected_option="B")
        with patch.dict(os.environ, {"PAVA_MODEL":cfg["model"], "OPENAI_BASE_URL":cfg["base_url"]}), patch.object(integration, "urlopen", fake), patch.object(native, "run_one", one):
            pava.worker(run, "fixture")
        item = read(run / "questions/fixture/result.json")
        assert item["result"]["selected_option"] == "B" and item["usage"]["total"]["known_prompt_tokens"] == 12
        assert item["ground_truth_loaded"] is False


def coordinator():
    fixture = '''import json, sys, time
from pathlib import Path
run = Path(sys.argv[2]); qid = sys.argv[3]
row = next(r for r in json.loads((run/'inputs.json').read_text()) if r['question_id'] == qid)
out = run/'questions'/qid; out.mkdir(parents=True)
begin = time.monotonic()
if run.name == 'stop' and qid == 'q0':
    (run/'service_stop.json').write_text('{"http_status":429}')
time.sleep(1)
item = dict(question_id=qid, dataset=row['dataset'], video_id=row['video_id'], result=dict(status='completed', selected_option='A'), usage=None, online_qa_seconds=1, fixture_start=begin, fixture_end=time.monotonic())
(out/'result.json').write_text(json.dumps(item))
'''
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "pava.py").write_text(fixture)
        video = root / "fixture.mp4"
        video.write_bytes(b"offline")
        stat = video.stat()
        save(root / "materials/preparation/v.json", dict(video_path=str(video), video_bytes=stat.st_size, video_mtime_ns=stat.st_mtime_ns))
        save(root / "prepared.json", dict(initial_frames=128))
        rows = [dict(question_id=f"q{i}", video_id="v", dataset=d) for i,d in enumerate(DATASETS*2)]
        save(root / "inputs.json", rows)
        cfg = read(ROOT / "config.example.json")
        for invalid in (0, -1, 1.5, True):
            save(root / "config.json", {**cfg, "qa_concurrency":invalid})
            expect(ValueError, lambda: pava.configure(root / "config.json"))
        slots = 6
        for name, override in (("configured", None), ("override", slots), ("stop", slots)):
            save(root / "config.json", {**cfg, "qa_concurrency":slots if override is None else 1})
            args = SimpleNamespace(config=root / "config.json", work=root, per_dataset=None, concurrency=override, execute=True, run_id=name)
            with patch.object(pava, "ROOT", root), patch.object(pava, "validate_materials"), patch.dict(os.environ, {"PAVA_MODEL":cfg["model"], "OPENAI_BASE_URL":cfg["base_url"]}), contextlib.redirect_stdout(io.StringIO()):
                pava.run_batch(args)
            items = read(root / "runs" / name / "results.json")
            if name != "stop":
                assert len(items) == 8
                events = sorted([(r["fixture_start"], 1) for r in items] + [(r["fixture_end"], -1) for r in items])
                active = peak = 0
                for _, delta in events:
                    active += delta
                    peak = max(peak, active)
                assert peak == slots
            else:
                assert 1 <= len(items) <= slots and read(root / "runs/stop/finished.json")["phase"] == "stopped"


if __name__ == "__main__":
    if sys.platform != "linux":
        raise RuntimeError("Use Ubuntu Linux")
    if "--worker-fixture" in sys.argv:
        worker_fixture()
        print("PASS worker integration fixture; 0 network requests")
    else:
        for test in (source_and_roster, transport_and_usage, frame_count_and_transcripts, reports, coordinator):
            test()
            print("PASS", test.__name__, flush=True)
        subprocess.run([sys.executable, "-B", __file__, "--worker-fixture"], check=True)
        print("ALL OFFLINE CHECKS PASSED; 0 model/network requests", flush=True)
