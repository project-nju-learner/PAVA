"""Summarize actual usage; optionally score only after the complete run has ended."""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
from pathlib import Path
import statistics

from common import read, save


def distribution(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return dict(count=0, total=0, mean=None, median=None, p95=None, maximum=None)
    index = (len(values)-1)*.95
    low = int(index)
    high = min(low+1, len(values)-1)
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=values[low]+(index-low)*(values[high]-values[low]), maximum=values[-1])


def aggregate(rows):
    keys = ("input_tokens", "output_tokens", "total_tokens", "requests", "image_submissions",
            "online_qa_seconds", "worker_wall_seconds", "max_context_tokens")
    return dict(questions=len(rows), failed=sum(r["status"] == "failed" for r in rows),
                unresolved=sum(r["status"] == "unresolved" for r in rows),
                calls_without_usage=sum(r["calls_without_usage"] for r in rows),
                metrics={k:distribution([r[k] for r in rows]) for k in keys})


def summarize(run, labels=None):
    finish = read(run / "finished.json")
    if not finish["all_workers_exited"]:
        raise ValueError("Workers are still active")
    with (run / "run.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        items = read(run / "results.json") if (run / "results.json").exists() else []
        inputs = read(run / "inputs.json")
        if len(items) != len({x["question_id"] for x in items}):
            raise ValueError("Duplicate results")
        if finish["completed"] != len(items) or not {x["question_id"] for x in items} <= {x["question_id"] for x in inputs}:
            raise ValueError("Result IDs/count do not match the finished batch")
        if labels and (finish["phase"] != "completed" or len(items) != len(inputs)):
            raise ValueError("Labels must remain closed until all planned questions have ended")
        rows, all_calls, stages = [], [], defaultdict(lambda: defaultdict(int))
        for item in items:
            qid = item["question_id"]
            usage = item.get("usage") or {}
            total = usage.get("total") or {}
            file = run / "questions" / qid / "http/calls.jsonl"
            calls = [json.loads(line) for line in file.read_text().splitlines() if line.strip()] if file.exists() else []
            ends = [c for c in calls if c.get("event") == "END"]
            all_calls.extend(ends)
            for stage, values in usage.get("stages", {}).items():
                for key in ("requests", "known_prompt_tokens", "known_completion_tokens", "calls_without_usage", "images"):
                    stages[stage][key] += values.get(key, 0)
            input_tokens = total.get("known_prompt_tokens", 0)
            output_tokens = total.get("known_completion_tokens", 0)
            contexts = [c["prompt_tokens"] for c in ends if c.get("prompt_tokens") is not None]
            final = item["result"].get("result", {}).get("final", {})
            rows.append(dict(question_id=qid, dataset=item["dataset"], video_id=item["video_id"],
                status=item["result"]["status"], prediction=item["result"].get("selected_option") or final.get("selected_option"),
                input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens+output_tokens,
                requests=total.get("requests", 0), calls_without_usage=total.get("calls_without_usage", 0),
                image_submissions=total.get("images", 0), online_qa_seconds=item.get("online_qa_seconds"),
                worker_wall_seconds=item.get("worker_wall_seconds"), max_context_tokens=max(contexts) if contexts else None,
                budget_exit=bool(usage.get("budget_exhausted") or final.get("forced_at_budget") or final.get("answer_mode") == "budget_forced")))
        summary = dict(planned=len(inputs), finished=len(rows), phase=finish["phase"], config=read(run / "config.json"),
            overall=aggregate(rows), by_dataset={d:aggregate([r for r in rows if r["dataset"] == d]) for d in dict.fromkeys(r["dataset"] for r in rows)},
            stages=dict(stages), batch_wall_seconds=finish["batch_wall_seconds"],
            per_call={k:distribution([c.get(k) for c in all_calls]) for k in ("elapsed_seconds", "prompt_tokens", "completion_tokens", "ttft_seconds", "approximate_decode_tokens_per_second")},
            tokens_are_provider_reported=True, missing_usage_is_not_zero_cost=True,
            online_qa_excludes_preprocessing_and_model_server_loading=True, images_are_submissions_not_unique_frames=True)
        work = Path(read(run / "started.json")["work"])
        prep = []
        counts = defaultdict(int)
        for item in inputs:
            counts[item["video_id"]] += 1
        for vid in counts:
            file = work / "materials/preparation" / (vid + ".json")
            if file.is_file():
                prep.append(read(file))
        cold = sum(r["preparation_seconds"] for r in prep)
        summary["preprocessing"] = dict(unique_videos=len(counts), videos_with_measurement=len(prep),
            measured_video_preparation_seconds=cold,
            amortized_seconds_per_planned_question=cold/len(inputs) if len(prep) == len(counts) and inputs else None,
            stage_seconds={k:distribution([r.get(k) for r in prep]) for k in ("index_decode_seconds", "index_encode_seconds", "index_total_seconds", "initial_frames_seconds", "transcript_preparation_seconds", "index_model_load_seconds", "asr_model_load_seconds")},
            imported_transcript_generation_cost="unknown unless separately supplied",
            model_server_loading_seconds=None,
            note="Preparation times include any local index/ASR model loading in that prepare session; component times overlap and must not be added to the total again. Original measured times remain attached to each video when reused; cache hits are not zero cold cost.")
        if rows and finish["phase"] == "completed" and len(prep) == len(counts) and all(r["online_qa_seconds"] is not None for r in rows):
            summary["accounted_preparation_plus_online_seconds_per_question"] = (cold+sum(r["online_qa_seconds"] for r in rows))/len(rows)
            summary["accounting_note"] = "Amortized accounting, not a newly measured cold wall-clock run; excludes model-server startup and unknown imported-transcript generation."
        if labels:
            answers = read(labels)
            if any(r["question_id"] not in answers for r in rows):
                raise ValueError("Official labels missing for some questions")
            for row in rows:
                row["correct"] = row["status"] == "completed" and row["prediction"] == answers[row["question_id"]]
            summary["evaluation"] = {d:dict(correct=sum(r["correct"] for r in group), questions=len(group), accuracy=sum(r["correct"] for r in group)/len(group)) for d, group in [("overall", rows), *[(d, [r for r in rows if r["dataset"] == d]) for d in dict.fromkeys(r["dataset"] for r in rows)]]}
            summary["evaluation"]["denominator"] = "All planned questions, including failed/unresolved; separate failure counts above."
        save(run / "cost_summary.json", summary)
        if rows:
            with (run / "question_costs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--labels", type=Path, help="Optional isolated JSON mapping question_id to official answer letter")
    args = parser.parse_args()
    summarize(args.run.resolve(strict=True), args.labels)
