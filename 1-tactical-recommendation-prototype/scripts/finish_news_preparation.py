"""Wait for the active training job, then serially finish bounded local preparation."""
import subprocess
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, ROOT, GPU_LOCK, read_json, write_json
from src.portfolio_reports import publish_evidence_snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Preserve completed queue stages after an interruption")
    args = parser.parse_args()
    from filelock import FileLock
    queue_lock = FileLock(ARTIFACTS / ".preparation_queue.lock")
    queue_lock.acquire(timeout=0)
    status_path = ARTIFACTS / "preparation_queue.json"
    status = read_json(status_path, {}) if args.resume else {}
    interrupted_command = status.get("current_command")
    status.setdefault("steps", [])
    status["status"] = "waiting_for_active_gpu_job"
    write_json(status_path, status)
    release_seconds_remaining = 3 * 3600
    with GPU_LOCK:
        if args.resume and interrupted_command and interrupted_command[0] == "scripts/build_news_release.py":
            release = read_json(ARTIFACTS / "release_manifest.json", {})
            if release.get("status") == "preparing":
                from src.portfolio_config import digest
                budget = read_json(ARTIFACTS / "processing_budget.json", {"consumed_seconds": 0, "limit_seconds": 86400})
                recovery_id = digest(release)
                if recovery_id not in budget.get("recovered_release_runs", []):
                    elapsed = release.get("elapsed_seconds", 0)
                    budget["consumed_seconds"] += elapsed
                    budget.setdefault("recovered_release_runs", []).append(recovery_id)
                    budget["recovery_note"] = "Interrupted indexing time recovered from last durable progress; completed training preserved"
                    status.setdefault("recovered_indexing_seconds", 0)
                    status["recovered_indexing_seconds"] += elapsed
                    write_json(ARTIFACTS / "processing_budget.json", budget)
                release_seconds_remaining = max(0, 3*3600-status.get("recovered_indexing_seconds", 0))
                write_json(status_path, status)
        if args.train_first:
            preparation = read_json(ARTIFACTS / "training_preparation.json", {})
            budget = read_json(ARTIFACTS / "processing_budget.json", {"consumed_seconds": 0, "limit_seconds": 86400})
            from src.portfolio_config import digest
            recovery_id = digest(preparation)
            if preparation.get("status") == "running" and recovery_id not in budget.get("recovered_preparation_runs", []):
                # A terminated process cannot execute its finally block. Account for the last durable progress timestamp.
                budget["consumed_seconds"] += preparation.get("seconds", 0)
                budget.setdefault("recovered_preparation_runs", []).append(recovery_id)
                budget["recovery_note"] = "Interrupted labeling time recovered from its last saved progress; not a successful training run"
                write_json(ARTIFACTS / "processing_budget.json", budget)
    commands = ([["scripts/run_news_experiments.py", "--examples", "100", "--steps", "24", "--label-hours", "2.5",
                  "--sft-hours", "1", "--domain-hours", "1"]] if args.train_first else []) + [
        ["scripts/evaluate_news.py", "predict", "--hours", "2"],
        ["scripts/build_news_release.py", "--limit", "300", "--hours", str(release_seconds_remaining/3600)],
        ["scripts/prepare_ranking_review.py"],
        ["scripts/build_news_site.py"],
    ]
    for command in commands:
        if args.resume and any(step["command"][0] == command[0] and step["exit_code"] == 0 for step in status["steps"]):
            print("Already finished; preserving", command[0], flush=True)
            continue
        if command[0].endswith("prepare_ranking_review.py") and (ARTIFACTS / "review" / "ranking.jsonl").exists():
            continue
        status.update(status="running", current_command=command)
        write_json(status_path, status)
        started = time.monotonic()
        print("Running", command, flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        status["steps"].append({"command": command, "exit_code": result.returncode,
                                "elapsed_seconds": time.monotonic()-started})
        publish_evidence_snapshot()
    status.update(status="local_stages_finished_review_required" if all(s["exit_code"] == 0 for s in status["steps"])
                  else "local_stages_finished_with_failures", current_command=None)
    write_json(status_path, status)
    publish_evidence_snapshot()
    print("Preparation stopped. Human review and validated publishing remain explicit gates.", flush=True)
