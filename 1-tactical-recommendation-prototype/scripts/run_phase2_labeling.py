"""Run only the resumable Phase 2 weak-label preparation job; never train adapters."""
import argparse
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news_pipeline import QwenBackend
from src.news_training import prepare_training_examples
from src.portfolio_config import ARTIFACTS, GPU_LOCK, read_json, read_jsonl, write_json


def recover_interrupted_job(budget, status_path):
    status = read_json(status_path, {})
    if status.get("status") != "running" or not status.get("job_id"):
        return
    recovered = budget.setdefault("recovered_phase2_jobs", [])
    if status["job_id"] in recovered:
        return
    started = status["started_at_epoch"]
    durable_seconds = 0.0
    for path in (ARTIFACTS / "labeling").glob("*/attempts.jsonl"):
        for attempt in read_jsonl(path):
            completed = datetime.fromisoformat(attempt["completed_at"]).timestamp()
            if completed >= started:
                durable_seconds += attempt.get("elapsed_seconds", 0)
    budget["consumed_seconds"] += durable_seconds
    recovered.append(status["job_id"])
    status.update(status="interrupted_recovered", recovered_seconds=durable_seconds,
                  recovery_note="Durable completed-attempt time only; outage tail and model load remain unmeasured")
    write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=60,
                        help="Total baseline plus new weak labels; maximum 500")
    parser.add_argument("--hours", type=float, default=3.0)
    args = parser.parse_args()
    if not 20 <= args.target <= 500:
        raise ValueError("Target must preserve the 20-example usable Phase 2 seed and cannot exceed 500")
    if not 0 < args.hours <= 6:
        raise ValueError("One Phase 2 invocation is limited to 6 hours")
    queue_path = ARTIFACTS / "phase2/labeling_candidates.jsonl"
    if not queue_path.exists():
        raise ValueError("Run scripts/prepare_phase2_data.py first")
    corpus = read_jsonl(ARTIFACTS / "corpus.jsonl")
    queue = read_jsonl(queue_path)
    budget_path = ARTIFACTS / "processing_budget.json"
    budget = read_json(budget_path, {"consumed_seconds": 0, "limit_seconds": 86400})
    status_path = ARTIFACTS / "phase2/labeling_job.json"
    recover_interrupted_job(budget, status_path)
    write_json(budget_path, budget)
    remaining = max(0, budget["limit_seconds"] - budget["consumed_seconds"])
    # Keep ten nominal ledger hours for validation, any justified retraining and re-indexing.
    available = max(0, remaining - 10 * 3600)
    seconds = min(args.hours * 3600, available)
    if seconds < 60:
        raise RuntimeError("No Phase 2 labeling budget remains after the ten-hour downstream reserve")
    started = time.monotonic()
    epoch = time.time()
    job_id = f"phase2-{int(epoch)}-{args.target}"
    write_json(status_path, {"status": "running", "job_id": job_id, "pid": os.getpid(), "target": args.target,
                             "budget_seconds": seconds, "started_at_epoch": epoch,
                             "note": "Machine-assisted weak labels; this job does not train or inspect held-out predictions."})
    try:
        # The release path uses larger chunks. Weak-label preparation uses a smaller,
        # explicitly versioned generation envelope to fit the 11 GB Pascal GPU.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        backend = QwenBackend(mode="base", max_new_tokens=256, chunk_tokens=512, chunk_stride=480)
        examples = prepare_training_examples(corpus, backend, target=args.target,
                                              seconds=seconds, candidate_queue=queue)
        status = "target_reached" if len(examples) >= args.target else "queue_or_time_limit_reached"
        write_json(status_path, {"status": status, "job_id": job_id, "target": args.target,
                                 "total_examples": len(examples),
                                 "elapsed_seconds": round(time.monotonic() - started, 3)})
        print(json.dumps(read_json(status_path), indent=2))
    except BaseException as exc:
        write_json(status_path, {"status": "failed_or_interrupted", "job_id": job_id, "target": args.target,
                                 "elapsed_seconds": round(time.monotonic() - started, 3),
                                 "error": f"{type(exc).__name__}: {exc}"[:1000]})
        raise
    finally:
        budget["consumed_seconds"] += time.monotonic() - started
        budget.setdefault("allocations", []).append({"phase": 2, "operation": "weak_label_preparation",
                                                       "target": args.target,
                                                       "elapsed_seconds": round(time.monotonic() - started, 3)})
        write_json(budget_path, budget)


if __name__ == "__main__":
    with GPU_LOCK:
        main()
