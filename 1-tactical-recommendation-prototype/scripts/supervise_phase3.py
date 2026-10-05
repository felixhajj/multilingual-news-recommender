"""Lightweight supervisor records native worker crashes as well as Python errors."""
import os
import argparse
import subprocess
import sys
from filelock import FileLock
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, ROOT, read_json, read_jsonl, write_json, write_jsonl


def main(worker_arguments=()):
    now = datetime.now(timezone.utc)
    directory = ARTIFACTS / "phase3"
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S-supervised")
    old_process = read_json(directory / "process.json")
    history_path = directory / "process_history.jsonl"
    if old_process:
        history = read_jsonl(history_path) if history_path.exists() else []
        if not any(row["started_at"] == old_process["started_at"] for row in history):
            write_jsonl(history_path, history + [old_process])
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONFAULTHANDLER": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NEWS_CPU_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    env["NEWS_SAFETENSORS_BACKEND"] = "pread"
    command = [sys.executable, "-u", str(ROOT / "scripts/run_phase3_job.py"), "--hours", "3", *worker_arguments]
    with (logs / f"{stamp}.out.log").open("w", encoding="utf-8") as out, \
         (logs / f"{stamp}.err.log").open("w", encoding="utf-8") as err:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        process = {"supervisor_pid": os.getpid(), "worker_pid": worker.pid, "started_at": now.isoformat(),
                   "command": command, "logs": stamp, "status": "running"}
        process["checkpoint_backend"] = "pread"
        write_json(directory / "process.json", process)
        code = worker.wait()
    process.update(status="exited", return_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / "process.json", process)
    state = read_json(directory / "job.json", {})
    if code != 0:
        state.update(status="needs_attention", error=state.get("error") or f"Worker exited {code}; inspect {stamp}.err.log",
                     updated_at=datetime.now(timezone.utc).isoformat())
        write_json(directory / "job.json", state)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extend-hours", type=float, default=0)
    parser.add_argument("--allocation-id")
    args = parser.parse_args()
    worker_arguments = ["--extend-hours", str(args.extend_hours)]
    if args.allocation_id:
        worker_arguments += ["--allocation-id", args.allocation_id]
    with FileLock(ARTIFACTS / "phase3/.supervisor.lock").acquire(timeout=0):
        raise SystemExit(main(worker_arguments))
