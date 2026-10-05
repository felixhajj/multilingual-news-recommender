"""Run resumable notebook checks within Phase 4's approved 90-minute budget."""
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock
from src.portfolio_config import ARTIFACTS, ROOT, read_json, read_jsonl, write_json, write_jsonl

ALLOCATION_ID = "phase4-app-notebooks-20261001"
ALLOWED_SECONDS = 5400


def process_exists(pid):
    if not pid:
        return False
    result = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and f'"{int(pid)}"' in result.stdout


def main():
    phase = ARTIFACTS / "phase4"
    logs = phase / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = phase / "notebook_validation_process.json"
    previous = read_json(state_path, {})
    if previous.get("status") == "running" and process_exists(previous.get("worker_pid")):
        raise RuntimeError("Notebook validation is already running")
    if previous:
        history_path = phase / "notebook_validation_process_history.jsonl"
        history = read_jsonl(history_path) if history_path.exists() else []
        if not any(row.get("started_at") == previous.get("started_at") for row in history):
            write_jsonl(history_path, history + [previous])

    budget_path = ARTIFACTS / "processing_budget.json"
    budget = read_json(budget_path)
    if not budget or budget["consumed_seconds"] > budget["limit_seconds"]:
        raise RuntimeError("Processing budget is unavailable or already over its cap")
    allocations = budget.setdefault("allocations", [])
    matches = [row for row in allocations if row.get("allocation_id") == ALLOCATION_ID]
    if len(matches) > 1:
        raise RuntimeError("Duplicate notebook-validation budget allocation")
    allocation = matches[0] if matches else {
        "phase": 4, "operation": "application_and_notebook_checks",
        "allocation_id": ALLOCATION_ID, "allowed_seconds": ALLOWED_SECONDS,
        "charged_seconds": 0.0, "accounting": "actual elapsed time, incrementally persisted"}
    if not matches:
        allocations.append(allocation)
        write_json(budget_path, budget)
    if allocation.get("allowed_seconds") != ALLOWED_SECONDS:
        raise RuntimeError("Notebook-validation allocation has unexpected limits")
    remaining = min(ALLOWED_SECONDS - allocation.get("charged_seconds", 0.0),
                    budget["limit_seconds"] - budget["consumed_seconds"])
    if remaining <= 0:
        raise RuntimeError("No Phase 4 notebook-validation budget remains")

    temp = ROOT / ".tmp" / "phase4-notebooks"
    temp.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    session_started = time.monotonic()
    last_charged = 0.0
    stamp = started.strftime("%Y%m%d-%H%M%S")
    stdout_path, stderr_path = logs / f"notebooks-{stamp}.out.log", logs / f"notebooks-{stamp}.err.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONFAULTHANDLER": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false", "NEWS_EMBEDDING_DEVICE": "cpu",
           "TEMP": str(temp), "TMP": str(temp), "TMPDIR": str(temp)}
    command = [sys.executable, "-u", str(ROOT / "scripts/validate_news_notebooks.py")]

    def charge_and_record(state, child):
        nonlocal last_charged
        elapsed = min(time.monotonic() - session_started, remaining)
        delta = elapsed - last_charged
        if delta > 0:
            current = read_json(budget_path)
            target = next(row for row in current["allocations"] if row.get("allocation_id") == ALLOCATION_ID)
            if (target["charged_seconds"] + delta > ALLOWED_SECONDS + 1e-6
                    or current["consumed_seconds"] + delta > current["limit_seconds"] + 1e-6):
                raise RuntimeError("Notebook validation reached the cumulative processing cap")
            target["charged_seconds"] = round(target["charged_seconds"] + delta, 3)
            current["consumed_seconds"] += delta
            write_json(budget_path, current)
            last_charged = elapsed
        state["elapsed_seconds"] = round(elapsed, 3)
        state["worker_pid"] = child.pid
        write_json(state_path, state)

    with FileLock(phase / ".notebook-validation-supervisor.lock").acquire(timeout=0):
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state = {"status": "running", "supervisor_pid": os.getpid(), "worker_pid": child.pid,
                     "started_at": started.isoformat(), "command": command,
                     "stdout": str(stdout_path), "stderr": str(stderr_path),
                     "temporary_directory": str(temp), "budget_allocation_id": ALLOCATION_ID}
            write_json(state_path, state)
            while True:
                elapsed = time.monotonic() - session_started
                time_left = remaining - elapsed
                if time_left <= 0:
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   capture_output=True, check=False)
                    code = child.wait()
                    state.update(status="timed_out", return_code=code)
                    break
                try:
                    code = child.wait(timeout=min(15, time_left))
                    state.update(status="exited", return_code=code)
                    break
                except subprocess.TimeoutExpired:
                    charge_and_record(state, child)
            charge_and_record(state, child)
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
