"""Meter and supervise the resumable Phase 4 live-app check."""
import json
import os
import subprocess
import sys
import time
import urllib.request
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


def send_completion_notice(report):
    status = report.get("status", "unknown")
    count = (report.get("article_smoke") or {}).get("successful", 0)
    message = f"Phase 4 live app check finished: {status}; {count} smoke articles saved on D:."
    request = urllib.request.Request("https://ntfy.sh/pigeons313_task_done", data=message.encode("utf-8"),
                                     headers={"Title": "Codex Task Complete"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"sent": response.status < 300, "status": response.status}
    except Exception as error:
        return {"sent": False, "error": str(error)}


def main():
    phase = ARTIFACTS / "phase4"
    out = phase / "live_app_check"
    logs = phase / "logs"
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    state_path = phase / "live_app_check_process.json"
    previous = read_json(state_path, {})
    if previous.get("status") == "running" and process_exists(previous.get("worker_pid")):
        raise RuntimeError("The Phase 4 live-app check is already running")
    if previous:
        history_path = phase / "live_app_check_process_history.jsonl"
        history = read_jsonl(history_path) if history_path.exists() else []
        if not any(row.get("started_at") == previous.get("started_at") for row in history):
            write_jsonl(history_path, history + [previous])

    budget_path = ARTIFACTS / "processing_budget.json"
    budget = read_json(budget_path)
    if not budget or budget["consumed_seconds"] > budget["limit_seconds"]:
        raise RuntimeError("Processing budget is unavailable or already over its cap")
    allocations = budget.setdefault("allocations", [])
    matches = [row for row in allocations if row.get("allocation_id") == ALLOCATION_ID]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one preapproved Phase 4 application-check allocation")
    allocation = matches[0]
    if allocation.get("allowed_seconds") != ALLOWED_SECONDS:
        raise RuntimeError("Application-check allocation has an unexpected limit")
    remaining = min(ALLOWED_SECONDS - allocation.get("charged_seconds", 0.0),
                    budget["limit_seconds"] - budget["consumed_seconds"])
    if remaining <= 0:
        raise RuntimeError("No approved Phase 4 application-check budget remains")

    temp = ROOT / ".tmp" / "phase4-live-app-check"
    temp.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    session_started = time.monotonic()
    last_charged = 0.0
    stamp = started.strftime("%Y%m%d-%H%M%S")
    stdout_path, stderr_path = logs / f"live-app-{stamp}.out.log", logs / f"live-app-{stamp}.err.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONFAULTHANDLER": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false", "NEWS_EMBEDDING_DEVICE": "cpu",
           "GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TEMP": str(temp), "TMP": str(temp), "TMPDIR": str(temp)}
    command = [sys.executable, "-u", str(ROOT / "scripts/phase4_live_app_check.py")]

    def charge_and_record(state, child):
        nonlocal last_charged
        elapsed = min(time.monotonic() - session_started, remaining)
        delta = elapsed - last_charged
        if delta > 0:
            with FileLock(ARTIFACTS / ".processing-budget.lock").acquire(timeout=30):
                current = read_json(budget_path)
                target = next(row for row in current["allocations"] if row.get("allocation_id") == ALLOCATION_ID)
                if (target["charged_seconds"] + delta > ALLOWED_SECONDS + 1e-6
                        or current["consumed_seconds"] + delta > current["limit_seconds"] + 1e-6):
                    raise RuntimeError("Live-app validation reached the cumulative processing cap")
                target["charged_seconds"] = round(target["charged_seconds"] + delta, 3)
                current["consumed_seconds"] += delta
                write_json(budget_path, current)
            last_charged = elapsed
        state["elapsed_seconds"] = round(elapsed, 3)
        state["worker_pid"] = child.pid
        write_json(state_path, state)

    with FileLock(phase / ".live-app-check-supervisor.lock").acquire(timeout=0):
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state = {"status": "running", "supervisor_pid": os.getpid(), "worker_pid": child.pid,
                     "started_at": started.isoformat(), "command": command,
                     "stdout": str(stdout_path), "stderr": str(stderr_path),
                     "temporary_directory": str(temp), "budget_allocation_id": ALLOCATION_ID,
                     "allowed_seconds_this_run": round(remaining, 3)}
            write_json(state_path, state)
            while True:
                time_left = remaining - (time.monotonic() - session_started)
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
        report = read_json(out / "report.json", {})
        state["report_status"] = report.get("status")
        if report.get("status") in {"passed", "completed_with_model_failures", "completed_with_open_checks"}:
            state["notification"] = send_completion_notice(report)
        else:
            state["notification"] = {"sent": False, "reason": "No completion notice for an incomplete run"}
        write_json(state_path, state)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
