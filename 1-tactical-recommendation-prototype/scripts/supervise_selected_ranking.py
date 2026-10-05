"""Launch and durably track the bounded selected-pipeline ranking evaluation."""
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock
from src.portfolio_config import ARTIFACTS, ROOT, read_json, read_jsonl, write_json, write_jsonl


def process_exists(pid):
    if not pid:
        return False
    try:
        result = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True, check=False)
        return result.returncode == 0 and f'"{int(pid)}"' in result.stdout
    except (OSError, ValueError):
        return False


def main():
    directory = ARTIFACTS / "phase4"
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = directory / "selected_ranking_process.json"
    previous = read_json(state_path, {})
    if previous.get("status") == "running" and process_exists(previous.get("worker_pid")):
        raise RuntimeError("Selected-pipeline ranking evaluation is already running")
    if previous:
        history_path = directory / "selected_ranking_process_history.jsonl"
        history = read_jsonl(history_path) if history_path.exists() else []
        if not any(row.get("started_at") == previous.get("started_at") for row in history):
            write_jsonl(history_path, history + [previous])
    temp = ROOT / ".tmp" / "phase4-ranking"
    temp.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    stdout_path, stderr_path = logs / f"ranking-{stamp}.out.log", logs / f"ranking-{stamp}.err.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONFAULTHANDLER": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NEWS_CPU_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "NEWS_SAFETENSORS_BACKEND": "pread", "NEWS_EMBEDDING_DEVICE": "cpu",
           "TEMP": str(temp), "TMP": str(temp), "TMPDIR": str(temp)}
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_selected_ranking.py"), "--hours", "1"]
    with FileLock(directory / ".selected-ranking-supervisor.lock").acquire(timeout=0):
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state = {"status": "running", "supervisor_pid": os.getpid(), "worker_pid": worker.pid,
                     "started_at": started.isoformat(), "command": command,
                     "stdout": str(stdout_path), "stderr": str(stderr_path),
                     "temporary_directory": str(temp)}
            write_json(state_path, state)
            code = worker.wait()
        state.update(status="exited", return_code=code,
                     finished_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
