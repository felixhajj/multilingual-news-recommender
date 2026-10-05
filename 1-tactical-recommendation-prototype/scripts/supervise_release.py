"""Run one detached release worker and record its real exit status and logs."""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from src.portfolio_config import ARTIFACTS, ROOT, read_json, read_jsonl, write_json, write_jsonl


def main(hours=4.5, limit=300):
    folder = ARTIFACTS / "phase4"
    logs = folder / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    previous = read_json(folder / "process.json")
    if previous:
        history_path = folder / "process_history.jsonl"
        history = read_jsonl(history_path) if history_path.exists() else []
        if not any(row["started_at"] == previous["started_at"] for row in history):
            write_jsonl(history_path, history + [previous])
    temporary = ROOT / ".tmp" / "release-worker"
    temporary.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONFAULTHANDLER": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NEWS_CPU_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "NEWS_SAFETENSORS_BACKEND": "pread", "NEWS_EMBEDDING_DEVICE": "cpu",
           "TEMP": str(temporary), "TMP": str(temporary), "TMPDIR": str(temporary)}
    command = [sys.executable, "-u", str(ROOT / "scripts/build_news_release.py"),
               "--limit", str(limit), "--hours", str(hours)]
    with (logs / f"{stamp}.out.log").open("w", encoding="utf-8") as out, \
         (logs / f"{stamp}.err.log").open("w", encoding="utf-8") as err:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        state = {"supervisor_pid": os.getpid(), "worker_pid": worker.pid,
                 "started_at": started.isoformat(), "status": "running", "command": command,
                 "stdout": str(logs / f"{stamp}.out.log"), "stderr": str(logs / f"{stamp}.err.log")}
        write_json(folder / "process.json", state)
        code = worker.wait()
    state.update(status="exited", return_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(folder / "process.json", state)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=4.5)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    (ARTIFACTS / "phase4").mkdir(parents=True, exist_ok=True)
    with FileLock(ARTIFACTS / "phase4/.supervisor.lock").acquire(timeout=0):
        raise SystemExit(main(args.hours, args.limit))
