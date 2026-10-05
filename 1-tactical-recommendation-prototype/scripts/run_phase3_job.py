"""Resumable v3 labeling/training/prediction job; final-test scores stay locked."""
import argparse
import gc
import json
import sys
import subprocess
import traceback
from contextlib import redirect_stdout, redirect_stderr
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finish_phase2 import verify_completion
from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3
from src.llm_extractor import validate_extraction_response
from src.news_evaluation import review_roles
from src.news_pipeline import QwenBackend
from src.news_selection import score_extraction
from src.news_training import build_labeling_queue
from src.news_training_v3 import train_v3
from src.phase3_resources import check_storage
from src.weak_label_recovery import POLICY, validate_weak_label
from src.portfolio_config import (ARTIFACTS, BASE_REVISION, GPU_LOCK, digest,
                                  file_digest, read_json, read_jsonl, write_json, write_jsonl)

RUNS = ("base-v3", "extraction-only-v3", "domain-extraction-v3")
FOLDER = ARTIFACTS / "phase3"


def release_gpu():
    gc.collect()
    import torch
    torch.cuda.empty_cache()


accept_weak_label = validate_weak_label


def reserve_budget(hours, extension_hours=0, allocation_id=None):
    state = read_json(FOLDER / "job.json")
    if state:
        if extension_hours:
            if not allocation_id or Path(allocation_id).name != allocation_id:
                raise ValueError("An explicit stable allocation ID is required to extend the budget")
            if state.get("status") == "predictions_ready_validation_only":
                return state
            budget_path = ARTIFACTS / "processing_budget.json"
            budget = read_json(budget_path)
            previous = next((r for r in budget["allocations"] if r.get("allocation_id") == allocation_id), None)
            if previous is None:
                charge = extension_hours * 3600
                if budget["consumed_seconds"] + charge > budget["limit_seconds"]:
                    raise ValueError("Cumulative budget exhausted; extension refused")
                now = datetime.now(timezone.utc)
                deadline = max(now, datetime.fromisoformat(state["deadline"])) + timedelta(seconds=charge)
                previous = {"phase": 3, "operation": "resume_v3_experiments", "allocation_id": allocation_id,
                            "charged_seconds": charge, "deadline": deadline.isoformat(),
                            "accounting": "additional prepaid allowance; old ledger and attempts are preserved"}
                budget["consumed_seconds"] += charge
                budget["allocations"].append(previous)
                write_json(budget_path, budget)
            elif previous["charged_seconds"] != extension_hours * 3600:
                raise ValueError("Allocation ID already belongs to a different allowance")
            if allocation_id not in state.get("extension_ids", []):
                state.setdefault("extension_ids", []).append(allocation_id)
                state["charged_seconds"] += previous["charged_seconds"]
                state["deadline"] = previous["deadline"]
                write_json(FOLDER / "job.json", state)
        return state
    budget_path = ARTIFACTS / "processing_budget.json"
    budget = read_json(budget_path)
    # Unknown historical preparation/linking overhead is reserved, not claimed as measured.
    guard = 14400 if not budget.get("phase3_historical_guard_seconds") else 0
    allocation = hours * 3600
    if budget["consumed_seconds"] + guard + allocation > budget["limit_seconds"]:
        raise ValueError("Insufficient cumulative budget; do not reset the ledger")
    budget["consumed_seconds"] += guard + allocation
    budget["phase3_historical_guard_seconds"] = guard or budget["phase3_historical_guard_seconds"]
    budget.setdefault("allocations", []).append({"phase": 3, "operation": "v3_experiments",
        "charged_seconds": allocation, "historical_uncertainty_guard_seconds": guard,
        "accounting": "conservative prepaid wall-clock allowance; not an actual-runtime claim"})
    now = datetime.now(timezone.utc)
    state = {"status": "starting", "started_at": now.isoformat(),
             "deadline": (now + timedelta(hours=hours)).isoformat(), "charged_seconds": allocation,
             "target_labels": 80, "steps": 48, "observed_seconds": 0,
             "policy": "No automatic promotion or final-test scoring. Completed stages are reused."}
    # Reserve first: a crash can overcount time, never silently reset it.
    write_json(budget_path, budget)
    write_json(FOLDER / "job.json", state)
    return state


def prepare_labels(remaining, update):
    from scripts.recover_phase3_labels import collect_recovery, recover
    selected = read_json(FOLDER / "training_dataset.json")
    if selected:
        path = ARTIFACTS / selected["path"]
        if not path.resolve().is_relative_to(ARTIFACTS.resolve()) or file_digest(path) != selected["sha256"]:
            raise ValueError("Selected training dataset changed")
        if file_digest(path.parent / "manifest.json") != selected["manifest_sha256"]:
            raise ValueError("Selected dataset manifest changed")
        return read_jsonl(path)
    directory = FOLDER / "weak_labels_v3"
    dataset_path = directory / "examples.jsonl"
    completed = read_json(directory / "manifest.json")
    if completed:
        if file_digest(dataset_path) != completed["examples_sha256"]:
            raise ValueError("Completed weak dataset changed")
        return read_jsonl(dataset_path)
    records = read_jsonl(ARTIFACTS / "corpus.jsonl")
    frozen = read_json(ARTIFACTS / "review/frozen_extraction_manifest.json")
    reviews = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
    excluded_ids = {r["article"]["article_id"] for r in reviews}
    excluded_groups = set(frozen["group_ids"]) | {r["group_id"] for r in records if r["article_id"] in excluded_ids}
    candidates = build_labeling_queue([r for r in records if len(r["body"]) <= 1600], [], excluded_groups, limit=160)
    backend = QwenBackend(mode="base", schema_version=EXTRACTION_SCHEMA_V3,
                          max_new_tokens=768, chunk_tokens=512, chunk_stride=480, base_revision=BASE_REVISION)
    identity = {"teacher": backend.identity, "queue_hash": digest(candidates), "target": 80,
                "corpus_sha256": file_digest(ARTIFACTS / "corpus.jsonl"),
                "source_annotation_policy": "hyperlinks are incomplete positive evidence; no exhaustive gold claim",
                "label_source": "machine_assisted_weak", "schema_version": EXTRACTION_SCHEMA_V3}
    previous = read_json(directory / "identity.json")
    if previous and previous != identity:
        raise ValueError("Weak-label identity changed; preserve this version")
    write_json(directory / "identity.json", identity)
    write_jsonl(directory / "queue.jsonl", candidates)
    attempts = read_jsonl(directory / "attempts.jsonl") if (directory / "attempts.jsonl").exists() else []
    examples = [row["accepted_example"] for row in attempts if row["outcome"] == "accepted"]
    if examples:
        write_jsonl(dataset_path, examples)
    seen = {r["article_id"] for r in attempts}
    usable, _, recovery_report = collect_recovery()
    update(f"labels: {len(usable)} usable ({len(examples)} raw accepted, {recovery_report['derived']} case-derived); continuing unattempted articles")
    started = time.monotonic()
    for candidate in candidates:
        languages = Counter(r["language"] for r in usable)
        enough = len(usable) >= 32 and min(languages.get("en", 0), languages.get("ar", 0)) >= 4
        if enough or remaining() < 2400 or time.monotonic() - started > 4200:
            break
        if candidate["article_id"] in seen:
            continue
        tick = time.monotonic()
        raw, metadata, error = [], [], None
        accepted_example = None
        try:
            labels, raw = backend.extract(candidate["text"])
            metadata = backend.last_chunk_metadata
            accept_weak_label(labels, candidate["text"])
            accepted_example = {**candidate, "split": "train", "schema_version": EXTRACTION_SCHEMA_V3,
                "labels": labels, "label_source": "machine_assisted_weak_not_human_gold",
                "teacher_identity": backend.identity, "exhaustiveness": "unverified",
                "field_checks": {c: "teacher_checked_literal_spans_not_human_verified" for c in CURRENT_FILTER_CATEGORIES}}
            examples.append(accepted_example)
        except ValueError as exc:
            raw = getattr(exc, "raw_outputs", raw)
            metadata = getattr(exc, "chunk_metadata", metadata)
            error = str(exc)
        attempts.append({"article_id": candidate["article_id"], "language": candidate["language"],
                         "source_hash": digest(candidate["text"]), "raw_outputs": raw,
                         "chunk_metadata": metadata, "error": error, "elapsed_seconds": time.monotonic() - tick,
                         "accepted_example": accepted_example,
                         "outcome": "accepted" if error is None else "rejected"})
        write_jsonl(directory / "attempts.jsonl", attempts)
        write_jsonl(dataset_path, examples)
        usable, _, recovery_report = collect_recovery()
        update(f"labels: {len(usable)} usable ({len(examples)} raw accepted, {recovery_report['derived']} case-derived) / {len(attempts)} attempted")
    del backend
    release_gpu()
    recover(save=True)
    return usable


def predict_run(run_id, remaining, update):
    reviews = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    roles = review_roles(reviews)
    adapter = None if run_id == "base-v3" else ARTIFACTS / "runs" / run_id / "adapter"
    backend = QwenBackend(mode="base" if adapter is None else "active", adapter_path=adapter,
                          schema_version=EXTRACTION_SCHEMA_V3, max_new_tokens=768,
                          chunk_tokens=512, chunk_stride=480, base_revision=BASE_REVISION)
    directory = ARTIFACTS / "evaluation" / run_id
    manifest = {"model": backend.identity, "frozen_ids": sorted(roles),
                "article_hashes": {r["article_id"]: r["content_hash"] for r in reviews},
                "full_input_hashes": {r["article_id"]: digest(r["title"] + "\n\n" + r["text"]) for r in reviews}}
    previous = read_json(directory / "manifest.json")
    if previous and previous != manifest:
        raise ValueError("Evaluation identity changed; choose a new run version")
    write_json(directory / "manifest.json", manifest)
    path = directory / "predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    seen = {r["article_id"] for r in rows}
    if len(seen) != len(rows):
        raise ValueError("Duplicated inference rows")
    for record in reviews:
        if record["article_id"] in seen:
            continue
        if remaining() < 180:
            raise TimeoutError("Budget reached; predictions resume without overwriting completed articles")
        tick = time.monotonic()
        try:
            _, raw = backend.extract(record["title"] + "\n\n" + record["text"])
            metadata, error = backend.last_chunk_metadata, None
        except ValueError as exc:
            raw, metadata, error = getattr(exc, "raw_outputs", []), getattr(exc, "chunk_metadata", []), str(exc)
        rows.append({"article_id": record["article_id"], "raw_outputs": raw, "error": error,
                     "chunk_metadata": metadata, "elapsed_seconds": time.monotonic() - tick})
        write_jsonl(path, rows)
        update(f"{run_id}: {len(rows)}/30 saved predictions; no test scoring")
    del backend
    release_gpu()
    report = score_extraction(run_id, "validation")
    update(f"{run_id}: validation F1={report['metrics']['micro_f1']:.3f}, schema={report['metrics']['schema_validity']:.2f}")


def completed_training(run_id):
    directory = ARTIFACTS / "runs" / run_id
    manifest = read_json(directory / "manifest.json", {})
    if manifest.get("status") != "completed":
        return False
    for field, name in (("adapter_sha256", "adapter_model.safetensors"),
                        ("adapter_config_sha256", "adapter_config.json")):
        if file_digest(directory / "adapter" / name) != manifest[field]:
            raise ValueError("Completed training artifact changed")
    if manifest.get("optimizer_steps") != 48 or not manifest.get("changed_tensors"):
        raise ValueError("Completed training has insufficient optimizer evidence")
    return True


def isolated_stage(stage):
    """The controller never loads Torch; process exit releases each model fully."""
    if stage.startswith("analysis:"):
        command = [sys.executable, "-u", str(Path(__file__).with_name("phase3_analysis.py")), stage.split(":", 1)[1]]
    else:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", stage]
    state = read_json(FOLDER / "job.json")
    timeout = (datetime.fromisoformat(state["deadline"]) - datetime.now(timezone.utc)).total_seconds()
    if timeout <= 0:
        raise TimeoutError("Prepaid stage deadline expired")
    record = {"stage": stage, "started_at": datetime.now(timezone.utc).isoformat(), "command": command}
    child = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr,
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    record.update(pid=child.pid, status="running")
    write_json(FOLDER / "stage_process.json", record)
    try:
        code = child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # The venv launcher has its own Python child on Windows: kill only our owned tree.
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False)
        else:
            child.terminate()
        code = child.wait()
        record["error"] = "Prepaid deadline reached; owned stage terminated, durable artifacts retained"
    record.update(status="exited", return_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(FOLDER / "stage_process.json", record)
    history = FOLDER / "stage_history.jsonl"
    write_jsonl(history, (read_jsonl(history) if history.exists() else []) + [record])
    if code:
        raise RuntimeError(f"Isolated stage {stage} exited {code}; inspect worker logs")


def single_stage(stage):
    verify_completion()
    state = read_json(FOLDER / "job.json")
    state["policy"] = "No automatic deployment; validation decisions are locked before final-test reports. Completed stages are reused."
    deadline = datetime.fromisoformat(state["deadline"])
    remaining = lambda: max(0, (deadline - datetime.now(timezone.utc)).total_seconds())
    def update(message):
        state.update(status="running", progress=message, storage=check_storage(ARTIFACTS),
                     updated_at=datetime.now(timezone.utc).isoformat())
        write_json(FOLDER / "job.json", state)
        print(message, flush=True)
    operation, run_id = stage.split(":", 1)
    try:
        check_storage(ARTIFACTS)
        if operation == "train":
            examples = prepare_labels(remaining, update)
            selected = read_json(FOLDER / "training_dataset.json")
            initial = ARTIFACTS / "runs/domain-v1/adapter" if run_id == "domain-extraction-v3" else None
            train_v3(run_id, examples, steps=48, initial_adapter=initial,
                     seconds=min(1800, remaining() - 180), progress=update,
                     dataset_artifact=selected["path"])
            if not completed_training(run_id):
                raise RuntimeError("Training time limit reached; checkpoint retained, no completed claim")
        else:
            predict_run(run_id, remaining, update)
    except Exception as exc:
        state.update(status="needs_attention", error=f"{type(exc).__name__}: {exc}")
        write_json(FOLDER / "job.json", state)
        raise


def main(hours, extension_hours=0, allocation_id=None):
    verify_completion()
    check_storage(ARTIFACTS)
    state = reserve_budget(hours, extension_hours, allocation_id)
    if state.get("status") == "predictions_ready_validation_only":
        print("Completed job retained; continue selection/linker/ranking audit, do not retrain.", flush=True)
        return
    state.setdefault("original_raw_label_target", state.get("target_labels", 80))
    state.update(target_labels=32, label_dataset_policy=POLICY)
    deadline = datetime.fromisoformat(state["deadline"])
    if state.get("error"):
        state.setdefault("prior_errors", []).append(state.pop("error"))
    started = time.monotonic()
    old_elapsed = state.get("observed_seconds", 0)
    def remaining():
        return max(0, (deadline - datetime.now(timezone.utc)).total_seconds())
    def update(message):
        state["storage"] = check_storage(ARTIFACTS)
        state.update(status="running", progress=message, updated_at=datetime.now(timezone.utc).isoformat(),
                     observed_seconds=old_elapsed + time.monotonic() - started)
        write_json(FOLDER / "job.json", state)
        print(message, flush=True)
    try:
        if remaining() < 180:
            raise TimeoutError("Prepaid deadline expired; inspect progress before allocating any further time")
        if not read_json(FOLDER / "training_dataset.json"):
            raise ValueError("Frozen training dataset pointer required; relabeling is disabled for this continuation")
        update("Reusing frozen v3 training labels; no relabeling")
        examples = prepare_labels(remaining, update)
        selected = read_json(FOLDER / "training_dataset.json")
        dataset_artifact = selected["path"] if selected else "phase3/weak_labels_v3/examples.jsonl"
        for run_id, initial in (("extraction-only-v3", None),
                                ("domain-extraction-v3", ARTIFACTS / "runs/domain-v1/adapter")):
            if completed_training(run_id):
                update(f"{run_id}: completed artifacts verified and retained; no retraining")
                continue
            if remaining() < 900:
                raise TimeoutError("Insufficient budget for another training stage")
            isolated_stage("train:" + run_id)
            state = read_json(FOLDER / "job.json")
        for run_id in RUNS:
            isolated_stage("predict:" + run_id)
            state = read_json(FOLDER / "job.json")
        for stage in ("linker", "ranking", "lock", "linker-test", "finish"):
            update(f"Phase 3 isolated analysis stage: {stage}")
            isolated_stage("analysis:" + stage)
            state = read_json(FOLDER / "job.json")
        state.update(status="phase3_evaluated", progress="Phase 3 reports saved; inspect completion evidence and unmet release gates")
    except Exception as exc:
        state.update(status="needs_attention", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Child stages own progress/errors; do not overwrite them with a stale parent snapshot.
        latest = read_json(FOLDER / "job.json", {})
        state = {**latest, **state} if state.get("status") != "running" else latest
        state.update(observed_seconds=old_elapsed + time.monotonic() - started,
                     updated_at=datetime.now(timezone.utc).isoformat())
        write_json(FOLDER / "job.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--extend-hours", type=float, default=0)
    parser.add_argument("--allocation-id")
    parser.add_argument("--stage", choices=["train:extraction-only-v3", "train:domain-extraction-v3",
                                           *("predict:" + run for run in RUNS)])
    args = parser.parse_args()
    if not 0 < args.hours <= 3:
        parser.error("This experiment is bounded to at most three hours")
    if not 0 <= args.extend_hours <= 2:
        parser.error("A resume allowance must be between zero and two hours")
    if args.stage:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = stamp + "-" + args.stage.replace(":", "-")
        logs = FOLDER / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / (name + ".out.log")).open("w", encoding="utf-8") as out, \
             (logs / (name + ".err.log")).open("w", encoding="utf-8") as err, \
             redirect_stdout(out), redirect_stderr(err), GPU_LOCK.acquire(timeout=0):
            try:
                single_stage(args.stage)
            except BaseException:
                traceback.print_exc(file=err)
                raise
    else:
        main(args.hours, args.extend_hours, args.allocation_id)
