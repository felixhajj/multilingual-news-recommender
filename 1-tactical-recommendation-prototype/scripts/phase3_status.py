"""Read durable Phase 3 progress without loading models or scoring test data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, read_json, read_jsonl


def status():
    phase = ARTIFACTS / "phase3"
    job = read_json(phase / "job.json", {"status": "not_started"})
    dataset = read_json(phase / "weak_labels_v3/manifest.json")
    selected = read_json(phase / "training_dataset.json")
    if selected:
        dataset = read_json((ARTIFACTS / selected["path"]).parent / "manifest.json")
    attempts = phase / "weak_labels_v3/attempts.jsonl"
    if not dataset and attempts.exists():
        rows = read_jsonl(attempts)
        dataset = {"attempts": len(rows), "accepted": sum(r["outcome"] == "accepted" for r in rows),
                   "status": "preparing"}
    runs = {}
    for run in ("base-v3", "extraction-only-v3", "domain-extraction-v3"):
        manifest = read_json(ARTIFACTS / "runs" / run / "manifest.json", {})
        checkpoint = read_json(ARTIFACTS / "runs" / run / "checkpoint.json", {})
        predictions = ARTIFACTS / "evaluation" / run / "predictions.jsonl"
        runs[run] = {"status": manifest.get("status", "not_trained" if run != "base-v3" else "pretrained"),
                     "durable_updates": checkpoint.get("step", 0),
                     "saved_predictions": len(read_jsonl(predictions)) if predictions.exists() else 0,
                     "validation_report": (phase / "reports" / f"{run}_validation.json").exists()}
    budget = read_json(ARTIFACTS / "processing_budget.json")
    completion = read_json(phase / "completion.json")
    return {"phase": 3, "complete": completion is not None,
            "completion": completion, "job": job, "dataset": dataset,
            "process": read_json(phase / "process.json"),
            "stage": read_json(phase / "stage_process.json"),
            "runs": runs, "selection_locked": (phase / "selection_lock.json").exists(),
            "budget_charged_hours": budget["consumed_seconds"] / 3600,
            "budget_remaining_hours": (budget["limit_seconds"] - budget["consumed_seconds"]) / 3600,
            "note": "Durable status is not a live-process check. Inspect worker PID/logs before resuming."}


if __name__ == "__main__":
    print(json.dumps(status(), indent=2, ensure_ascii=False))
