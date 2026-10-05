"""Validation-gated deployment and explicit rollback; never select using test scores."""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, GPU_LOCK, digest, file_digest, read_json, read_jsonl, write_json
from src.news_selection import plain_id, verify_selection_lock
from src.learned_linker import linker_artifact_identity


def promote(run_id):
    plain_id(run_id)
    lock = verify_selection_lock()
    if lock["selected_run"] != run_id or lock["decision"] != "promote_candidate":
        raise ValueError("Only the validation-locked selected candidate may be promoted")
    model = lock["run_evidence"][run_id]["model"]
    release = read_json(ARTIFACTS / "release_manifest.json", {})
    choices = read_json(ARTIFACTS / "phase3/component_decisions.json", {})
    linker_path = choices.get("linker_selected")
    if not linker_path or not (ARTIFACTS / linker_path).resolve().is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("No validation-selected linker")
    linker_identity = linker_artifact_identity(ARTIFACTS / linker_path)
    if not linker_identity["training_identity_complete"]:
        raise ValueError("Legacy linker provenance is incomplete; validate a fully bound linker before promotion")
    if (release.get("status") != "prepared" or release.get("articles", 0) < 300
            or release.get("model") != model
            or release.get("linker_version") != digest(linker_identity)):
        raise ValueError("Build a complete index matching the selected model and current linker before promotion")
    with sqlite3.connect((ARTIFACTS / "release.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        records = db.execute("SELECT payload, embedding_version FROM articles").fetchall()
    matching = [json.loads(payload) for payload, encoder in records
                if encoder == release.get("embedding_version")]
    if (len(matching) != release["articles"]
            or any(r.get("status") != "ready" or r.get("model") != model
                   or r.get("pipeline_identity") != release.get("pipeline_identity") for r in matching)):
        raise ValueError("Indexed article identities disagree with the selected release")
    gate = {"passed": True, "run_id": run_id, "selection_lock_hash": lock["lock_hash"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(), "policy": lock["gate"]}
    previous = read_json(ARTIFACTS / "active_model.json", {"run_id": "legacy-20-mock-examples"})
    write_json(ARTIFACTS / "previous_model.json", previous)
    write_json(ARTIFACTS / "runs" / run_id / "promotion_gate.json", gate)
    write_json(ARTIFACTS / "active_model.json", {"run_id": run_id, "adapter": f"runs/{run_id}/adapter",
                                               "model": model, "linker": linker_path, "gate": gate})
    print("Promoted the validation-locked candidate with a matching prepared index. Restart the app.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["promote", "rollback"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    with GPU_LOCK:
        if args.operation == "promote":
            promote(args.run_id)
        else:
            previous = read_json(ARTIFACTS / "previous_model.json")
            if not previous:
                raise SystemExit("No prior deployment to restore")
            current = read_json(ARTIFACTS / "active_model.json", {})
            write_json(ARTIFACTS / "active_model.json", previous)
            write_json(ARTIFACTS / "previous_model.json", current)
            print("Restored prior model. Restart the app and rebuild or restore its matching index.")
