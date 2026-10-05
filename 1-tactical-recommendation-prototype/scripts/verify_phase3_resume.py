"""Verify durable progress and protected inputs without training or scoring tests."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finish_phase2 import verify_completion
from src.news_training_v3 import assert_resume_identity, check_checkpoint, training_identity
from src.portfolio_config import (ARTIFACTS, GPU_LOCK, ROOT, digest, file_digest,
                                  read_json, read_jsonl, write_json)


def verify():
    phase2 = verify_completion()
    phase = ARTIFACTS / "phase3"
    selected = read_json(phase / "training_dataset.json")
    dataset = ARTIFACTS / selected["path"]
    if (not dataset.resolve().is_relative_to(ARTIFACTS.resolve())
            or file_digest(dataset) != selected["sha256"]
            or file_digest(dataset.parent / "manifest.json") != selected["manifest_sha256"]):
        raise ValueError("Frozen training dataset changed")
    examples = read_jsonl(dataset)
    attempts = read_jsonl(phase / "weak_labels_v3/attempts.jsonl")
    directory = ARTIFACTS / "runs/extraction-only-v3"
    manifest = read_json(directory / "manifest.json")
    old_backend = os.environ.get("NEWS_SAFETENSORS_BACKEND")
    try:
        os.environ["NEWS_SAFETENSORS_BACKEND"] = manifest["identity"]["checkpoint_backend"]
        assert_resume_identity(manifest, training_identity(examples, manifest["identity"]["max_steps"]))
    finally:
        if old_backend is None:
            os.environ.pop("NEWS_SAFETENSORS_BACKEND", None)
        else:
            os.environ["NEWS_SAFETENSORS_BACKEND"] = old_backend
    pointer = read_json(directory / "checkpoint.json")
    checkpoint = check_checkpoint(directory, pointer)
    import torch
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    if (state["step"] != pointer["step"] or state["identity_hash"] != digest(manifest["identity"])
            or len(state["log"]) != state["step"] or not state["optimizer"]["state"]
            or not state.get("rng_cuda") or state["rng_cpu"].numel() == 0):
        raise ValueError("Checkpoint progress/optimizer/RNG identity mismatch")
    if file_digest(directory / "initial_weights.pt") != manifest["initial_weights_sha256"]:
        raise ValueError("Initial weight evidence changed")
    backup = phase / "history/recovery-20260930-1528"
    inventory = json.loads((backup / "preserved_sha256.json").read_text(encoding="utf-8-sig"))
    for item in inventory:
        relative = Path(item["Path"]).relative_to(ROOT)
        if file_digest(backup / relative).casefold() != item["Hash"].casefold():
            raise ValueError("Recovery backup inventory mismatch")
    budget = read_json(ARTIFACTS / "processing_budget.json")
    proof = {"verified_at": datetime.now(timezone.utc).isoformat(), "status": "resume_supported",
             "checkpoint": str(checkpoint), "pointer": pointer, "optimizer_states": len(state["optimizer"]["state"]),
             "durable_updates": state["step"], "loss_first": state["log"][0]["loss"],
             "loss_latest": state["log"][-1]["loss"], "supervised_tokens": state["tokens"],
             "unique_articles_seen": len(state["seen"]), "training_examples": len(examples),
             "label_attempts": len(attempts), "dataset_sha256": selected["sha256"],
             "trainer_sha256": file_digest(ROOT / "src/news_training_v3.py"),
             "protected_review_hashes": {name: file_digest(ARTIFACTS / "review" / name)
                                         for name in ("extraction.jsonl", "ranking.jsonl")},
             "phase2_snapshot": phase2["snapshot_directory"], "backup_files_verified": len(inventory),
             "budget_sha256": file_digest(ARTIFACTS / "processing_budget.json"),
             "budget_charged_hours": budget["consumed_seconds"] / 3600,
             "budget_remaining_hours": (budget["limit_seconds"] - budget["consumed_seconds"]) / 3600,
             "note": "Resume proof only; training is incomplete and no final-test metric is computed."}
    write_json(phase / "reports" / f"resume_verified_step-{state['step']:04d}.json", proof)
    return proof


if __name__ == "__main__":
    with GPU_LOCK.acquire(timeout=0):
        print(json.dumps(verify(), indent=2))
