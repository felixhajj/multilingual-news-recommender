"""Save genuine ablation predictions; score only independently reviewed references."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news_pipeline import QwenBackend
from src.news_evaluation import extraction_metrics, validate_review, review_roles
from src.portfolio_config import ARTIFACTS, GPU_LOCK, digest, read_json, read_jsonl, write_json, write_jsonl
from src.portfolio_reports import publish_evidence_snapshot


def predict(hours):
    reviews = read_jsonl(ARTIFACTS / "review" / "extraction.jsonl")
    roles = review_roles(reviews)
    started = time.monotonic()
    budget = read_json(ARTIFACTS / "processing_budget.json", {"consumed_seconds": 0, "limit_seconds": 86400})
    seconds = min(hours*3600, max(0,budget["limit_seconds"]-budget["consumed_seconds"]))
    try:
        for run_id in ("base", "extraction-only-v1", "domain-extraction-v1"):
            adapter = None if run_id == "base" else ARTIFACTS / "runs" / run_id / "adapter"
            if adapter and not (adapter / "adapter_model.safetensors").exists():
                print(f"Skipping {run_id}: no trained artifact", flush=True)
                continue
            backend = QwenBackend(mode="base" if run_id == "base" else "active", adapter_path=adapter)
            destination = ARTIFACTS / "evaluation" / run_id
            manifest = {"model": backend.identity, "frozen_ids": sorted(roles),
                        "article_hashes": {r["article_id"]: r["content_hash"] for r in reviews}}
            previous = read_json(destination / "manifest.json")
            if previous and previous != manifest:
                raise ValueError("Prediction identity changed; preserve previous results in a distinct evaluation version")
            write_json(destination / "manifest.json", manifest)
            path = destination / "predictions.jsonl"
            rows = read_jsonl(path) if path.exists() else []
            seen = {r["article_id"] for r in rows}
            for record in reviews:
                if time.monotonic()-started >= seconds:
                    return
                if record["article_id"] in seen:
                    continue
                inference_start = time.monotonic()
                try:
                    _, raw = backend.extract(record["title"] + "\n\n" + record["text"])
                    error = None
                except ValueError as exc:
                    raw, error = [getattr(exc, "raw_output", "")], str(exc)
                rows.append({"article_id": record["article_id"], "raw_outputs": raw,
                             "error": error, "elapsed_seconds": time.monotonic()-inference_start})
                write_jsonl(path, rows)
                print(f"{run_id}: {len(rows)}/{len(reviews)} predictions", flush=True)
            del backend
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    finally:
        budget["consumed_seconds"] += time.monotonic()-started
        write_json(ARTIFACTS / "processing_budget.json", budget)


def score():
    raise ValueError("Combined validation/test scoring is retired. Use scripts/phase3.py validation; final test requires a selection lock.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["predict", "score", "freeze-roles"])
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if args.operation == "predict":
        with GPU_LOCK:
            predict(args.hours)
    elif args.operation == "score":
        score()
    else:
        print(review_roles(read_jsonl(ARTIFACTS / "review" / "extraction.jsonl")))
