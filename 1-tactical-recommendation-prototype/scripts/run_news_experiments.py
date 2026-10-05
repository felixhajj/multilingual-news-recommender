"""Resumable stage orchestration, with a cumulative 24-hour process budget."""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, GPU_LOCK, read_json, read_jsonl, write_json
from src.news_evaluation import freeze_extraction_review
from src.news_pipeline import QwenBackend
from src.news_training import prepare_training_examples, train_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--label-hours", type=float, default=3)
    parser.add_argument("--sft-hours", type=float, default=3)
    parser.add_argument("--domain-hours", type=float, default=2)
    parser.add_argument("--run-prefix", default="", help="Use a new prefix for an explicit new experiment")
    parser.add_argument("--corrections-version", help="Approved dataset hash; never used implicitly")
    args = parser.parse_args()
    started = time.monotonic()
    budget_path = ARTIFACTS / "processing_budget.json"
    budget = read_json(budget_path, {"consumed_seconds": 0, "limit_seconds": 86400})
    def remaining():
        return max(0, budget["limit_seconds"]-budget["consumed_seconds"]-(time.monotonic()-started))
    records = read_jsonl(ARTIFACTS / "corpus.jsonl")
    freeze_extraction_review(records)
    try:
        backend = QwenBackend(mode="base")
        examples = prepare_training_examples(records, backend, target=args.examples,
                                              seconds=min(args.label_hours*3600, remaining()))
        if args.corrections_version:
            if Path(args.corrections_version).name != args.corrections_version:
                raise ValueError("Expected a plain approved dataset version")
            examples += read_jsonl(ARTIFACTS / "datasets" / args.corrections_version / "examples.jsonl")
        del backend
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        if len(examples) < 20:
            raise RuntimeError("Fewer than 20 grounded examples passed. Expand/review data before training.")
        for name, stage in (("extraction-only-v1", "sft"), ("domain-v1", "domain"), ("domain-extraction-v1", "sft")):
            run_id = args.run_prefix + name
            previous = read_json(ARTIFACTS / "runs" / run_id / "manifest.json", {})
            if previous:
                if previous.get("status") not in {"completed", "time_limited"}:
                    raise RuntimeError(f"{run_id} was not successful; inspect its manifest instead of silently restarting")
                continue
            if remaining() < 120:
                raise RuntimeError("Cumulative processing budget exhausted")
            data = examples if stage == "sft" else [{"article_id": r["article_id"], "split": "train", "text": r["body"]}
                                                     for r in records if r["split"] == "train" and r["domain_hits"] > 0][:500]
            initial = ARTIFACTS / "runs" / (args.run_prefix + "domain-v1") / "adapter" if name == "domain-extraction-v1" else None
            train_adapter(run_id, data, stage=stage, initial_adapter=initial, max_steps=args.steps,
                          seconds=min((args.domain_hours if stage == "domain" else args.sft_hours)*3600, remaining()))
        write_json(ARTIFACTS / "experiment_status.json", {"status": "training_finished_evaluation_pending",
                   "promotion": "blocked_until_independent_validation", "review_path": "review/extraction.jsonl"})
    finally:
        budget["consumed_seconds"] += time.monotonic()-started
        write_json(budget_path, budget)


if __name__ == "__main__":
    with GPU_LOCK:
        main()
