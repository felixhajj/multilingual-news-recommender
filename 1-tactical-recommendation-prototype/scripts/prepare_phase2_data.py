"""Create the train-only labeling queue and lock complete human-review inputs."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news_evaluation import ensure_review_integrity_manifests, validate_ranking_review, validate_review
from src.news_training import build_labeling_queue, validate_training_examples
from src.portfolio_config import ARTIFACTS, digest, read_json, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=300,
                        help="Maximum deterministic train-only candidates; this does not run a model")
    args = parser.parse_args()
    if not 1 <= args.candidates <= 1000:
        raise ValueError("Candidate queue size must be between 1 and 1000")

    corpus = read_jsonl(ARTIFACTS / "corpus.jsonl")
    baseline = read_jsonl(ARTIFACTS / "extraction_training.jsonl")
    frozen = read_json(ARTIFACTS / "review/frozen_extraction_manifest.json")
    frozen_groups = set(frozen["group_ids"])
    usable, quarantined = [], []
    for row in baseline:
        try:
            validate_training_examples([row], corpus, frozen_groups)
            usable.append(row)
        except ValueError as exc:
            quarantined.append({"article_id": row["article_id"], "source_content_hash": row["source_content_hash"],
                                "reason": str(exc), "policy": "excluded_from_phase2_seed_original_record_preserved"})
    coverage = validate_training_examples(usable, corpus, frozen_groups)
    extraction_manifest, ranking_manifest = ensure_review_integrity_manifests()
    extraction = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    ranking = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
    validate_review(extraction)
    validate_ranking_review(ranking)

    queue = build_labeling_queue(corpus, baseline, frozen_groups, limit=args.candidates)
    queue_identity = digest([{key: row[key] for key in ("article_id", "content_hash", "group_id", "language")}
                             for row in queue])
    directory = ARTIFACTS / "phase2"
    write_jsonl(directory / "usable_training_seed.jsonl", usable)
    write_jsonl(directory / "quarantined_training_examples.jsonl", quarantined)
    queue_path = directory / "labeling_candidates.jsonl"
    manifest_path = directory / "data_preparation.json"
    existing = read_json(manifest_path)
    if existing and existing.get("candidate_identity") != queue_identity:
        raise ValueError("Candidate selection changed; preserve the prior queue before creating a new version")
    if queue_path.exists() and digest(read_jsonl(queue_path)) != digest(queue):
        raise ValueError("Existing candidate queue content changed")
    write_jsonl(queue_path, queue)
    manifest = {
        "status": "prepared_human_review_pending",
        "baseline_training": coverage,
        "historical_baseline_examples": len(baseline),
        "usable_phase2_seed_examples": len(usable),
        "quarantined_historical_examples": len(quarantined),
        "quarantine_reasons": dict(Counter(row["reason"].rsplit(": ", 1)[-1] for row in quarantined)),
        "baseline_dataset_hash": digest(baseline),
        "usable_seed_hash": digest(usable),
        "candidate_identity": queue_identity,
        "candidate_rows": len(queue),
        "candidate_languages": dict(Counter(row["language"] for row in queue)),
        "candidate_reasons": dict(Counter(reason for row in queue for reason in row["selection_reasons"])),
        "selection_policy": "Train split only; frozen groups excluded; EN/AR round-robin; company/system/domain hints prioritized.",
        "labels_created": 0,
        "labeling_note": "This command selects candidates only. It does not call Qwen or claim labels.",
        "review": {
            "extraction_identity": extraction_manifest["identity"],
            "extraction_pending": len(extraction),
            "ranking_identity": ranking_manifest["identity"],
            "ranking_pending": len(ranking),
            "human_review_required": True,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
