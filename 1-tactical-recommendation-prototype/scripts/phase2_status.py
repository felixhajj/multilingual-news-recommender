"""Report Phase 2 progress without importing models or reading predictions."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio_config import ARTIFACTS, digest, file_digest, read_json, read_jsonl


def main():
    extraction = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    ranking = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
    latest = read_json(ARTIFACTS / "training_preparation_v2.json")
    completion = read_json(ARTIFACTS / "phase2/completion.json", {})
    seal_valid = False
    if completion.get("status") == "complete":
        directory = ARTIFACTS / completion["snapshot_directory"]
        manifest = read_json(directory / "manifest.json", {})
        seal_valid = bool(manifest) and digest(manifest) == completion["snapshot_manifest_hash"]
        for name, checksum in manifest.get("files", {}).items():
            seal_valid = seal_valid and (directory / name).exists() and (ARTIFACTS / name).exists()
            if seal_valid:
                seal_valid = file_digest(directory / name) == checksum and file_digest(ARTIFACTS / name) == checksum
    budget = read_json(ARTIFACTS / "processing_budget.json")
    result = {
        "phase": 2,
        "data_preparation": read_json(ARTIFACTS / "phase2/data_preparation.json"),
        "labeling_job": read_json(ARTIFACTS / "phase2/labeling_job.json"),
        "latest_labeling_dataset": latest,
        "human_review": {
            "schema_version": "extraction-v3-locations",
            "extraction": dict(Counter(row["review_status"] for row in extraction)),
            "recommendation": dict(Counter(row["review_status"] for row in ranking)),
        },
        "schema_migration": read_json(ARTIFACTS / "phase2/location_schema_migration.json"),
        "budget": {"consumed_seconds": budget["consumed_seconds"],
                   "nominal_remaining_seconds": budget["limit_seconds"] - budget["consumed_seconds"]},
        "completion": completion,
        "completion_seal_valid": seal_valid,
        "complete": seal_valid and all(row["review_status"] == "human_reviewed" for row in extraction + ranking),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
