"""Add location review without treating legacy blank fields as negative labels."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction_data import EXTRACTION_SCHEMA_V2, EXTRACTION_SCHEMA_V3
from src.news_evaluation import extraction_review_identity, review_roles, validate_review
from src.portfolio_config import ARTIFACTS, digest, read_json, read_jsonl, write_json, write_jsonl


def main():
    review_dir = ARTIFACTS / "review"
    path = review_dir / "extraction.jsonl"
    records = read_jsonl(path)
    ranking_path = review_dir / "ranking.jsonl"
    ranking_before = digest(read_jsonl(ranking_path))
    roles = review_roles(records)
    reopened = []

    for record in records:
        if record.get("schema_version") == EXTRACTION_SCHEMA_V3:
            continue
        if record.get("review_status") == "human_reviewed":
            prior = {
                "schema_version": EXTRACTION_SCHEMA_V2,
                "labels": record["labels"],
                "reviewed_by": record.get("reviewed_by"),
                "reviewed_at": record.get("reviewed_at"),
                "review_scope": record.get("review_scope", "named_entities_v2"),
            }
            history = list(record.get("prior_schema_reviews", []))
            if not any(item.get("schema_version") == EXTRACTION_SCHEMA_V2 for item in history):
                history.append(prior)
            record["prior_schema_reviews"] = history
            draft = dict(record["labels"])
            draft["locations"] = []
            record.update(
                labels=draft,
                review_status="pending",
                reviewed_by=None,
                reviewed_at=None,
                review_scope=None,
                migration_note="Existing v2 entities are preserved as a draft; locations still require human review.",
            )
            reopened.append(record["article_id"])
        record["schema_version"] = EXTRACTION_SCHEMA_V3
        record["review_protocol_version"] = 3

    write_jsonl(path, records)
    identity = extraction_review_identity(records, roles)
    v2_integrity = read_json(review_dir / "extraction_integrity_v2.json", {})
    if v2_integrity and identity != v2_integrity.get("identity"):
        raise ValueError("Location migration changed frozen article inputs or roles")
    integrity = {
        "version": 3,
        "schema_version": EXTRACTION_SCHEMA_V3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "items": len(records),
        "roles": {role: sum(value == role for value in roles.values())
                  for role in ("validation", "test")},
        "reopened_v2_review_ids": reopened,
        "policy": "Frozen inputs and roles are unchanged. V2 labels are drafts until locations are checked under v3.",
    }
    write_json(review_dir / "extraction_integrity_v3.json", integrity)
    validate_review(records)
    ranking_after = digest(read_jsonl(ranking_path))
    if ranking_before != ranking_after:
        raise ValueError("Ranking judgments changed during extraction migration")

    legacy_seed = ARTIFACTS / "phase2" / "usable_training_seed.jsonl"
    status = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "legacy_schema": EXTRACTION_SCHEMA_V2,
        "target_schema": EXTRACTION_SCHEMA_V3,
        "legacy_seed_examples": len(read_jsonl(legacy_seed)) if legacy_seed.exists() else 0,
        "legacy_seed_status": "historical_only_not_valid_v3_gold",
        "rule": "Do not add empty locations to v2 training examples. Relabel source articles under v3 before v3 training.",
        "review_items": len(records),
        "reopened_v2_reviews": reopened,
        "ranking_digest_unchanged": ranking_after,
    }
    write_json(ARTIFACTS / "phase2" / "location_schema_migration.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
