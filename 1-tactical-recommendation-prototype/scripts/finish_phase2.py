"""Validate and seal Phase 2 inputs without scoring model predictions."""
import argparse
import json
import shutil
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction_data import EXTRACTION_SCHEMA_V3
from src.news_evaluation import review_roles, validate_ranking_review, validate_review
from src.news_training import validate_training_examples
from src.portfolio_config import ARTIFACTS, digest, file_digest, read_json, read_jsonl, write_json, write_jsonl


def verify_completion():
    report = read_json(ARTIFACTS / "phase2/completion.json", {})
    if report.get("status") != "complete":
        raise ValueError("Phase 2 has no completed integrity handoff")
    directory = ARTIFACTS / report["snapshot_directory"]
    manifest = read_json(directory / "manifest.json")
    if not manifest or digest(manifest) != report["snapshot_manifest_hash"]:
        raise ValueError("Phase 2 snapshot manifest changed")
    for name, checksum in manifest["files"].items():
        if file_digest(directory / name) != checksum:
            raise ValueError(f"Sealed Phase 2 artifact changed: {name}")
        if file_digest(ARTIFACTS / name) != checksum:
            raise ValueError(f"Current Phase 2 artifact changed since handoff: {name}")
    return report


def _rejection_kind(reason):
    value = (reason or "").lower()
    if "schema" in value or "contain exactly" in value:
        return "schema_fields"
    if "json" in value:
        return "json_parse"
    if "relationship" in value:
        return "relationship_grounding_or_structure"
    if "literal" in value or "grounded" in value:
        return "entity_grounding"
    if "published positive" in value:
        return "published_positive_omissions"
    return "other"


def finish():
    completion_path = ARTIFACTS / "phase2/completion.json"
    if completion_path.exists():
        return verify_completion()
    with ExitStack() as stack:
        for name in ("extraction", "ranking"):
            stack.enter_context(FileLock(str(ARTIFACTS / f"review/{name}.jsonl") + ".lock"))
        extraction = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
        ranking = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
        if validate_review(extraction) != 30 or validate_ranking_review(ranking) != 30:
            raise ValueError("All 30 extraction and 30 recommendation items must be reviewed")
        if any(row.get("schema_version") != EXTRACTION_SCHEMA_V3
               or row.get("review_scope") != "named_entities_v3" for row in extraction):
            raise ValueError("Extraction references must use the reviewed v3 protocol")
        roles = review_roles(extraction)
        expected_roles = Counter({"validation": 10, "test": 20})
        if Counter(roles.values()) != expected_roles or Counter(r["role"] for r in ranking) != expected_roles:
            raise ValueError("Frozen validation/test counts changed")

        preparation = read_json(ARTIFACTS / "training_preparation_v2.json")
        version = preparation["version"]
        label_dir = ARTIFACTS / "labeling" / version
        attempts = read_jsonl(label_dir / "attempts.jsonl")
        additions = read_jsonl(label_dir / "accepted.jsonl")
        seed = read_jsonl(ARTIFACTS / "phase2/usable_training_seed.jsonl")
        examples = seed + additions
        corpus = read_jsonl(ARTIFACTS / "corpus.jsonl")
        groups = set(read_json(ARTIFACTS / "review/frozen_extraction_manifest.json")["group_ids"])
        coverage = validate_training_examples(examples, corpus, groups)
        if len(examples) > 500 or digest(examples) != preparation["dataset_hash"]:
            raise ValueError("Training dataset size or identity changed")
        if any(row.get("label_source") != "model_assisted" for row in examples):
            raise ValueError("Weak-label provenance changed")
        sources = {row["article_id"]: row for row in corpus}
        export = []
        for row in examples:
            source = sources[row["article_id"]]
            if row["group_id"] != source["group_id"] or row["language"] != source["language"]:
                raise ValueError("Training group or language disagrees with corpus provenance")
            enriched = dict(row)
            for field in ("source_url", "license_id", "license_url", "source_revision"):
                if row.get(field) and row[field] != source.get(field):
                    raise ValueError(f"Training {field} disagrees with its verified source")
                enriched[field] = source.get(field)
            if any(not enriched.get(field) for field in ("source_url", "license_id", "license_url")):
                raise ValueError("Verified training source lacks required license metadata")
            enriched.update(schema_version="extraction-v2",
                            provenance_note="Source license metadata restored from the matching verified corpus record; labels unchanged.")
            export.append(enriched)
        export_name = f"phase2/datasets/{digest(export)}/examples.jsonl"
        export_path = ARTIFACTS / export_name
        if export_path.exists() and digest(read_jsonl(export_path)) != digest(export):
            raise ValueError("Existing versioned training export changed")
        write_jsonl(export_path, export)
        queue = read_jsonl(ARTIFACTS / "phase2/labeling_candidates.jsonl")
        by_id = {row["article_id"]: row for row in queue}
        if len(by_id) != len(queue) or len({a["attempt_id"] for a in attempts}) != len(attempts):
            raise ValueError("Duplicate labeling candidates or attempts")
        for attempt in attempts:
            source = by_id.get(attempt["article_id"])
            if not source or any(attempt[key] != source[key] for key in ("content_hash", "group_id", "language")):
                raise ValueError("Labeling attempt provenance changed")
            if attempt["dataset_version"] != version or attempt["teacher"] != preparation["policy"]["teacher"]:
                raise ValueError("Labeling attempt model or dataset identity changed")
        accepted_ids = {a["article_id"] for a in attempts if a["outcome"] == "accepted"}
        if accepted_ids != {row["article_id"] for row in additions}:
            raise ValueError("Accepted attempts and saved additions disagree")
        outcomes = dict(Counter(a["outcome"] for a in attempts))
        if len(attempts) != preparation["attempts"] or outcomes != preparation["outcomes"]:
            raise ValueError("Labeling totals do not match the durable manifest")
        if set(by_id) != {a["article_id"] for a in attempts}:
            raise ValueError("The bounded candidate queue is not exhausted")

        files = [export_name,
            "review/extraction.jsonl", "review/ranking.jsonl", "review/review_events.jsonl",
            "review/extraction_roles.json", "review/extraction_integrity_v3.json",
            "review/ranking_integrity_v2.json", "review/frozen_extraction_manifest.json",
            "review/frozen_ranking_manifest.json", "review/annotation_protocol_v3.json",
            "phase2/usable_training_seed.jsonl", "phase2/quarantined_training_examples.jsonl",
            "phase2/labeling_candidates.jsonl", "phase2/location_schema_migration.json",
            "training_preparation_v2.json",
        ] + [f"labeling/{version}/{name}" for name in
             ("accepted.jsonl", "attempts.jsonl", "manifest.json", "quarantined.jsonl")]
        checksums = {name: file_digest(ARTIFACTS / name) for name in files}
        snapshot_id = digest(checksums)
        directory = ARTIFACTS / "phase2/snapshots" / snapshot_id
        for name in files:
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and file_digest(destination) != checksums[name]:
                raise ValueError("Existing snapshot contains a conflicting file")
            shutil.copyfile(ARTIFACTS / name, destination)
        manifest = {"version": 1, "snapshot_id": snapshot_id, "files": checksums,
                    "policy": "Human references are evaluation-only. V2 machine labels are historical weak training inputs."}
        write_json(directory / "manifest.json", manifest)
        budget = read_json(ARTIFACTS / "processing_budget.json")
        report = {
            "phase": 2, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_directory": directory.relative_to(ARTIFACTS).as_posix(),
            "snapshot_manifest_hash": digest(manifest), "dataset_version": version,
            "review": {"extraction": 30, "recommendation": 30, "schema_version": EXTRACTION_SCHEMA_V3,
                       "extraction_roles": dict(Counter(roles.values())),
                       "ranking_roles": dict(Counter(r["role"] for r in ranking)),
                       "ranking_queries": len({r["query_id"] for r in ranking}),
                       "extraction_reviewers": dict(Counter(r["reviewed_by"] for r in extraction)),
                       "ranking_reviewers": dict(Counter(r["reviewed_by"] for r in ranking)),
                       "references_used_for_training": False},
            "training": {"schema_version": "extraction-v2", "label_source": "machine_assisted",
                         "export_path": export_name, "export_hash": digest(export),
                         "coverage": coverage, "usable_seed": len(seed), "new_accepted": len(additions),
                         "human_gold_examples": 0, "v3_training_examples": 0,
                         "by_language": {language: dict(Counter(a["outcome"] for a in attempts if a["language"] == language))
                                         for language in ("en", "ar")},
                         "rejection_categories": dict(Counter(_rejection_kind(a.get("reason")) for a in attempts if a["outcome"] == "rejected")),
                         "attempts": len(attempts), "outcomes": outcomes},
            "processing": {"completed_attempt_seconds": round(sum(a["elapsed_seconds"] for a in attempts), 3),
                           "phase2_invocation_seconds": round(sum(a["elapsed_seconds"] for a in budget.get("allocations", []) if a.get("phase") == 2), 3),
                           "ledger_consumed_seconds": budget["consumed_seconds"],
                           "nominal_remaining_seconds": budget["limit_seconds"] - budget["consumed_seconds"],
                           "note": "Completed attempts exclude loading and failed/interrupted overhead. Remaining time is a ledger upper bound; reconcile unmetered historical time in Phase 3."},
            "next_phase": 3,
            "unresolved": ["Prepare v3 training labels using train articles; preserve reviewed validation/test references.",
                           "Diagnose weak-label rejection rate and missing company/equipment coverage.",
                           "Implement validation-only scoring and a selection lock before final-test scoring.",
                           "Validate extraction, linking and ranking before model promotion."],
            "model_metrics_computed": False,
        }
        write_json(completion_path, report)
    return verify_completion()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing seal without creating artifacts")
    args = parser.parse_args()
    print(json.dumps(verify_completion() if args.check else finish(), indent=2))
