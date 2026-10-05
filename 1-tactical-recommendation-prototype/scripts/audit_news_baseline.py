"""Audit saved metadata without inference or reading held-out model outputs.

Prediction files are counted as opaque lines and hashed, never parsed or scored.
Only --output writes a report; --compare verifies a prior protected-file snapshot.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def protected(artifacts):
    paths = [p for p in artifacts.rglob("*") if p.is_file()
             and p.suffix in {".json", ".jsonl", ".safetensors", ".npy", ".npz", ".sqlite"}
             and not {"logs", "package", "notebook_validation"}.intersection(p.relative_to(artifacts).parts)]
    legacy = ROOT / "output/entity_extraction_qwen25_3b/adapter"
    result = {"portfolio/" + p.relative_to(artifacts).as_posix(): sha(p) for p in sorted(paths)}
    result.update({"legacy/" + p.name: sha(p) for p in sorted(legacy.glob("*")) if p.is_file()})
    return result


def audit(artifacts):
    before = protected(artifacts)
    corpus = rows(artifacts / "corpus.jsonl")
    training = rows(artifacts / "extraction_training.jsonl")
    reviews = rows(artifacts / "review/extraction.jsonl")
    ranking = rows(artifacts / "review/ranking.jsonl")
    roles = read(artifacts / "review/extraction_roles.json")
    groups = defaultdict(set)
    for article in corpus:
        groups[article["group_id"]].add(article["split"])
    reviewed_groups = {r["group_id"] for r in reviews}
    by_id = {r["article_id"]: r for r in corpus}
    failures = rows(artifacts / "release_failures.jsonl")
    rejections = rows(artifacts / "training_rejections.jsonl")
    categories = ("countries", "companies", "organizations", "profiles", "systems", "topics")
    db = sqlite3.connect((artifacts / "release.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        # Aggregate operational release records, not held-out extraction predictions.
        indexed = [json.loads(r[0]) for r in db.execute("SELECT payload FROM articles")]
        analysis_count = db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    finally:
        db.close()
    runs = {}
    for folder in sorted((artifacts / "runs").iterdir()):
        if not (folder / "manifest.json").exists():
            continue
        manifest = read(folder / "manifest.json")
        keys = ("status", "stage", "dataset_examples", "encoded_sequences", "optimizer_steps",
                "unique_articles_seen", "supervised_tokens", "maximum_weight_change", "elapsed_seconds",
                "dataset_hash", "base_revision", "adapter_sha256", "promotion_status")
        runs[folder.name] = {k: manifest.get(k) for k in keys}
        weights = folder / "adapter/adapter_model.safetensors"
        runs[folder.name]["adapter_hash_matches_manifest"] = weights.exists() and sha(weights) == manifest.get("adapter_sha256")
    prediction_counts = {}
    for path in sorted((artifacts / "evaluation").glob("*/predictions.jsonl")):
        with path.open("rb") as stream:
            prediction_counts[path.parent.name] = sum(bool(line.strip()) for line in stream)
    budget = read(artifacts / "processing_budget.json")
    release = read(artifacts / "release_manifest.json")
    current = [r for r in indexed if r["pipeline_identity"] == release["pipeline_identity"]]
    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "policy": "No inference, training, label edits or held-out prediction inspection; release records aggregated only.",
        "corpus": {"articles": len(corpus), "languages": dict(Counter(r["language"] for r in corpus)),
                   "splits": dict(Counter(r["split"] for r in corpus)),
                   "duplicate_article_ids": len(corpus) - len(by_id),
                   "duplicate_content_hashes": len(corpus) - len({r["content_hash"] for r in corpus}),
                   "groups_crossing_splits": sum(len(s) > 1 for s in groups.values())},
        "training": {"accepted": len(training), "languages": dict(Counter(r["language"] for r in training)),
                     "review_statuses": dict(Counter(r["review_status"] for r in training)),
                     "non_train_rows": sum(r["split"] != "train" for r in training),
                     "frozen_review_group_overlap": sum(r["group_id"] in reviewed_groups for r in training),
                     "source_hash_mismatches": sum(by_id.get(r["article_id"], {}).get("content_hash") != r["source_content_hash"] for r in training),
                     "nonliteral_entity_values": sum(v not in r["text"] for r in training for c in categories for v in r["labels"][c]),
                     "nonempty_fields": {c: sum(bool(r["labels"][c]) for r in training) for c in categories},
                     "recorded_rejections": len(rejections),
                     "rejection_reasons": dict(Counter(r["reason"] for r in rejections)),
                     "rejection_log_limit": "The current preparer overwrites rejection history on resume; this is not guaranteed cumulative."},
        "runs": runs,
        "review": {"extraction_items": len(reviews), "extraction_statuses": dict(Counter(r["review_status"] for r in reviews)),
                   "extraction_roles": dict(Counter(roles.values())),
                   "recommendation_items": len(ranking), "recommendation_statuses": dict(Counter(r["review_status"] for r in ranking)),
                   "recommendation_roles": dict(Counter(r["role"] for r in ranking)),
                   "recommendation_queries": len({r["query_id"] for r in ranking})},
        "opaque_prediction_line_counts": prediction_counts,
        "release": {"sqlite_check": integrity, "indexed_articles": len(indexed), "cached_analyses": analysis_count,
                    "current_version_articles": len(current),
                    "current_version_languages": dict(Counter(r["article"].get("language") for r in current)),
                    "current_version_no_grounded_mentions": sum(not r["evidence"] for r in current),
                    "model_run": release.get("model", {}).get("run_id"),
                    "languages": dict(Counter(r["article"].get("language") for r in indexed)),
                    "pipeline_identities": dict(Counter(r["pipeline_identity"] for r in indexed)),
                    "no_grounded_mentions": sum(not r["evidence"] for r in indexed),
                    "review_statuses": dict(Counter(r["review_status"] for r in indexed)),
                    "frozen_extraction_overlap": len({r["article"]["article_id"] for r in indexed} & set(roles)),
                    "missing_frozen_ranking_articles": sorted({r["article"]["article_id"] for r in ranking} - {r["article"]["article_id"] for r in current}),
                    "recorded_failed_attempts": len(failures),
                    "failure_reasons": dict(Counter(r["error"] for r in failures))},
        "budget": {**budget, "ledger_remaining_seconds": budget["limit_seconds"] - budget["consumed_seconds"],
                   "limitation": "Recovered time is a durable-progress lower bound; unmetered early preparation is not proven accounted for."},
        "portfolio_artifact_bytes": sum(p.stat().st_size for p in artifacts.rglob("*") if p.is_file()),
        "protected_sha256": before,
    }
    after = protected(artifacts)
    if before != after:
        raise RuntimeError("Artifacts changed during audit; stop workers and repeat the read-only audit")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "output/portfolio")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = audit(args.artifacts)
    if args.compare:
        previous = read(args.compare)["protected_sha256"]
        current = report["protected_sha256"]
        changes = sorted(k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k))
        if changes:
            raise SystemExit("Protected artifacts changed: " + ", ".join(changes))
        report["comparison"] = "All protected files unchanged"
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(args.artifacts.resolve()) or output.is_relative_to((ROOT / "output/entity_extraction_qwen25_3b").resolve()):
            raise SystemExit("Write audit reports outside protected artifact directories")
        if output.exists() and output == (args.compare.resolve() if args.compare else None):
            raise SystemExit("Do not overwrite the baseline being compared")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "protected_sha256"}, indent=2))


if __name__ == "__main__":
    main()
