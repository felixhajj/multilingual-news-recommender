"""Reproducible metrics; incomplete hyperlinks are never treated as exhaustive gold."""
import math
from datetime import datetime, timezone

from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3, categories_for_schema
from src.llm_extractor import parse_first_json_object, validate_extraction_response
from src.portfolio_config import ARTIFACTS, digest, read_json, read_jsonl, write_json, write_jsonl

EXTRACTION_IMMUTABLE_FIELDS = ("article_id", "source_url", "title", "text", "language",
                               "content_hash", "group_id", "instructions")
RANKING_IMMUTABLE_FIELDS = ("query_id", "interest", "article", "required_filters", "role", "instructions")
ENTITY_CATEGORIES = tuple(category for category in CURRENT_FILTER_CATEGORIES if category != "topics")


def extraction_review_identity(records, roles):
    return digest({
        "items": [{key: row.get(key) for key in EXTRACTION_IMMUTABLE_FIELDS} for row in records],
        "roles": roles,
    })


def extraction_metrics(predictions, references, schema_version=EXTRACTION_SCHEMA_V3):
    categories = tuple(c for c in categories_for_schema(schema_version) if c != "topics")
    gold = {r["article_id"]: r for r in references}
    if any(r.get("review_status") != "human_reviewed" for r in references):
        raise ValueError("Extraction F1 requires human-reviewed, exhaustive references")
    if len({p["article_id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate predictions")
    if {p["article_id"] for p in predictions} != set(gold):
        raise ValueError("Predictions and references must cover the same articles")
    tp = fp = fn = valid = schema = 0
    for prediction in predictions:
        expected = gold[prediction["article_id"]]["labels"]
        validate_extraction_response(expected, EXTRACTION_SCHEMA_V3)
        try:
            chunks = [parse_first_json_object(value) for value in prediction.get("raw_outputs", [prediction.get("raw_output", "")])]
            if not chunks:
                raise ValueError("Missing model output")
            valid += 1
            for chunk in chunks:
                validate_extraction_response(chunk, schema_version)
            schema += 1
            parsed = {category: list({v for chunk in chunks for v in chunk[category]}) for category in categories}
        except ValueError:
            parsed = {}
        for category in categories:
            actual_values = parsed.get(category, [])
            if not isinstance(actual_values, list):
                actual_values = []
            actual = {v.casefold().strip() for v in actual_values if isinstance(v, str)}
            target = {v.casefold().strip() for v in expected[category]}
            tp += len(actual & target)
            fp += len(actual - target)
            fn += len(target - actual)
    n = len(predictions)
    return {"examples": n, "json_parse_validity": valid/max(1,n), "schema_validity": schema/max(1,n),
            "precision": tp/max(1,tp+fp), "recall": tp/max(1,tp+fn), "micro_f1": 2*tp/max(1,2*tp+fp+fn),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "schema_version": schema_version,
            "scored_categories": list(categories),
            "scope": "exact normalized surface named entities; topics and relationships are schema-validated but not scored"}


def review_roles(records):
    path = ARTIFACTS / "review" / "extraction_roles.json"
    existing = read_json(path)
    if existing:
        if set(existing) != {r["article_id"] for r in records}:
            raise ValueError("Evaluation role IDs changed")
        if any(role not in {"validation", "test"} for role in existing.values()):
            raise ValueError("Evaluation roles must be validation or test")
        return existing
    # Freeze roles before any labels/predictions are inspected: ten validation, twenty test.
    if any(r.get("review_status") == "human_reviewed" for r in records):
        raise ValueError("Freeze evaluation roles before reviewing labels")
    roles = {}
    for language in ("en", "ar"):
        subset = sorted([r for r in records if r["language"] == language], key=lambda r: digest(r["article_id"]))
        roles.update({r["article_id"]: "validation" if i < 5 else "test" for i, r in enumerate(subset)})
    write_json(path, roles)
    return roles


def ndcg_at_k(ranked_ids, relevance, k=5):
    def dcg(values):
        return sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(values))
    actual = dcg([relevance.get(article_id, 0) for article_id in ranked_ids[:k]])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual/ideal if ideal else 0.0


def ranking_metrics(ranked_ids, relevance, k=5):
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Ranked article IDs must be unique")
    if any(isinstance(grade, bool) or grade not in (0, 1, 2) for grade in relevance.values()):
        raise ValueError("Relevance grades must be integers 0, 1 or 2")
    if set(ranked_ids) - set(relevance):
        raise ValueError("Unjudged articles must not be silently scored as irrelevant")
    relevant = {key for key, grade in relevance.items() if grade > 0}
    return {"ndcg_at_5": ndcg_at_k(ranked_ids, relevance, k),
            "recall_at_5": len(set(ranked_ids[:k]) & relevant)/len(relevant) if relevant else 0,
            "judged_articles": len(relevance)}


def freeze_extraction_review(records):
    path = ARTIFACTS / "review" / "extraction.jsonl"
    if path.exists():
        return read_jsonl(path)
    selected = []
    for language in ("en", "ar"):
        candidates = sorted([r for r in records if r["language"] == language and r["split"] == "test"
                             and 300 <= len(r["body"]) <= 2600], key=lambda r: digest(r["article_id"]))[:15]
        selected.extend({"article_id": r["article_id"], "source_url": r["source_url"],
                         "title": r["title"], "text": r["body"], "language": language,
                         "content_hash": r["content_hash"], "group_id": r["group_id"],
                         "review_status": "pending", "reviewed_by": None, "labels": None,
                         "instructions": "Extract all explicitly present mentions using original spelling. Empty arrays mean checked and absent. Do not infer relationships."}
                        for r in candidates)
    write_jsonl(path, selected)
    write_json(ARTIFACTS / "review" / "frozen_extraction_manifest.json", {
        "examples": len(selected), "items": {r["article_id"]: r["content_hash"] for r in selected},
        "group_ids": sorted({r["group_id"] for r in selected}), "selection_version": digest(selected)})
    return selected


def validate_review(records):
    frozen = read_json(ARTIFACTS / "review" / "frozen_extraction_manifest.json")
    ids = [r["article_id"] for r in records]
    if len(ids) != len(set(ids)) or set(frozen["items"]) != set(ids):
        raise ValueError("The frozen review set must not be changed")
    roles = review_roles(records)
    integrity = (read_json(ARTIFACTS / "review" / "extraction_integrity_v3.json")
                 or read_json(ARTIFACTS / "review" / "extraction_integrity_v2.json"))
    if integrity and extraction_review_identity(records, roles) != integrity["identity"]:
        raise ValueError("Frozen extraction titles, bodies, metadata or roles changed")
    for record in records:
        if digest(record["text"]) != frozen["items"][record["article_id"]]:
            raise ValueError("Reviewed article text changed")
        if record["review_status"] == "human_reviewed":
            if not record.get("reviewed_by") or not record.get("reviewed_at"):
                raise ValueError("Reviewer attribution and timestamp are required")
            validate_extraction_response(record["labels"], EXTRACTION_SCHEMA_V3)
            source = record["title"] + "\n\n" + record["text"]
            if any(value not in source for category in CURRENT_FILTER_CATEGORIES for value in record["labels"][category]):
                raise ValueError("Reviewed entity labels must be literal source passages")
            if any(relation[key] not in source for relation in record["labels"]["relationships"] for key in ("subject", "object")):
                raise ValueError("Relationship subjects and objects must be literal source passages")
            if record.get("review_scope") == "named_entities_v3" and (record["labels"]["topics"] or record["labels"]["relationships"]):
                raise ValueError("Named-entity review must leave unscored topics and relationships empty")
        elif record["review_status"] != "pending":
            raise ValueError("Review status must be pending or human_reviewed")
    return sum(r["review_status"] == "human_reviewed" for r in records)


def ranking_review_identity(records):
    return digest([{key: row.get(key) for key in RANKING_IMMUTABLE_FIELDS} for row in records])


def validate_ranking_review(records):
    frozen = read_json(ARTIFACTS / "review" / "frozen_ranking_manifest.json")
    integrity = read_json(ARTIFACTS / "review" / "ranking_integrity_v2.json")
    expected = integrity.get("identity") if integrity else frozen.get("items") if frozen else None
    if not expected or ranking_review_identity(records) != expected:
        raise ValueError("Frozen ranking inputs were changed")
    keys = [(row["query_id"], row["article"]["article_id"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen ranking judgments contain duplicate query/article pairs")
    for row in records:
        if row["review_status"] == "human_reviewed":
            if (isinstance(row["relevance"], bool) or row["relevance"] not in (0, 1, 2)
                    or not row.get("reviewed_by") or not row.get("reviewed_at")):
                raise ValueError("A reviewed judgment needs a timestamped, attributed integer 0, 1 or 2 grade")
        elif row["review_status"] != "pending" or row.get("relevance") is not None:
            raise ValueError("Pending ranking judgments must not contain a relevance grade")
    return sum(row["review_status"] == "human_reviewed" for row in records)


def ensure_review_integrity_manifests():
    """Upgrade frozen inputs with full immutable identities without changing item selection."""
    directory = ARTIFACTS / "review"
    extraction = read_jsonl(directory / "extraction.jsonl")
    roles = review_roles(extraction)
    legacy = read_json(directory / "frozen_extraction_manifest.json")
    if not legacy or set(legacy["items"]) != {r["article_id"] for r in extraction}:
        raise ValueError("Legacy extraction freeze is missing or inconsistent")
    if any(digest(r["text"]) != legacy["items"][r["article_id"]] for r in extraction):
        raise ValueError("Extraction bodies changed before integrity upgrade")
    extraction_manifest = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": extraction_review_identity(extraction, roles),
        "items": len(extraction),
        "roles": {role: sum(value == role for value in roles.values()) for role in ("validation", "test")},
        "legacy_selection_version": legacy.get("selection_version"),
        "policy": "Identity binds title, body, source metadata, instructions and frozen role; judgments remain editable.",
    }
    extraction_path = directory / "extraction_integrity_v2.json"
    existing = read_json(extraction_path)
    if existing and existing["identity"] != extraction_manifest["identity"]:
        raise ValueError("Existing extraction integrity identity does not match frozen inputs")
    if not existing:
        write_json(extraction_path, extraction_manifest)

    ranking = read_jsonl(directory / "ranking.jsonl")
    legacy_ranking = read_json(directory / "frozen_ranking_manifest.json")
    # The historical identity omitted instructions; verify it before adding the stronger identity.
    old_identity = digest([{key: row[key] for key in ("query_id", "interest", "article", "required_filters", "role")}
                           for row in ranking])
    if not legacy_ranking or old_identity != legacy_ranking["items"]:
        raise ValueError("Legacy ranking freeze is missing or inconsistent")
    ranking_manifest = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": ranking_review_identity(ranking),
        "judgments": len(ranking),
        "queries": len({r["query_id"] for r in ranking}),
        "roles": {role: sum(r["role"] == role for r in ranking) for role in ("validation", "test")},
        "legacy_identity": legacy_ranking["items"],
        "policy": "Identity binds queries, full article inputs, filters, instructions and roles; grades remain editable.",
    }
    ranking_path = directory / "ranking_integrity_v2.json"
    existing = read_json(ranking_path)
    if existing and existing["identity"] != ranking_manifest["identity"]:
        raise ValueError("Existing ranking integrity identity does not match frozen inputs")
    if not existing:
        write_json(ranking_path, ranking_manifest)
    return extraction_manifest, ranking_manifest
