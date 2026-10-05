"""Atomic, attributed human-review updates for the frozen evaluation batch."""
import json
from datetime import datetime, timezone

from filelock import FileLock

from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3
from src.llm_extractor import validate_extraction_response
from src.news_evaluation import validate_ranking_review, validate_review
from src.portfolio_config import ARTIFACTS, digest, read_json, read_jsonl, write_jsonl


def empty_labels():
    return {key: [] for key in (*CURRENT_FILTER_CATEGORIES, "relationships")}


def require_open_review():
    if read_json(ARTIFACTS / "phase2/completion.json", {}).get("status") == "complete":
        raise ValueError("These reviews are sealed for evaluation. Corrections require an explicit new reference version.")


def _reviewer(value):
    value = value.strip() if isinstance(value, str) else ""
    if len(value) < 2:
        raise ValueError("Enter your name or stable reviewer ID before saving")
    return value


def _append_event(kind, item_id, reviewer, before, after):
    path = ARTIFACTS / "review/review_events.jsonl"
    events = read_jsonl(path) if path.exists() else []
    events.append({"kind": kind, "item_id": item_id, "reviewed_by": reviewer,
                   "reviewed_at": after["reviewed_at"], "before_hash": digest(before),
                   "after_hash": digest(after)})
    write_jsonl(path, events)


def save_extraction_review(article_id, labels, reviewer, scope="named_entities_v3"):
    reviewer = _reviewer(reviewer)
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Labels are not valid JSON: {exc}") from exc
    validate_extraction_response(labels, EXTRACTION_SCHEMA_V3)
    if scope == "named_entities_v3" and (labels["topics"] or labels["relationships"]):
        raise ValueError("The named-entity protocol does not collect topics or relationships")
    if scope not in {"full_schema_v3", "named_entities_v3"}:
        raise ValueError("Unknown extraction review protocol")
    path = ARTIFACTS / "review/extraction.jsonl"
    with FileLock(str(path) + ".lock"):
        require_open_review()
        records = read_jsonl(path)
        validate_review(records)
        matches = [row for row in records if row["article_id"] == article_id]
        if len(matches) != 1:
            raise ValueError("Article is not a unique frozen review item")
        record = matches[0]
        if record["review_status"] == "human_reviewed" and record.get("reviewed_by") != reviewer:
            raise ValueError(f"This item was already reviewed by {record.get('reviewed_by')}; choose another item")
        source = record["title"] + "\n\n" + record["text"]
        if any(value not in source for category in CURRENT_FILTER_CATEGORIES for value in labels[category]):
            raise ValueError("Every entity label must be copied exactly from the title or article")
        if any(relation[key] not in source for relation in labels["relationships"] for key in ("subject", "object")):
            raise ValueError("Every relationship subject and object must be copied from the article")
        before = dict(record)
        record.update(labels=labels, review_status="human_reviewed", reviewed_by=reviewer,
                      reviewed_at=datetime.now(timezone.utc).isoformat(), review_scope=scope)
        validate_review(records)
        write_jsonl(path, records)
        _append_event("extraction", article_id, reviewer, before, record)
    return sum(row["review_status"] == "human_reviewed" for row in records)


def save_ranking_review(query_id, article_id, relevance, reviewer):
    reviewer = _reviewer(reviewer)
    if isinstance(relevance, bool) or relevance not in (0, 1, 2):
        raise ValueError("Choose integer 0, 1 or 2")
    path = ARTIFACTS / "review/ranking.jsonl"
    with FileLock(str(path) + ".lock"):
        require_open_review()
        records = read_jsonl(path)
        validate_ranking_review(records)
        matches = [row for row in records if row["query_id"] == query_id
                   and row["article"]["article_id"] == article_id]
        if len(matches) != 1:
            raise ValueError("Query/article pair is not a unique frozen review item")
        record = matches[0]
        if record["review_status"] == "human_reviewed" and record.get("reviewed_by") != reviewer:
            raise ValueError(f"This item was already reviewed by {record.get('reviewed_by')}; choose another item")
        before = dict(record)
        record.update(relevance=relevance, review_status="human_reviewed", reviewed_by=reviewer,
                      reviewed_at=datetime.now(timezone.utc).isoformat())
        validate_ranking_review(records)
        write_jsonl(path, records)
        _append_event("ranking", f"{query_id}:{article_id}", reviewer, before, record)
    return sum(row["review_status"] == "human_reviewed" for row in records)
