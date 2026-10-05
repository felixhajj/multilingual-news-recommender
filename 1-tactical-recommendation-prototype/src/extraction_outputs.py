import json
from copy import deepcopy
from pathlib import Path

from src.extraction_data import FILTER_CATEGORIES
from src.llm_extractor import validate_extraction_response


def load_extraction_outputs(path):
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid extraction JSON on line {line_number}: {exc}") from exc
            if not record.get("article_id"):
                raise ValueError(f"Extraction line {line_number} is missing article_id")
            validate_extraction_response(record.get("labels"))
            records.append(record)
    return records


def apply_extraction_outputs(articles, extraction_records, include_pending=True):
    records_by_article = {record["article_id"]: record for record in extraction_records}
    enriched_articles = []

    for source_article in articles:
        article = deepcopy(source_article)
        record = records_by_article.get(article.get("article_id"))
        if record and (include_pending or record.get("review_status") == "approved"):
            tags = article.setdefault("tags", {category: [] for category in FILTER_CATEGORIES})
            for category in FILTER_CATEGORIES:
                existing = tags.setdefault(category, [])
                for value in record["labels"][category]:
                    if value not in existing:
                        existing.append(value)
                existing.sort(key=str.casefold)
            article["llm_extraction"] = {
                "model": record.get("model"),
                "review_status": record.get("review_status"),
                "relationships": record["labels"]["relationships"],
                "unknown_candidates": record.get("unknown_candidates", []),
            }
        enriched_articles.append(article)

    return enriched_articles
