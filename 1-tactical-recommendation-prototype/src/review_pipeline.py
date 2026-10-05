import json
from datetime import datetime
from pathlib import Path

from src.extraction_data import load_extraction_examples, validate_extraction_example


EXTRACTION_FIELDS = ("example_id", "split", "language", "text", "labels")


def _load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def extraction_example_from_review(record):
    review_status = record.get("review_status")
    if review_status not in {"approved", "rejected", "pending"}:
        raise ValueError("review_status must be approved, rejected, or pending")
    if review_status != "approved":
        return None

    for field in ("reviewed_by", "reviewed_at"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Approved review records need a non-empty {field}")
    try:
        datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an ISO timestamp") from exc

    example = {key: record.get(key) for key in EXTRACTION_FIELDS}
    example["split"] = "train"
    validate_extraction_example(example)
    return example


def promote_reviewed_examples(queue_path, extraction_dataset_path):
    existing = load_extraction_examples(extraction_dataset_path)
    existing_ids = {example["example_id"] for example in existing}
    existing_text = {" ".join(example["text"].lower().split()) for example in existing}

    promoted = []
    skipped = []
    for record in _load_jsonl(queue_path):
        example = extraction_example_from_review(record)
        if example is None:
            skipped.append(
                {
                    "example_id": record.get("example_id", "<missing>"),
                    "reason": f"review_status is {record.get('review_status')}",
                }
            )
            continue

        normalized_text = " ".join(example["text"].lower().split())
        if example["example_id"] in existing_ids:
            skipped.append({"example_id": example["example_id"], "reason": "duplicate example_id"})
            continue
        if normalized_text in existing_text:
            skipped.append({"example_id": example["example_id"], "reason": "duplicate text"})
            continue

        promoted.append(example)
        existing_ids.add(example["example_id"])
        existing_text.add(normalized_text)

    if promoted:
        dataset_path = Path(extraction_dataset_path)
        temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for example in existing + promoted:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        temporary_path.replace(dataset_path)

    return promoted, skipped
