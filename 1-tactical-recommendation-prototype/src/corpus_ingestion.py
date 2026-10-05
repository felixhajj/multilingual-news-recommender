import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


VALID_RIGHTS_STATUSES = {"approved", "discovery_only", "requires_authorization"}
VALID_SPLITS = {"train", "validation", "test"}
VALID_REVIEW_STATUSES = {"approved", "pending", "rejected"}


def _read_jsonl(path):
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


def load_source_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_source_manifest(manifest)
    return manifest


def load_corpus_documents(path):
    documents = _read_jsonl(path)
    if not documents:
        raise ValueError("The domain corpus is empty")
    for document in documents:
        validate_corpus_document(document)
    return documents


def validate_source_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("The source manifest must use schema_version 1")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The source manifest needs a non-empty sources list")

    seen_ids = set()
    for source in sources:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Every corpus source needs a non-empty source_id")
        if source_id in seen_ids:
            raise ValueError(f"Duplicate corpus source_id: {source_id}")
        seen_ids.add(source_id)

        if source.get("rights_status") not in VALID_RIGHTS_STATUSES:
            raise ValueError(
                f"{source_id}: rights_status must be one of {sorted(VALID_RIGHTS_STATUSES)}"
            )
        if not isinstance(source.get("allowed_uses"), list):
            raise ValueError(f"{source_id}: allowed_uses must be a list")
        if source["rights_status"] == "approved" and "model_training" in source["allowed_uses"]:
            if not source.get("license_id"):
                raise ValueError(f"{source_id}: training-approved sources need a license_id")


def _validate_url(document_id, value):
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{document_id}: source_url must be an HTTP(S) URL")


def _validate_published_at(document_id, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{document_id}: published_at must be a non-empty ISO timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{document_id}: published_at must be an ISO timestamp") from exc


def validate_corpus_document(document):
    document_id = document.get("document_id", "<missing document_id>")
    required_strings = (
        "document_id",
        "source_id",
        "source_url",
        "published_at",
        "language",
        "title",
        "text",
        "review_status",
    )
    for field in required_strings:
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"{document_id}: {field} must be a non-empty string")

    _validate_url(document_id, document["source_url"])
    _validate_published_at(document_id, document["published_at"])

    if document["review_status"] not in VALID_REVIEW_STATUSES:
        raise ValueError(
            f"{document_id}: review_status must be one of {sorted(VALID_REVIEW_STATUSES)}"
        )
    if not isinstance(document.get("allowed_for_training"), bool):
        raise ValueError(f"{document_id}: allowed_for_training must be true or false")

    split = document.get("split")
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"{document_id}: split must be one of {sorted(VALID_SPLITS)}")


def sources_by_id(manifest):
    return {source["source_id"]: source for source in manifest["sources"]}


def stable_split(document_id):
    """Assign an 80/10/10 split that stays stable as the corpus grows."""
    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def content_fingerprint(document):
    normalized = " ".join(f"{document['title']} {document['text']}".lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def training_rejection_reasons(document, source):
    reasons = []
    if source is None:
        reasons.append("unknown source_id")
        return reasons
    if source["rights_status"] != "approved":
        reasons.append(f"source rights_status is {source['rights_status']}")
    if "model_training" not in source["allowed_uses"]:
        reasons.append("source does not allow model_training")
    if not document["allowed_for_training"]:
        reasons.append("document is not approved for training")
    if document["review_status"] != "approved":
        reasons.append(f"document review_status is {document['review_status']}")
    return reasons


def select_approved_training_documents(documents, manifest):
    source_map = sources_by_id(manifest)
    selected = []
    rejected = []
    seen_ids = set()
    seen_fingerprints = set()

    for document in documents:
        validate_corpus_document(document)
        document_id = document["document_id"]
        reasons = training_rejection_reasons(document, source_map.get(document["source_id"]))

        if document_id in seen_ids:
            reasons.append("duplicate document_id")
        seen_ids.add(document_id)

        fingerprint = content_fingerprint(document)
        if fingerprint in seen_fingerprints:
            reasons.append("duplicate article content")

        if reasons:
            rejected.append({"document_id": document_id, "reasons": reasons})
            continue

        seen_fingerprints.add(fingerprint)
        selected.append(
            {
                **document,
                "split": document.get("split") or stable_split(document_id),
                "license_id": source_map[document["source_id"]]["license_id"],
            }
        )

    return selected, rejected


def training_text(document):
    return f"Title: {document['title'].strip()}\n\n{document['text'].strip()}"


def write_training_corpus(documents, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            record = {
                "document_id": document["document_id"],
                "split": document["split"],
                "language": document["language"],
                "text": training_text(document),
                "provenance": {
                    "source_id": document["source_id"],
                    "source_url": document["source_url"],
                    "published_at": document["published_at"],
                    "license_id": document["license_id"],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def corpus_summary(documents, rejected):
    return {
        "approved_documents": len(documents),
        "rejected_documents": len(rejected),
        "splits": dict(Counter(document["split"] for document in documents)),
        "languages": dict(Counter(document["language"] for document in documents)),
    }
