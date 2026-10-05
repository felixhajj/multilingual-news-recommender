"""Offline weak-label derivation; never relaxes live extraction or evaluation."""
import copy

from src.extraction_data import CURRENT_FILTER_CATEGORIES
from src.extraction_data import EXTRACTION_SCHEMA_V3
from src.llm_extractor import parse_first_json_object, validate_extraction_response
from src.portfolio_config import digest

POLICY = "case-only-top-level-keys-v1"


def validate_weak_label(labels, source):
    validate_extraction_response(labels, EXTRACTION_SCHEMA_V3)
    if any(value not in source for category in CURRENT_FILTER_CATEGORIES for value in labels[category]):
        raise ValueError("nonliteral entity")
    if any(rel[key] not in source for rel in labels["relationships"] for key in ("subject", "object")):
        raise ValueError("nonliteral relationship")
    if not any(labels[category] for category in CURRENT_FILTER_CATEGORIES if category != "topics"):
        raise ValueError("no named entities; published positives make this an unsuitable training example")


def recover_case_only(candidate, attempt, expected_chunks, validate):
    if attempt["outcome"] != "rejected":
        raise ValueError("Recovery applies only to preserved rejected attempts")
    if attempt["source_hash"] != digest(candidate["text"]):
        raise ValueError("Source text changed")
    raw = attempt["raw_outputs"]
    if len(raw) != expected_chunks or expected_chunks != 1:
        raise ValueError("Incomplete chunk generation cannot label the full article")
    original = parse_first_json_object(raw[0])
    keys = list(original)
    expected = set(CURRENT_FILTER_CATEGORIES) | {"relationships"}
    lowered = [key.lower() for key in keys]
    if len(set(lowered)) != len(keys) or set(lowered) != expected:
        raise ValueError("Missing, unknown or colliding keys; no inference or padding allowed")
    if keys == lowered:
        raise ValueError("No case-only correction to apply")
    labels = {key.lower(): copy.deepcopy(value) for key, value in original.items()}
    validate(labels, candidate["text"])
    return labels, {"policy": POLICY, "original_outcome": "rejected",
                    "original_error": attempt["error"], "raw_schema_valid": False,
                    "raw_output_hash": digest(raw),
                    "key_changes": {key: key.lower() for key in keys if key != key.lower()},
                    "note": "Offline machine-assisted derivation, not a valid raw generation or human correction"}
