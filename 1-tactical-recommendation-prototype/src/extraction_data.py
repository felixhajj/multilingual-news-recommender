import json
from collections import Counter
from pathlib import Path


EXTRACTION_SCHEMA_V2 = "extraction-v2"
EXTRACTION_SCHEMA_V3 = "extraction-v3-locations"

# The deployed legacy adapter and its historical examples use v2. New review,
# training and evaluation work uses v3; never pad v2 gold with empty locations.
LEGACY_FILTER_CATEGORIES = (
    "countries",
    "companies",
    "organizations",
    "profiles",
    "systems",
    "topics",
)
FILTER_CATEGORIES = LEGACY_FILTER_CATEGORIES
CURRENT_FILTER_CATEGORIES = (
    "countries",
    "locations",
    "companies",
    "organizations",
    "profiles",
    "systems",
    "topics",
)
SCHEMA_CATEGORIES = {
    EXTRACTION_SCHEMA_V2: LEGACY_FILTER_CATEGORIES,
    EXTRACTION_SCHEMA_V3: CURRENT_FILTER_CATEGORIES,
}
VALID_SPLITS = {"train", "validation", "test"}
SYSTEM_PROMPT_V2 = """You enrich news articles with structured filters.
Extract only information explicitly present in the article. Do not invent facts.
Use canonical English names when the alias is unambiguous; otherwise preserve the article wording.
Return valid JSON with countries, companies, organizations, profiles, systems, topics, and relationships.
Each relationship must contain subject, relation, and object. Return empty arrays when nothing is found."""
SYSTEM_PROMPT_V3 = """You extract structured facts from news articles.
Extract only information explicitly present in the article. Do not invent facts.
Preserve entity names exactly as written, including Arabic spelling.
Return valid JSON with countries, locations, companies, organizations, profiles, systems, topics, and relationships.
Countries are sovereign countries explicitly named in the article.
Locations are named non-country geographic places such as territories, cities, regions, seas, straits, borders, and named bases.
Do not classify nationality adjectives, generic directions, or unnamed places as locations.
Each relationship must contain subject, relation, and object. Return empty arrays when nothing is found."""
SYSTEM_PROMPT = SYSTEM_PROMPT_V2


def categories_for_schema(schema_version):
    try:
        return SCHEMA_CATEGORIES[schema_version]
    except KeyError as exc:
        raise ValueError(f"Unknown extraction schema: {schema_version}") from exc


def load_extraction_examples(path):
    examples = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    validate_extraction_examples(examples)
    return examples


def _validate_string_list(example_id, field, values):
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{example_id}: labels.{field} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{example_id}: labels.{field} contains duplicates")


def validate_extraction_example(example):
    example_id = example.get("example_id", "<missing example_id>")
    if not isinstance(example.get("example_id"), str) or not example["example_id"].strip():
        raise ValueError("Every extraction example needs a non-empty example_id")
    if example.get("split") not in VALID_SPLITS:
        raise ValueError(f"{example_id}: split must be one of {sorted(VALID_SPLITS)}")
    if not isinstance(example.get("language"), str) or not example["language"].strip():
        raise ValueError(f"{example_id}: language must be a non-empty string")
    if not isinstance(example.get("text"), str) or not example["text"].strip():
        raise ValueError(f"{example_id}: text must be a non-empty string")

    labels = example.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{example_id}: labels must be an object")
    schema_version = example.get("schema_version", EXTRACTION_SCHEMA_V2)
    categories = categories_for_schema(schema_version)
    expected_fields = set(categories) | {"relationships"}
    if set(labels) != expected_fields:
        raise ValueError(f"{example_id}: labels must contain exactly {sorted(expected_fields)}")

    for category in categories:
        _validate_string_list(example_id, category, labels[category])

    relationships = labels["relationships"]
    if not isinstance(relationships, list):
        raise ValueError(f"{example_id}: labels.relationships must be a list")
    for relationship in relationships:
        if not isinstance(relationship, dict) or set(relationship) != {"subject", "relation", "object"}:
            raise ValueError(
                f"{example_id}: every relationship needs subject, relation, and object"
            )
        if any(not isinstance(value, str) or not value.strip() for value in relationship.values()):
            raise ValueError(f"{example_id}: relationship values must be non-empty strings")


def validate_extraction_examples(examples):
    if not examples:
        raise ValueError("The extraction dataset is empty")

    seen_ids = set()
    split_counts = Counter()
    for example in examples:
        validate_extraction_example(example)
        example_id = example["example_id"]
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example_id: {example_id}")
        seen_ids.add(example_id)
        split_counts[example["split"]] += 1

    missing_splits = VALID_SPLITS.difference(split_counts)
    if missing_splits:
        raise ValueError(f"Dataset is missing splits: {sorted(missing_splits)}")
    return dict(split_counts)


def examples_by_split(examples):
    return {
        split: [example for example in examples if example["split"] == split]
        for split in ("train", "validation", "test")
    }


def format_extraction_response(labels):
    return json.dumps(labels, ensure_ascii=False, sort_keys=True)


def build_extraction_messages(example):
    schema_version = example.get("schema_version", EXTRACTION_SCHEMA_V2)
    prompt = SYSTEM_PROMPT_V3 if schema_version == EXTRACTION_SCHEMA_V3 else SYSTEM_PROMPT_V2
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": example["text"]},
        {"role": "assistant", "content": format_extraction_response(example["labels"])},
    ]
