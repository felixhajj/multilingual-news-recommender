import re
import unicodedata
from copy import deepcopy


TAG_CATEGORIES = ("countries", "companies", "organizations", "profiles", "systems", "topics")
TYPE_TO_CATEGORY = {
    "country": "countries",
    "company": "companies",
    "organization": "organizations",
    "profile": "profiles",
    "system": "systems",
    "topic": "topics",
}
UNKNOWN_PATTERNS = (
    re.compile(r"(?<![\w-])[A-Z]{2,8}(?:-[A-Z0-9]{1,8})+(?![\w-])"),
    re.compile(r"(?<![\w-])[A-Z]{3,8}(?![\w-])"),
    re.compile(r"(?<![\w-])(?:[A-Z][A-Za-z]+\s+){1,3}[A-Z][A-Za-z]+(?![\w-])"),
)


def normalize_entity_text(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^\w\u0600-\u06ff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _contains_alias(text_normalized, alias):
    alias_normalized = normalize_entity_text(alias)
    if not alias_normalized:
        return False
    return f" {alias_normalized} " in f" {text_normalized} "


def _source_text(article):
    return "\n".join(
        str(article.get(field, "")) for field in ("title", "summary", "body") if article.get(field)
    )


def find_catalog_entities(text, catalogue):
    text_normalized = normalize_entity_text(text)
    matches = []
    for entity in catalogue:
        names = [entity["canonical_name"], *entity.get("aliases", [])]
        matched_aliases = sorted({name for name in names if _contains_alias(text_normalized, name)})
        if matched_aliases:
            matches.append(
                {
                    "entity_id": entity["entity_id"],
                    "type": entity["type"],
                    "canonical_name": entity["canonical_name"],
                    "matched_aliases": matched_aliases,
                    "confidence": 1.0,
                }
            )
    return matches


def find_unknown_candidates(text, matched_entities):
    known_names = {
        normalize_entity_text(name)
        for entity in matched_entities
        for name in [entity["canonical_name"], *entity.get("matched_aliases", [])]
    }
    candidates = {}
    for pattern in UNKNOWN_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            normalized = normalize_entity_text(value)
            overlaps_known_name = any(
                f" {known_name} " in f" {normalized} " or f" {normalized} " in f" {known_name} "
                for known_name in known_names
            )
            if overlaps_known_name or len(normalized) < 3:
                continue
            candidates[normalized] = {
                "value": value,
                "type": "unknown",
                "confidence": 0.35,
                "status": "needs_review",
            }
    return sorted(candidates.values(), key=lambda item: normalize_entity_text(item["value"]))


def enrich_article(article, catalogue):
    enriched = deepcopy(article)
    tags = {category: list(enriched.get("tags", {}).get(category, [])) for category in TAG_CATEGORIES}
    source_text = _source_text(enriched)
    entities = find_catalog_entities(source_text, catalogue)

    for entity in entities:
        category = TYPE_TO_CATEGORY.get(entity["type"])
        if category and entity["canonical_name"] not in tags[category]:
            tags[category].append(entity["canonical_name"])

    for values in tags.values():
        values.sort(key=normalize_entity_text)

    enriched["tags"] = tags
    enriched["entities"] = entities
    enriched["unknown_candidates"] = find_unknown_candidates(source_text, entities)
    return enriched


def enrich_articles(articles, catalogue):
    return [enrich_article(article, catalogue) for article in articles]
