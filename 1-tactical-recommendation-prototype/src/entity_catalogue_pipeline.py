import json
from datetime import datetime
from pathlib import Path


VALID_TYPES = {"country", "company", "organization", "profile", "system", "topic"}


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


def _validate_approved_candidate(candidate):
    if candidate.get("review_status") != "approved":
        return
    for field in ("reviewed_by", "reviewed_at"):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise ValueError(f"Approved entity candidates need a non-empty {field}")
    try:
        datetime.fromisoformat(candidate["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an ISO timestamp") from exc
    if candidate.get("type") not in VALID_TYPES:
        raise ValueError(f"Unsupported entity type: {candidate.get('type')}")
    if not isinstance(candidate.get("canonical_name"), str) or not candidate["canonical_name"].strip():
        raise ValueError("Approved entity candidates need a canonical_name")
    aliases = candidate.get("aliases")
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise ValueError("Approved entity candidates need an aliases list")


def promote_reviewed_entities(candidate_path, catalogue_path):
    catalogue_path = Path(catalogue_path)
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8-sig"))
    candidates = _load_jsonl(candidate_path)

    by_qid = {entity.get("wikidata_id"): entity for entity in catalogue if entity.get("wikidata_id")}
    by_name = {entity["canonical_name"].casefold(): entity for entity in catalogue}
    promoted = []
    skipped = []

    for candidate in candidates:
        _validate_approved_candidate(candidate)
        candidate_id = candidate.get("entity_id", "<missing>")
        if candidate.get("review_status") != "approved":
            skipped.append({"entity_id": candidate_id, "reason": "not approved"})
            continue

        entity = by_qid.get(candidate.get("wikidata_id")) or by_name.get(
            candidate["canonical_name"].casefold()
        )
        if entity is None:
            entity = {
                "entity_id": candidate_id,
                "wikidata_id": candidate.get("wikidata_id"),
                "type": candidate["type"],
                "canonical_name": candidate["canonical_name"],
                "aliases": [],
            }
            catalogue.append(entity)
            by_name[entity["canonical_name"].casefold()] = entity
            if entity.get("wikidata_id"):
                by_qid[entity["wikidata_id"]] = entity
        elif entity["type"] != candidate["type"]:
            skipped.append({"entity_id": candidate_id, "reason": "type conflicts with catalogue"})
            continue

        existing_aliases = {alias.casefold() for alias in entity["aliases"]}
        for alias in candidate["aliases"]:
            alias = alias.strip()
            if alias and alias.casefold() not in existing_aliases:
                entity["aliases"].append(alias)
                existing_aliases.add(alias.casefold())
        if candidate.get("wikidata_id") and not entity.get("wikidata_id"):
            entity["wikidata_id"] = candidate["wikidata_id"]
            by_qid[candidate["wikidata_id"]] = entity
        promoted.append(candidate_id)

    if promoted:
        temporary_path = catalogue_path.with_suffix(catalogue_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(catalogue_path)

    return promoted, skipped
