"""Prepare independent annotation aids from published hyperlinks and Wikidata metadata."""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio_config import ARTIFACTS, digest, read_json, read_jsonl, write_json

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
TYPE_HINTS = {
    "profiles": {"Q5"},
    "countries": {"Q6256", "Q3624078", "Q7275"},
    "locations": {
        "Q515", "Q3957", "Q532", "Q486972", "Q82794", "Q15642541",
        "Q2221906", "Q23442", "Q4022", "Q23397", "Q165", "Q9430",
    },
    "companies": {"Q4830453", "Q783794", "Q6881511", "Q891723"},
    "organizations": {"Q43229", "Q327333", "Q484652", "Q7210356", "Q7278"},
    "systems": {"Q728", "Q18643213", "Q11436", "Q12796", "Q11019", "Q15142889"},
}
DESCRIPTION_HINTS = {
    "countries": re.compile(r"\b(country|sovereign state|island country|landlocked state)\b", re.I),
    "locations": re.compile(
        r"\b(city|town|village|territor(?:y|ial)|geographic region|region of|sea|strait|"
        r"border|island|river|lake|ocean|desert|military base|refugee camp)\b", re.I),
    "companies": re.compile(r"\b(company|corporation|manufacturer|business|contractor|enterprise)\b", re.I),
    "organizations": re.compile(
        r"\b(organi[sz]ation|agency|political party|ministry|university|military unit|committee|"
        r"council|government|armed group|trade union|nonprofit|institution)\b", re.I),
    "systems": re.compile(
        r"\b(weapon|aircraft|missile|warship|vehicle|tank|rifle|satellite|military system|drone)\b", re.I),
    "profiles": re.compile(
        r"\b(politician|journalist|activist|diplomat|businessperson|military officer|statesman|human)\b", re.I),
}
IGNORE_DESCRIPTION = re.compile(
    r"\b(year|decade|century|massacre|battle|war|election|referendum|event|castle|building|"
    r"airport|road|disease|ethnic group|citizens|residents|office|position|title)\b",
    re.I,
)
GENERIC_SURFACE = re.compile(
    r"^(the\s+)?(government|prime minister|president|leader|coup|military|army|police|officials?|"
    r"minister|parliament|company|organi[sz]ation|country|city|state|council|suburb|region|territory)$",
    re.I,
)
NATIONALITY_ADJECTIVE = re.compile(r"(?:ian|ean|ican|ese|ish|i)$", re.I)
ARABIC_GENERIC_SURFACE = re.compile(
    r"^(?:ال)?(?:عاصمة|مدينة|منطقة|إقليم|دولة|بلد|مقاطعة|ضاحية|حكومة)$"
)
ARABIC_PLURAL_DEMONYM = re.compile(r"^ال.+(?:يين|يون)$")


def fetch_metadata(qids, cache):
    values = dict(cache.get("entities", {}))
    missing = sorted(set(qids) - set(values))
    session = requests.Session()
    session.headers["User-Agent"] = "MultilingualNewsRecommender/1.0 (portfolio annotation aid)"
    for offset in range(0, len(missing), 50):
        batch = missing[offset:offset + 50]
        response = session.get(WIKIDATA_API, params={"action": "wbgetentities", "format": "json",
            "ids": "|".join(batch), "props": "labels|descriptions|claims", "languages": "en|ar"}, timeout=(20, 60))
        response.raise_for_status()
        for qid, entity in response.json()["entities"].items():
            values[qid] = {"labels": {k: v["value"] for k, v in entity.get("labels", {}).items()},
                           "descriptions": {k: v["value"] for k, v in entity.get("descriptions", {}).items()},
                           "instance_of": [claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                                           for claim in entity.get("claims", {}).get("P31", [])
                                           if claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")]}
        time.sleep(0.1)
    return {"source": WIKIDATA_API, "license": "CC0", "fetched_at": datetime.now(timezone.utc).isoformat(),
            "entities": values}


def suggestion(metadata, surface=""):
    if (GENERIC_SURFACE.fullmatch(surface.strip())
            or ARABIC_GENERIC_SURFACE.fullmatch(surface.strip())):
        return "ignore", "generic_role_or_common_word"
    labels = {value.casefold() for value in metadata.get("labels", {}).values()}
    if surface.casefold() not in labels and ARABIC_PLURAL_DEMONYM.fullmatch(surface.strip()):
        return "ignore", "nationality_or_regional_adjective"
    if (surface.casefold() not in labels and NATIONALITY_ADJECTIVE.search(surface)
            and re.search(r"\b(country|state|continent|territor)",
                          " ".join(metadata.get("descriptions", {}).values()), re.I)):
        return "ignore", "nationality_or_regional_adjective"
    instances = set(metadata.get("instance_of", []))
    for category, qids in TYPE_HINTS.items():
        if instances & qids:
            return category, "direct_instance_hint"
    description = " ".join(metadata.get("descriptions", {}).values())
    if IGNORE_DESCRIPTION.search(description):
        return "ignore", "non_entity_type_hint"
    for category, pattern in DESCRIPTION_HINTS.items():
        if pattern.search(description):
            return category, "description_hint"
    return "ignore", "unclassified"


def surface_present(surface, text):
    """Match a published surface as a phrase, not inside a larger Latin word."""
    if not surface or surface not in text:
        return False
    if surface[0].isascii() and surface[-1].isascii() and surface[0].isalnum() and surface[-1].isalnum():
        return bool(re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", text))
    return True


def build_candidate_links(corpus, review):
    """Augment each article's links with matching published surfaces from the corpus."""
    surface_qids = defaultdict(Counter)
    for source in corpus.values():
        for mention in source["mentions"]:
            surface = mention["text"].strip()
            if (len(surface) >= 4 or (len(surface) >= 3 and not surface.isascii())) and not surface.isdigit():
                surface_qids[surface][mention["qid"]] += 1

    result = {}
    for row in review:
        source = corpus[row["article_id"]]
        direct = {(mention["text"], mention["qid"]): "article_hyperlink"
                  for mention in source["mentions"] if mention["text"] in row["text"]}
        direct_surfaces = {surface for surface, _ in direct}
        for surface, qids in surface_qids.items():
            if surface not in direct_surfaces and surface_present(surface, row["title"] + "\n" + row["text"]):
                direct[(surface, qids.most_common(1)[0][0])] = "corpus_hyperlink_surface"
        result[row["article_id"]] = direct
    return result


def main():
    corpus = {row["article_id"]: row for row in read_jsonl(ARTIFACTS / "corpus.jsonl")}
    review = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    candidate_links = build_candidate_links(corpus, review)
    qids = {qid for links in candidate_links.values() for _, qid in links}
    cache_path = ARTIFACTS / "review/wikidata_candidate_metadata.json"
    metadata = fetch_metadata(qids, read_json(cache_path, {}))
    write_json(cache_path, metadata)
    result = []
    for row in review:
        mentions = []
        seen = set()
        for (surface, qid), candidate_source in candidate_links[row["article_id"]].items():
            key = (surface, qid)
            if key in seen:
                continue
            seen.add(key)
            entity = metadata["entities"].get(qid, {})
            category, reason = suggestion(entity, surface)
            mentions.append({"mention": surface, "qid": qid,
                             "suggested_category": category, "suggestion_reason": reason,
                             "label": entity.get("labels", {}).get(row["language"])
                                      or entity.get("labels", {}).get("en"),
                             "description": entity.get("descriptions", {}).get(row["language"])
                                            or entity.get("descriptions", {}).get("en"),
                              "source": candidate_source + "_plus_wikidata_metadata"})
        mentions.sort(key=lambda item: (item["suggested_category"] == "ignore", item["mention"].casefold()))
        result.append({"article_id": row["article_id"], "content_hash": row["content_hash"],
                       "candidates": mentions})
    output = {"version": 3, "schema_version": "extraction-v3-locations",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "identity": digest(result), "items": result,
              "policy": "Annotation aids only. Direct and corpus-wide published hyperlink surfaces plus Wikidata type hints are independent of evaluated model predictions and require human confirmation."}
    write_json(ARTIFACTS / "review/extraction_candidates.json", output)
    write_json(ARTIFACTS / "review/annotation_protocol_v3.json", {
        "version": 3, "schema_version": "extraction-v3-locations",
        "primary_metric_categories": ["countries", "locations", "companies", "organizations", "profiles", "systems"],
        "not_human_scored": ["topics", "relationships"],
        "category_rule": "Countries are sovereign states. Locations are named non-country geography including territories, cities, regions, seas, straits, borders and named bases.",
        "reason": "Locations are required for geopolitical relevance. Topics are subjective and relationships are not used by ranking; both remain JSON schema-validity checks.",
        "candidate_identity": output["identity"], "candidate_source": output["policy"]})
    print(json.dumps({"articles": len(result), "candidate_mentions": sum(len(r["candidates"]) for r in result),
                      "suggestions": {category: sum(c["suggested_category"] == category for r in result for c in r["candidates"])
                                      for category in ("countries", "locations", "companies", "organizations", "profiles", "systems", "ignore")}}, indent=2))


if __name__ == "__main__":
    main()
