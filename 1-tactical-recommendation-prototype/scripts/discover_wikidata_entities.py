import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SPARQL_URL = "https://query.wikidata.org/sparql"
MENA_COUNTRY_QIDS = (
    "Q851",   # Saudi Arabia
    "Q878",   # United Arab Emirates
    "Q846",   # Qatar
    "Q842",   # Oman
    "Q398",   # Bahrain
    "Q817",   # Kuwait
    "Q810",   # Jordan
    "Q822",   # Lebanon
    "Q79",    # Egypt
    "Q262",   # Algeria
    "Q1028",  # Morocco
    "Q948",   # Tunisia
    "Q1016",  # Libya
    "Q805",   # Yemen
    "Q858",   # Syria
    "Q796",   # Iraq
    "Q794",   # Iran
    "Q801",   # Israel
    "Q219060", # Palestine
    "Q43",    # Turkey
)


CATEGORY_PATTERNS = {
    "company": """
        { ?item wdt:P31/wdt:P279* wd:Q1934969. }
        UNION { ?item wdt:P31/wdt:P279* wd:Q2538889. }
        UNION { ?item wdt:P452 wd:Q392933. }
    """,
    "system": "?item wdt:P31/wdt:P279* wd:Q728.",
    "profile": f"""
        VALUES ?country {{ {' '.join(f'wd:{qid}' for qid in MENA_COUNTRY_QIDS)} }}
        {{ ?country wdt:P35 ?item. }}
        UNION {{ ?country wdt:P6 ?item. }}
    """,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover multilingual defence entities and aliases from Wikidata CC0 data."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORY_PATTERNS),
        default=sorted(CATEGORY_PATTERNS),
    )
    parser.add_argument("--limit-per-category", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "entities" / "wikidata_candidates.jsonl",
    )
    return parser.parse_args()


def build_query(category, limit):
    if category not in CATEGORY_PATTERNS:
        raise ValueError(f"Unsupported category: {category}")
    safe_limit = min(max(int(limit), 1), 1000)
    return f"""
        SELECT ?item ?sitelinks
               (SAMPLE(?labelEnValue) AS ?labelEn)
               (SAMPLE(?labelArValue) AS ?labelAr)
               (GROUP_CONCAT(DISTINCT ?aliasEnValue; separator="|||") AS ?aliasesEn)
               (GROUP_CONCAT(DISTINCT ?aliasArValue; separator="|||") AS ?aliasesAr)
        WHERE {{
            {{
                SELECT DISTINCT ?item ?sitelinks
                WHERE {{
                    {CATEGORY_PATTERNS[category]}
                    ?item wikibase:sitelinks ?sitelinks.
                }}
                ORDER BY DESC(?sitelinks)
                LIMIT {safe_limit}
            }}
            OPTIONAL {{
                ?item rdfs:label ?labelEnValue.
                FILTER(LANG(?labelEnValue) = "en")
            }}
            OPTIONAL {{
                ?item rdfs:label ?labelArValue.
                FILTER(LANG(?labelArValue) = "ar")
            }}
            OPTIONAL {{
                ?item skos:altLabel ?aliasEnValue.
                FILTER(LANG(?aliasEnValue) = "en")
            }}
            OPTIONAL {{
                ?item skos:altLabel ?aliasArValue.
                FILTER(LANG(?aliasArValue) = "ar")
            }}
        }}
        GROUP BY ?item ?sitelinks
        ORDER BY DESC(?sitelinks)
    """


def _request_json(query, attempts=3):
    url = f"{SPARQL_URL}?{urlencode({'query': query, 'format': 'json'})}"
    for attempt in range(attempts):
        request = Request(
            url,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "TacticalRecommendationPrototype/1.0 (entity catalogue research)",
            },
        )
        try:
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in {429, 502, 503} or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("Wikidata request failed after all retry attempts")


def _binding_value(binding, key):
    return binding.get(key, {}).get("value", "").strip()


def bindings_to_candidates(bindings, category):
    discovered_at = datetime.now(timezone.utc).isoformat()
    candidates = []
    for binding in bindings:
        item_url = _binding_value(binding, "item")
        qid = item_url.rsplit("/", 1)[-1]
        label_en = _binding_value(binding, "labelEn")
        label_ar = _binding_value(binding, "labelAr")
        canonical_name = label_en or label_ar
        if not qid.startswith("Q") or not canonical_name:
            continue

        aliases = []
        for value in (label_ar, *_binding_value(binding, "aliasesEn").split("|||"), *_binding_value(binding, "aliasesAr").split("|||")):
            value = value.strip()
            if value and value.casefold() != canonical_name.casefold() and value not in aliases:
                aliases.append(value)

        candidates.append(
            {
                "entity_id": f"wikidata.{qid}",
                "wikidata_id": qid,
                "type": category,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "source_url": item_url,
                "license_id": "CC0-1.0",
                "review_status": "pending",
                "discovered_at": discovered_at,
            }
        )
    return candidates


def discover_category(category, limit):
    payload = _request_json(build_query(category, limit))
    return bindings_to_candidates(payload.get("results", {}).get("bindings", []), category)


def write_candidates(candidates, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    candidates = []
    for category in args.categories:
        try:
            candidates.extend(discover_category(category, args.limit_per_category))
        except HTTPError as exc:
            raise SystemExit(
                f"Wikidata returned HTTP {exc.code} while loading {category}; rerun after the public endpoint recovers."
            ) from exc

    write_candidates(candidates, args.output)
    counts = {category: sum(item["type"] == category for item in candidates) for category in args.categories}
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "categories": counts,
                "output": str(args.output),
                "review_status": "pending",
                "note": "Review category and aliases before merging candidates into data/entities.json.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
