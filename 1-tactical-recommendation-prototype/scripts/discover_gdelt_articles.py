import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = (
    '("Saudi Arabia" OR UAE OR Qatar OR Egypt OR Jordan OR Iran OR Israel) '
    "(defense OR defence OR military OR missile OR drone OR radar)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover recent MENA defence articles through GDELT without copying article bodies."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--timespan", default="3months")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "discovery" / "gdelt_candidates.jsonl",
    )
    return parser.parse_args()


def _candidate_id(url):
    return "gdelt_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _request_json(url, attempts=3):
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": "TacticalRecommendationPrototype/1.0"})
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("GDELT request failed after all retry attempts")


def discover(query, max_records, timespan):
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(max(max_records, 1), 250),
        "timespan": timespan,
        "format": "json",
        "sort": "datedesc",
    }
    payload = _request_json(f"{API_URL}?{urlencode(params)}")

    discovered_at = datetime.now(timezone.utc).isoformat()
    candidates = []
    for article in payload.get("articles", []):
        url = article.get("url")
        title = article.get("title")
        if not url or not title:
            continue
        candidates.append(
            {
                "candidate_id": _candidate_id(url),
                "source_id": "gdelt_discovery",
                "source_url": url,
                "title": title,
                "seen_at": article.get("seendate"),
                "language": article.get("language"),
                "source_country": article.get("sourcecountry"),
                "publisher_domain": article.get("domain"),
                "discovered_at": discovered_at,
                "article_text": None,
                "rights_review_status": "pending",
                "allowed_for_training": False,
            }
        )
    return candidates


def write_candidates(candidates, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    try:
        candidates = discover(args.query, args.max_records, args.timespan)
    except HTTPError as exc:
        raise SystemExit(
            f"GDELT returned HTTP {exc.code}. Wait for its public API rate limit to reset and rerun."
        ) from exc
    write_candidates(candidates, args.output)
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "output": str(args.output),
                "training_approved": 0,
                "note": "Discovery metadata only; publisher article text requires a separate rights review.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
