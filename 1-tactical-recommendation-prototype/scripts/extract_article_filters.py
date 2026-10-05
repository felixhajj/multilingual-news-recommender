import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_entity_catalogue  # noqa: E402
from src.llm_extractor import (  # noqa: E402
    QwenExtractionAdapter,
    article_extraction_text,
    merge_extraction_with_catalogue,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Tactical Report filters from articles with the trained Qwen QLoRA adapter."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "incoming_articles.json",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "output" / "entity_extraction_qwen25_3b" / "adapter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "extractions" / "incoming_article_filters.jsonl",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_articles(path):
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    args = parse_args()
    articles = load_articles(args.input)
    if args.limit is not None:
        articles = articles[: max(args.limit, 0)]

    extractor = QwenExtractionAdapter(args.adapter)
    catalogue = load_entity_catalogue(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for article in articles:
            response, raw_output = extractor.extract(article_extraction_text(article))
            labels, catalogue_entities, unknown_candidates = merge_extraction_with_catalogue(
                article,
                response,
                catalogue,
            )
            record = {
                "article_id": article.get("article_id"),
                "model": extractor.base_model_name,
                "adapter": str(args.adapter),
                "labels": labels,
                "model_labels": response,
                "catalogue_entities": catalogue_entities,
                "unknown_candidates": unknown_candidates,
                "raw_model_output": raw_output,
                "review_status": "pending",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Extracted {record['article_id']}")

    print(f"Wrote {len(articles)} pending-review extractions to {args.output}")


if __name__ == "__main__":
    main()
