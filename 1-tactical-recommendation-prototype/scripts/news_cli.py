"""Analyze, index, recommend or explain through the application's shared pipeline."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.news_pipeline import NewsPipeline
from src.portfolio_config import read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["analyze", "ingest", "recommend", "explain"])
    parser.add_argument("--article", type=Path, help="JSON object with title/body and source attribution")
    parser.add_argument("--interest", default="")
    parser.add_argument("--filters", default="{}", help="JSON required categories")
    args = parser.parse_args()
    pipeline = NewsPipeline()
    required = json.loads(args.filters)
    if args.operation == "recommend":
        result = pipeline.recommend(args.interest, required)
    else:
        if not args.article:
            parser.error("--article is required")
        result = pipeline.analyze_article(read_json(args.article), persist=args.operation == "ingest")
        if args.operation == "explain":
            result = pipeline.explain(args.interest, result, required)
    print(json.dumps(result, ensure_ascii=False, indent=2))
