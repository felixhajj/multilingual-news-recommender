"""Reproduce retrieval metrics within frozen, fully judged pools."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.news_pipeline import NewsPipeline
from src.news_evaluation import ranking_metrics, validate_ranking_review
from src.portfolio_config import ARTIFACTS, digest, read_jsonl, write_json
from src.portfolio_reports import publish_evidence_snapshot


def main():
    rows = read_jsonl(ARTIFACTS / "review" / "ranking.jsonl")
    if validate_ranking_review(rows) != 30:
        raise ValueError("Review all 30 frozen judgments before claiming retrieval metrics")
    pipeline = NewsPipeline()
    output = {}
    for query_id in sorted({r["query_id"] for r in rows}):
        pool = [r for r in rows if r["query_id"] == query_id]
        query = pool[0]
        relevance = {r["article"]["article_id"]: r["relevance"] for r in pool}
        output[query_id] = {"role": query["role"], "methods": {}}
        for method in ("keyword", "e5", "hybrid"):
            ranked = pipeline.recommend(query["interest"], query["required_filters"], method=method,
                                        limit=10, candidate_ids=set(relevance))
            ids = [r["article"]["article_id"] for r in ranked["results"]]
            output[query_id]["methods"][method] = {"metrics": ranking_metrics(ids, relevance),
                                                   "ranked_ids": ids, "references": relevance}
    report = {"status": "evaluated", "reference_hash": digest(rows), "pipeline_identity": pipeline.identity,
              "scope": "Three ten-article judged pools; not exhaustive 5,000-article Recall", "queries": output}
    write_json(ARTIFACTS / "recommendation_evaluation.json", report)
    publish_evidence_snapshot()
    print(report)


if __name__ == "__main__":
    main()
