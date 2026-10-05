"""Freeze 30 judgments in three ten-article pools; unjudged items are not negatives."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.news_pipeline import NewsPipeline
from src.news_evaluation import ranking_review_identity
from src.portfolio_config import ARTIFACTS, digest, write_json, write_jsonl

if __name__ == "__main__":
    path = ARTIFACTS / "review" / "ranking.jsonl"
    if path.exists():
        raise SystemExit("Ranking review already frozen; refusing to replace it")
    pipeline = NewsPipeline()
    queries = [("Iran nuclear diplomacy and international sanctions", {"countries": ["Iran"]}),
               ("Lebanon and regional security in the Middle East", {"countries": ["Lebanon"]}),
               ("Elections, political reform and diplomatic relations", {})]
    rows = []
    for index, (query, required) in enumerate(queries):
        pools = [pipeline.recommend(query, required, method=method, limit=10)["results"] for method in ("keyword", "e5", "hybrid")]
        selected, seen = [], set()
        for rank in range(10):
            for pool in pools:
                if rank < len(pool):
                    article = pool[rank]["article"]
                    if article["article_id"] not in seen and len(selected) < 10:
                        selected.append(article)
                        seen.add(article["article_id"])
        if len(selected) < 10:
            raise SystemExit("Prepare at least ten release articles before freezing ranking judgments")
        for article in selected:
            rows.append({"query_id": f"query-{index+1}", "interest": query, "article": article,
                         "required_filters": required, "role": "validation" if index == 0 else "test",
                         "relevance": None, "review_status": "pending", "reviewed_by": None,
                         "instructions": "Grade 0=irrelevant, 1=partially relevant, 2=directly relevant"})
    write_jsonl(path, rows)
    write_json(ARTIFACTS / "review" / "frozen_ranking_manifest.json", {"items": ranking_review_identity(rows), "judgments": len(rows),
        "pipeline_identity": pipeline.identity,
        "scope": "Three ten-article judged pools, not an exhaustive full-corpus retrieval benchmark"})
