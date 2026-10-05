import unittest
from types import SimpleNamespace

from src.news_pipeline import NewsPipeline
from src.portfolio_config import digest
from scripts.evaluate_selected_ranking import (check_analysis, failed_analysis,
                                               unique_judged_articles)
from scripts.phase4_live_app_check import distinct_interest_rankings_changed


class SelectedRankingEvaluationTests(unittest.TestCase):
    def test_live_interest_check_requires_distinct_queries_and_top_five_orders(self):
        first = {"query": "Iran diplomacy", "top_results": [
            {"article_id": f"iran-{index}"} for index in range(5)]}
        second = {"query": "Gaza aid", "top_results": [
            {"article_id": f"gaza-{index}"} for index in range(5)]}
        self.assertTrue(distinct_interest_rankings_changed([first, second]))
        self.assertFalse(distinct_interest_rankings_changed([first, first]))
        self.assertFalse(distinct_interest_rankings_changed([
            first, {"query": "Gaza aid", "top_results": first["top_results"]}]))

    def test_repeated_articles_across_frozen_queries_are_deduplicated(self):
        article = {"article_id": "en-1", "title": "Title", "body": "Article body"}
        rows = [{"article": article}, {"article": dict(article)}]
        self.assertEqual(unique_judged_articles(rows), {"en-1": article})

    def test_conflicting_frozen_article_versions_are_rejected(self):
        rows = [
            {"article": {"article_id": "en-1", "title": "Title", "body": "Original"}},
            {"article": {"article_id": "en-1", "title": "Title", "body": "Edited"}},
        ]
        with self.assertRaisesRegex(ValueError, "disagree"):
            unique_judged_articles(rows)

    def test_analysis_must_match_frozen_text_and_pipeline_cache_key(self):
        article = {"article_id": "en-1", "title": "Title", "body": "Article body"}
        pipeline_identity = "selected-pipeline"
        model_identity = {"run_id": "extraction-only-v3", "adapter_sha256": "adapter-hash"}
        record = {"status": "ready", "pipeline_identity": pipeline_identity,
                  "model": dict(model_identity), "article": dict(article),
                  "cache_key": digest({"title": article["title"], "body": article["body"],
                                        "pipeline": pipeline_identity})}
        check_analysis(record, article, pipeline_identity, model_identity)
        record["cache_key"] = "wrong"
        with self.assertRaisesRegex(ValueError, "cache key"):
            check_analysis(record, article, pipeline_identity, model_identity)

    def test_malformed_generation_is_preserved_as_failure_not_empty_prediction(self):
        article = {"article_id": "en-1", "title": "Title", "body": "Article body"}
        pipeline_identity = "selected-pipeline"
        model_identity = {"run_id": "extraction-only-v3", "adapter_sha256": "adapter-hash"}
        pipeline = SimpleNamespace(identity=pipeline_identity,
                                   backend=SimpleNamespace(identity=model_identity))
        error = ValueError("invalid relation type")
        error.raw_outputs = ['{"relationships":[{"subject":"A","relation":null}]}']
        record = failed_analysis(article, pipeline, error)
        check_analysis(record, article, pipeline_identity, model_identity)
        self.assertEqual(record["status"], "extraction_failed")
        self.assertEqual(record["failure"]["raw_outputs"], error.raw_outputs)
        self.assertNotIn("extraction", record)


if __name__ == "__main__":
    unittest.main()
