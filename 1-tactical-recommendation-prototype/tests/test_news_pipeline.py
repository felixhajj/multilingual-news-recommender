import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.news_evaluation import extraction_metrics, ranking_metrics
from src.news_pipeline import NewsPipeline, compare_filters, validate_article
from src.news_store import NewsStore
from src.portfolio_config import digest


def encode(texts):
    values = []
    for text in texts:
        row = np.asarray([text.lower().count("iran"), text.lower().count("lebanon"), 0.1], dtype=np.float32)
        values.append(row/np.linalg.norm(row))
    return np.stack(values)


class Backend:
    identity = {"model": "test", "revision": "v1"}
    def __init__(self):
        self.calls = 0
    def extract(self, text):
        self.calls += 1
        labels = {key: [] for key in ("countries", "companies", "organizations", "profiles", "systems", "topics", "relationships")}
        labels["countries"] = [name for name in ("Iran", "Lebanon") if name in text]
        return labels, [json.dumps(labels)]


class Linker:
    version = "trained-test-v1"
    def link(self, mention, context):
        return {"mention": mention, "canonical_name": mention, "entity_id": "Q"+mention,
                "status": "linked", "link_score": 0.95}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = NewsStore(Path(self.directory.name)/"test.sqlite")
        self.backend = Backend()
        self.pipeline = NewsPipeline(self.backend, Linker(), encode, self.store)
        self.article = {"title": "Iran news", "body": "Iran officials discussed international diplomacy.", "article_id": "one"}

    def test_same_content_reuses_inference_but_edited_text_does_not(self):
        first = self.pipeline.analyze_article(self.article, persist=True)
        cached = self.pipeline.analyze_article(self.article)
        changed = self.pipeline.analyze_article({**self.article, "body": "Lebanon officials discussed regional diplomacy."})
        self.assertEqual(self.backend.calls, 2)
        self.assertTrue(cached["cache_hit"])
        self.assertNotEqual(first["cache_key"], changed["cache_key"])

    def test_model_revision_invalidates_cache_and_release_index(self):
        self.pipeline.analyze_article(self.article, persist=True)
        backend = Backend()
        backend.identity = {"model": "test", "revision": "v2"}
        newer = NewsPipeline(backend, Linker(), encode, self.store)
        self.assertEqual(newer.recommend("Iran")["total_articles"], 0)
        newer.analyze_article(self.article)
        self.assertEqual(backend.calls, 1)

    def test_visitor_inputs_are_not_persisted(self):
        self.pipeline.analyze_article(self.article)
        self.assertEqual(self.store.collection(self.pipeline.embedding_version)[0], [])

    def test_failed_generation_never_inserts_mock_tags(self):
        with patch.object(self.backend, "extract", side_effect=ValueError("Invalid JSON")):
            with self.assertRaises(ValueError):
                self.pipeline.analyze_article({**self.article, "tags": {"countries": ["Iran"]}}, persist=True)
        self.assertFalse(self.store.collection(self.pipeline.embedding_version)[0])

    def test_ungrounded_names_are_reported_and_excluded_from_filters(self):
        labels, raw = self.backend.extract("Iran")
        labels["countries"].append("Invented Country")
        with patch.object(self.backend, "extract", return_value=(labels, raw)):
            result = self.pipeline.analyze_article(self.article)
        self.assertNotIn("Invented Country", result["tags"]["countries"])
        self.assertEqual(result["ungrounded_predictions"][0]["prediction"], "Invented Country")

    def test_changed_interest_changes_ranking(self):
        self.pipeline.analyze_article(self.article, persist=True)
        self.pipeline.analyze_article({"article_id": "two", "title": "Lebanon news", "body": "Lebanon discusses reforms and Lebanon elections."}, persist=True)
        self.assertEqual(self.pipeline.recommend("Iran")["results"][0]["article"]["article_id"], "one")
        self.assertEqual(self.pipeline.recommend("Lebanon")["results"][0]["article"]["article_id"], "two")

    def test_empty_index_returns_an_explicit_empty_state(self):
        result = self.pipeline.recommend("Iran diplomacy")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["total_articles"], 0)
        self.assertEqual(result["status"], "collection_not_prepared")

    def test_filter_logic_is_or_within_and_across_categories(self):
        required = {"countries": ["Iran", "Lebanon"], "companies": ["Acme"]}
        _, coverage, direct = compare_filters(required, {"countries": ["Lebanon"], "companies": ["Acme"]})
        self.assertEqual(coverage, 1)
        self.assertTrue(direct)
        self.assertFalse(compare_filters(required, {"countries": ["Iran"]})[2])
        self.assertFalse(compare_filters({}, {})[2])

    def test_bad_filter_shapes_and_article_lengths_are_rejected(self):
        for required in ({"countries": "Iran"}, {"unknown": []}, {"countries": [""]}):
            with self.assertRaises(ValueError):
                compare_filters(required, {})
        for text in ("", "short", "x"*15001):
            with self.assertRaises(ValueError):
                validate_article(text)

    def test_unknown_entity_is_preserved_without_fabricated_id(self):
        with patch.object(self.pipeline.linker, "link", return_value={"mention": "Iran", "status": "unresolved", "canonical_name": None, "entity_id": None}):
            result = self.pipeline.analyze_article(self.article)
        self.assertIsNone(result["entity_links"][0]["entity_id"])
        self.assertEqual(result["tags"]["countries"], ["Iran"])
        self.assertEqual(result["review_status"], "pending")

    def test_store_rejects_failed_or_non_normalized_records(self):
        result = self.pipeline.analyze_article(self.article)
        with self.assertRaises(ValueError):
            self.store.save(result, "encoder", [9, 0])
        with self.assertRaises(ValueError):
            self.store.save({**result, "status": "failed"}, "encoder", [1, 0])


class EvaluationTests(unittest.TestCase):
    def test_machine_labels_cannot_masquerade_as_gold(self):
        with self.assertRaises(ValueError):
            extraction_metrics([], [{"article_id": "a", "review_status": "machine_checked"}])

    def test_valid_json_is_not_the_same_as_valid_schema_or_correct_entities(self):
        labels, _ = Backend().extract("Iran")
        labels["locations"] = []
        reference = [{"article_id": "a", "review_status": "human_reviewed", "labels": labels}]
        metrics = extraction_metrics([{"article_id": "a", "raw_output": '{"wrong": []}'}], reference)
        self.assertEqual(metrics["json_parse_validity"], 1)
        self.assertEqual(metrics["schema_validity"], 0)
        self.assertEqual(metrics["micro_f1"], 0)

    def test_ndcg_penalizes_bad_order_and_unjudged_is_not_negative(self):
        grades = {"a": 2, "b": 1, "c": 0}
        good = ranking_metrics(["a", "b", "c"], grades)
        bad = ranking_metrics(["c", "b", "a"], grades)
        self.assertGreater(good["ndcg_at_5"], bad["ndcg_at_5"])
        with self.assertRaises(ValueError):
            ranking_metrics(["missing"], grades)

    def test_oversize_embedding_request_does_not_evict_its_own_results(self):
        from src import embeddings
        class FakeModel:
            def encode(self, texts, **kwargs):
                return np.asarray([[float(text), 1] for text in texts])
        with patch.object(embeddings, "VECTOR_CACHE_LIMIT", 2), patch.object(embeddings, "_get_model", return_value=FakeModel()):
            embeddings._VECTOR_CACHE.clear()
            embeddings.encode_texts(["1"])
            result = embeddings.encode_texts(["1", "2", "3", "4"])
            np.testing.assert_array_equal(result[:, 0], [1, 2, 3, 4])
            embeddings._VECTOR_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
