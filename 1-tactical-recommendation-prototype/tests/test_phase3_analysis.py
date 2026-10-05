import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.phase3_analysis import immutable, ranking_report, lock_decisions, validation_diagnostics, POLICY
from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3
from src.news_pipeline import rank_records
from src.learned_linker import cached_training_vectors
from src.portfolio_config import write_json, read_json, file_digest


class AnalysisTests(unittest.TestCase):
    def test_immutable_results_cannot_be_overwritten_by_flattering_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            immutable(path, {"f1": .2})
            with self.assertRaises(ValueError):
                immutable(path, {"f1": .8})
            self.assertEqual(read_json(path)["f1"], .2)

    def test_shared_ranking_filters_are_not_semantic_vectors(self):
        records = [{"article": {"article_id": key, "title": key, "body": "news text"},
                    "tags": {"countries": [country]}, "cache_key": key}
                   for key, country in (("a", "Iran"), ("b", "France"))]
        vectors = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
        e5 = rank_records(records, vectors, "diplomacy", {"countries": ["Iran"]}, "e5", .75, np.array([1, 0]))
        hybrid = rank_records(records, vectors, "diplomacy", {"countries": ["Iran"]}, "hybrid", .75, np.array([1, 0]))
        self.assertEqual(e5[0]["article"]["article_id"], "b")
        self.assertEqual(hybrid[0]["article"]["article_id"], "a")

    def test_final_ranking_requires_selection_lock_before_reading_grades(self):
        with patch("scripts.phase3_analysis.verify_selection_lock", side_effect=ValueError("missing lock")), \
             patch("scripts.phase3_analysis.read_json") as read:
            with self.assertRaisesRegex(ValueError, "missing lock"):
                ranking_report("test")
            read.assert_not_called()

    def test_policy_discloses_legacy_index_and_reused_linker_holdout(self):
        self.assertIn("legacy", POLICY["ranking"]["scope"])
        self.assertIn("not a fresh", POLICY["final_linker_scope"])

    def test_encoding_resume_reuses_finished_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            vector = np.zeros((1, 768), dtype=np.float32)
            vector[0, 0] = 1
            with patch("src.learned_linker.encode_batch", side_effect=[vector, RuntimeError("outage")]), \
                 patch("src.learned_linker.inspect.getsource", return_value="fixed_encoder"), \
                 patch("src.phase3_resources.check_storage"):
                with self.assertRaisesRegex(RuntimeError, "outage"):
                    cached_training_vectors(destination, "mentions", ["a", "b"], batch_size=1)
            with patch("src.learned_linker.encode_batch", return_value=vector) as encoder, \
                 patch("src.learned_linker.inspect.getsource", return_value="fixed_encoder"), \
                 patch("src.phase3_resources.check_storage"):
                result = cached_training_vectors(destination, "mentions", ["a", "b"], batch_size=1)
                encoder.assert_called_once_with(["b"])
                self.assertEqual(result.shape, (2, 768))
                with self.assertRaisesRegex(ValueError, "inputs or encoder changed"):
                    cached_training_vectors(destination, "mentions", ["changed", "b"], batch_size=1)

    def test_encoding_resume_rejects_changed_vector_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            vector = np.zeros((1, 768), dtype=np.float32)
            vector[0, 0] = 1
            with patch("src.learned_linker.encode_batch", return_value=vector), \
                 patch("src.learned_linker.inspect.getsource", return_value="fixed_encoder"), \
                 patch("src.phase3_resources.check_storage"):
                cached_training_vectors(destination, "mentions", ["a"])
                np.save(destination / "encoding_cache/mentions/000000.npy", vector * 2)
                with self.assertRaisesRegex(ValueError, "vectors changed"):
                    cached_training_vectors(destination, "mentions", ["a"])

    def test_diagnostics_do_not_score_final_test_predictions(self):
        import json
        labels = {c: [] for c in CURRENT_FILTER_CATEGORIES}
        labels.update(countries=["Lebanon"], relationships=[])
        prediction = {"article_id": "validation", "raw_outputs": [json.dumps(labels)],
                      "error": None, "chunk_metadata": []}
        references = [{"article_id": "validation", "language": "en", "labels": labels},
                      {"article_id": "test", "language": "ar", "labels": None}]
        # Invalid final output would raise if the diagnostic tried to parse it.
        predictions = [prediction, {"article_id": "test", "raw_outputs": [None]}]
        with tempfile.TemporaryDirectory() as folder, \
             patch("scripts.phase3_analysis.PHASE", Path(folder)), \
             patch("scripts.phase3_analysis.extraction_inputs", return_value=(
                 references, {"validation": "validation", "test": "test"}, predictions, EXTRACTION_SCHEMA_V3, {})):
            report = validation_diagnostics("run")
            self.assertEqual(len(report["examples"]), 1)
            self.assertEqual(report["examples"][0]["entity_differences"]["countries"]["correct"], ["lebanon"])


if __name__ == "__main__":
    unittest.main()
