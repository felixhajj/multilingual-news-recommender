import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.manage_news_model import promote
from scripts.validate_phase3_linker import metrics
from src.learned_linker import LearnedEntityLinker, linker_artifact_identity
from src.portfolio_config import EMBEDDING_MODEL, digest, write_json
from src.phase3_resources import check_storage
from types import SimpleNamespace


class Phase3IdentityTests(unittest.TestCase):
    def test_low_disk_space_stops_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            with patch("src.phase3_resources.shutil.disk_usage", return_value=SimpleNamespace(free=1024)):
                with self.assertRaisesRegex(RuntimeError, "2 GiB"):
                    check_storage(path)
            self.assertTrue((path / "entity_vectors.npy").exists())

    def fixture(self, directory):
        entities = [{"qid": "Q1", "name": "Name", "normalized_aliases": ["name"]}]
        write_json(directory / "entities.json", entities)
        write_json(directory / "model.json", {"encoder": EMBEDDING_MODEL, "catalogue_hash": digest(entities)})
        vectors = np.zeros((1, 768), dtype=np.float32)
        vectors[0, 0] = 1
        np.save(directory / "entity_vectors.npy", vectors)
        return vectors

    def test_vector_bytes_change_cache_identity_without_upgrading_legacy_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            vector = self.fixture(path)
            first = LearnedEntityLinker(path)
            vector[0, 0], vector[0, 1] = 0, 1
            np.save(path / "entity_vectors.npy", vector)
            second = LearnedEntityLinker(path)
            self.assertNotEqual(first.version, second.version)
            self.assertEqual(first.legacy_version, second.legacy_version)
            self.assertFalse(first.artifact_identity["training_identity_complete"])

    def test_bound_vector_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            from src.portfolio_config import read_json
            config = read_json(path / "model.json")
            config["entity_vectors_sha256"] = "wrong"
            write_json(path / "model.json", config)
            with self.assertRaisesRegex(ValueError, "vectors"):
                linker_artifact_identity(path)

    def test_old_report_cannot_promote_without_selection_lock(self):
        with patch("scripts.manage_news_model.verify_selection_lock", side_effect=ValueError("selection lock missing")):
            with self.assertRaisesRegex(ValueError, "selection lock missing"):
                promote("candidate")

    def test_unselected_run_cannot_be_promoted(self):
        with patch("scripts.manage_news_model.verify_selection_lock", return_value={"selected_run": None}), \
             patch("scripts.manage_news_model.write_json") as write:
            with self.assertRaisesRegex(ValueError, "selected candidate"):
                promote("candidate")
            write.assert_not_called()

    def test_unseen_alias_is_not_confused_with_an_unknown_entity(self):
        entities = [{"qid": "Q1", "normalized_aliases": ["known"]}]
        rows = [{"text": "new alias", "qid": "Q1", "score": .8, "predicted_qid": "Q1",
                 "accepted_qid": "Q1", "gold_in_catalogue": True, "gold_retrieved": True, "alias_prediction": None},
                {"text": "unknown", "qid": "Q2", "score": .1, "predicted_qid": "Q1",
                 "accepted_qid": None, "gold_in_catalogue": False, "gold_retrieved": False, "alias_prediction": None}]
        report = metrics(rows, entities, .5)
        self.assertEqual(report["unseen_known_alias_mentions"], 1)
        self.assertEqual(report["unknown_abstention"], 1)
        self.assertEqual(report["accepted_precision"], 1)
        self.assertEqual(report["correct_link_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
