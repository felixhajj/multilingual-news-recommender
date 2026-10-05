import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.learned_linker import LearnedEntityLinker
from src.news_store import NewsStore


class Phase4IndexTests(unittest.TestCase):
    def test_legacy_index_migrates_and_keeps_old_and_new_article_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "release.sqlite"
            vector = np.zeros(768, dtype=np.float32)
            vector[0] = 1
            old = {"status": "ready", "pipeline_identity": "legacy-pipeline",
                   "cache_key": "legacy-key", "article": {"article_id": "same", "title": "Old"}}
            db = sqlite3.connect(path)
            try:
                db.execute("CREATE TABLE articles (article_id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, "
                           "payload TEXT NOT NULL, embedding_version TEXT NOT NULL, vector BLOB NOT NULL)")
                db.execute("INSERT INTO articles VALUES (?,?,?,?,?)",
                           ("same", "legacy-key", json.dumps(old), "e5-v1", vector.tobytes()))
                db.commit()
            finally:
                db.close()

            store = NewsStore(path)
            old_records, _ = store.collection("e5-v1", "legacy-pipeline")
            self.assertEqual([row["article"]["title"] for row in old_records], ["Old"])
            db = sqlite3.connect(path)
            try:
                self.assertEqual(db.execute("SELECT count(*) FROM articles_legacy_v1").fetchone()[0], 1)
            finally:
                db.close()

            fresh = {"status": "ready", "pipeline_identity": "selected-pipeline",
                     "cache_key": "selected-key", "article": {"article_id": "same", "title": "New"}}
            store.save(fresh, "e5-v1", vector)
            new_records, _ = store.collection("e5-v1", "selected-pipeline")
            all_records, _ = store.collection("e5-v1")
            self.assertEqual([row["article"]["title"] for row in new_records], ["New"])
            self.assertEqual({row["article"]["title"] for row in all_records}, {"Old", "New"})

    def test_batched_linking_matches_scalar_results_and_batches_encoder_calls(self):
        class Encoder:
            def __init__(self):
                self.calls = 0

            def __call__(self, texts):
                self.calls += 1
                result = []
                for text in texts:
                    row = np.zeros(3, dtype=np.float32)
                    row[0 if "iran" in text.casefold() else 1] = 1
                    result.append(row)
                return np.stack(result)

        linker = object.__new__(LearnedEntityLinker)
        linker.entities = [
            {"qid": "Q1", "name": "Iran", "normalized_aliases": ["iran"]},
            {"qid": "Q2", "name": "Lebanon", "normalized_aliases": ["lebanon"]},
        ]
        linker.vectors = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        linker.config = {"mean": [0, 0, 0, 0], "scale": [1, 1, 1, 1],
                         "coef": [0, 0, 8, 0], "intercept": -4, "threshold": 0.5}
        linker.version = "test-linker"
        encoder = Encoder()
        linker.encoder = encoder
        mentions = [("Iran", "Officials in Iran discussed the matter."),
                    ("Lebanon", "Officials in Lebanon discussed the matter.")]

        scalar = [linker.link(*pair) for pair in mentions]
        scalar_calls = encoder.calls
        encoder.calls = 0
        batched = linker.link_many(mentions)

        self.assertEqual(batched, scalar)
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(scalar_calls, len(mentions))


if __name__ == "__main__":
    unittest.main()
