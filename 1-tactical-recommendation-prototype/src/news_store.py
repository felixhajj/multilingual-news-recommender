"""Content/version addressed inference and compact vectors in SQLite."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class NewsStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS analyses (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(articles)")}
            if not columns:
                db.execute("""CREATE TABLE articles (
                    pipeline_identity TEXT NOT NULL, article_id TEXT NOT NULL, cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL, embedding_version TEXT NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY (pipeline_identity, article_id))""")
            elif "pipeline_identity" not in columns:
                self._migrate_article_index(db)
            db.execute("CREATE INDEX IF NOT EXISTS articles_lookup_v2 ON articles(embedding_version, pipeline_identity, article_id)")

    @staticmethod
    def _migrate_article_index(db):
        """Move the single-version index forward transactionally and keep its original table."""
        db.execute("""CREATE TABLE articles_v2 (
            pipeline_identity TEXT NOT NULL, article_id TEXT NOT NULL, cache_key TEXT NOT NULL,
            payload TEXT NOT NULL, embedding_version TEXT NOT NULL, vector BLOB NOT NULL,
            PRIMARY KEY (pipeline_identity, article_id))""")
        rows = db.execute("SELECT article_id, cache_key, payload, embedding_version, vector FROM articles").fetchall()
        migrated = []
        for article_id, cache_key, payload, embedding_version, vector in rows:
            analysis = json.loads(payload)
            pipeline_identity = analysis.get("pipeline_identity") or "legacy-unversioned:" + cache_key
            migrated.append((pipeline_identity, article_id, cache_key, payload, embedding_version, vector))
        db.executemany("INSERT INTO articles_v2 VALUES (?,?,?,?,?,?)", migrated)
        db.execute("ALTER TABLE articles RENAME TO articles_legacy_v1")
        db.execute("ALTER TABLE articles_v2 RENAME TO articles")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def cached(self, key):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM analyses WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, analysis, embedding_version, vector):
        if analysis["status"] != "ready":
            raise ValueError("Only successful model inference can enter the release collection")
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1) > 0.001:
            raise ValueError("Expected one normalized finite embedding")
        encoded = json.dumps(analysis, ensure_ascii=False)
        pipeline_identity = analysis.get("pipeline_identity")
        if not isinstance(pipeline_identity, str) or not pipeline_identity:
            raise ValueError("Indexed analysis must identify its complete pipeline version")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO analyses VALUES (?,?)", (analysis["cache_key"], encoded))
            db.execute("""INSERT OR REPLACE INTO articles
                (pipeline_identity, article_id, cache_key, payload, embedding_version, vector)
                VALUES (?,?,?,?,?,?)""", (pipeline_identity, analysis["article"]["article_id"],
                analysis["cache_key"], encoded, embedding_version, vector.tobytes()))

    def collection(self, embedding_version, pipeline_identity=None):
        with self.connect() as db:
            if pipeline_identity is None:
                rows = db.execute("SELECT payload, vector FROM articles WHERE embedding_version=? ORDER BY pipeline_identity, article_id", (embedding_version,)).fetchall()
            else:
                rows = db.execute("SELECT payload, vector FROM articles WHERE embedding_version=? AND pipeline_identity=? ORDER BY article_id",
                                  (embedding_version, pipeline_identity)).fetchall()
        records, vectors = [], []
        for payload, blob in rows:
            record = json.loads(payload)
            records.append(record)
            vectors.append(np.frombuffer(blob, dtype=np.float32).copy())
        return records, np.stack(vectors) if vectors else np.empty((0, 768), dtype=np.float32)
