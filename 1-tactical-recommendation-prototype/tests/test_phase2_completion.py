import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.finish_phase2 import verify_completion
from src.portfolio_config import digest, file_digest, write_json


class CompletionSealTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "review/ranking.jsonl"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("saved human judgment\n", encoding="utf-8")
        self.snapshot = self.root / "phase2/snapshots/example"
        sealed = self.snapshot / "review/ranking.jsonl"
        sealed.parent.mkdir(parents=True)
        sealed.write_bytes(self.source.read_bytes())
        self.manifest = {"files": {"review/ranking.jsonl": file_digest(self.source)}}
        write_json(self.snapshot / "manifest.json", self.manifest)
        write_json(self.root / "phase2/completion.json", {
            "status": "complete", "snapshot_directory": "phase2/snapshots/example",
            "snapshot_manifest_hash": digest(self.manifest),
        })
        self.override = patch("scripts.finish_phase2.ARTIFACTS", self.root)
        self.override.start()
        self.addCleanup(self.override.stop)

    def test_unchanged_reference_seal_verifies(self):
        self.assertEqual(verify_completion()["status"], "complete")

    def test_changed_live_judgment_invalidates_completion(self):
        self.source.write_text("changed judgment\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since handoff"):
            verify_completion()

    def test_changed_sealed_reference_is_rejected(self):
        (self.snapshot / "review/ranking.jsonl").write_text("changed snapshot\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Sealed Phase 2 artifact changed"):
            verify_completion()

    def test_changed_manifest_is_rejected(self):
        write_json(self.snapshot / "manifest.json", {"files": {}})
        with self.assertRaisesRegex(ValueError, "snapshot manifest changed"):
            verify_completion()

    def test_review_saves_are_closed_after_completion(self):
        from src.news_review import require_open_review
        with patch("src.news_review.ARTIFACTS", self.root):
            with self.assertRaisesRegex(ValueError, "sealed for evaluation"):
                require_open_review()
