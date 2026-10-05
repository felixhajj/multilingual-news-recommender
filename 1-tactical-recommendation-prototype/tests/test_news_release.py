import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.news_evaluation import extraction_metrics, ranking_review_identity
from src.news_corpus import assign_groups, clean_wikitext
from src.portfolio_config import ROOT


class ReleaseTests(unittest.TestCase):
    def test_every_notebook_cell_is_explained_and_has_stable_stage(self):
        for filename in ("tactical_report_a_to_z_walkthrough.ipynb", "tactical_report_entity_extraction_qlora.ipynb", "tactical_report_recommendation_demo.ipynb"):
            value = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
            ids = []
            for i, cell in enumerate(value["cells"]):
                if cell["cell_type"] == "code":
                    self.assertEqual(value["cells"][i-1]["cell_type"], "markdown")
                    ids.append(cell["metadata"]["stage_id"])
                    compile("".join(cell["source"]), filename, "exec")
            self.assertEqual(len(ids), len(set(ids)))

    def test_stage_map_matches_notebook_source(self):
        stages = json.loads((ROOT / "data/release/notebook_stages.json").read_text(encoding="utf-8"))
        value = json.loads((ROOT / "notebooks/tactical_report_a_to_z_walkthrough.ipynb").read_text(encoding="utf-8"))
        for cell in value["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(stages[cell["metadata"]["stage_id"]]["source"], "".join(cell["source"]))

    def test_related_versions_cannot_cross_split(self):
        rows = [{"article_id": name, "body": "A government held elections and met diplomats in its capital.",
                 "date": "2018-01-01", "mentions": [{"qid": qid} for qid in ("Q1", "Q2", "Q3")]}
                for name in ("en-1", "ar-1")]
        assign_groups(rows)
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[0]["split"], rows[1]["split"])

    def test_wikitext_is_cleaned_without_erasing_entity_surface(self):
        text = clean_wikitext("{{date|January 1, 2018}}\n[[Lebanon]] hosted talks.\n[[Category:Politics]]")
        self.assertIn("Lebanon hosted talks", text)
        self.assertNotIn("Category:", text)

    def test_rank_review_hash_excludes_only_judgments(self):
        row = {"query_id": "a", "interest": "diplomacy", "article": {"body": "news"}, "required_filters": {}, "role": "test", "relevance": None}
        self.assertEqual(ranking_review_identity([row]), ranking_review_identity([{**row, "relevance": 2}]))
        self.assertNotEqual(ranking_review_identity([row]), ranking_review_identity([{**row, "interest": "different"}]))

    def test_malformed_chunk_is_not_hidden_by_a_merged_json(self):
        labels = {key: [] for key in ("countries", "locations", "companies", "organizations", "profiles", "systems", "topics", "relationships")}
        references = [{"article_id": "a", "labels": labels, "review_status": "human_reviewed"}]
        result = extraction_metrics([{"article_id": "a", "raw_outputs": [json.dumps(labels), "{invalid"]}], references)
        self.assertEqual(result["json_parse_validity"], 0)

    def test_final_release_refuses_missing_evidence(self):
        from scripts.package_news_release import package
        with patch("scripts.package_news_release.release_gates", return_value={"review": False}):
            with self.assertRaisesRegex(ValueError, "gates remain open"):
                package()

    def test_bundle_path_traversal_rejected(self):
        import hashlib
        import zipfile
        from scripts.restore_news_bundle import restore
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "x")
                bundle.writestr("bundle_manifest.json", json.dumps({"files": {"../escape.txt": hashlib.sha256(b"x").hexdigest()}}))
            with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                restore(archive, Path(folder) / "safe")

    def test_runtime_dependencies_exclude_checkpoints_and_include_freezes(self):
        from scripts.package_news_release import runtime_artifact_paths
        from src.portfolio_config import ARTIFACTS
        if not (ARTIFACTS / "phase3/selection_lock.json").exists():
            self.skipTest("Research artifacts not installed")
        paths = runtime_artifact_paths()
        self.assertIn("review/frozen_extraction_manifest.json", paths)
        self.assertIn("review/frozen_ranking_manifest.json", paths)
        self.assertIn("review/ranking.jsonl", paths)
        self.assertFalse(any("checkpoints" in path or "history" in path for path in paths))

    def test_restore_rejects_unlisted_file(self):
        import zipfile
        from scripts.restore_news_bundle import restore
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "extra.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("bundle_manifest.json", json.dumps({"files": {}}))
                bundle.writestr("unexpected.txt", "x")
            with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
                restore(archive, Path(folder) / "safe")


if __name__ == "__main__":
    unittest.main()
