import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_phase3_job import accept_weak_label
from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3
from src.news_evaluation import extraction_metrics
from src.news_selection import passes_gate, score_extraction
from src.news_training_v3 import (assert_resume_identity, check_checkpoint, sample_indices,
                                  save_checkpoint, training_identity)


class Phase3Tests(unittest.TestCase):
    def example(self):
        labels = {c: [] for c in CURRENT_FILTER_CATEGORIES}
        labels.update(countries=["Lebanon"], locations=["Beirut"], relationships=[])
        return {"article_id": "training-one", "split": "train", "schema_version": EXTRACTION_SCHEMA_V3,
                "text": "Officials in Lebanon met in Beirut.", "labels": labels}

    def test_grounded_weak_labels_reject_invented_locations(self):
        row = self.example()
        accept_weak_label(row["labels"], row["text"])
        row["labels"]["locations"] = ["Paris"]
        with self.assertRaisesRegex(ValueError, "nonliteral"):
            accept_weak_label(row["labels"], row["text"])

    def test_resume_refuses_changed_dataset_and_steps(self):
        row = self.example()
        identity = training_identity([row], 48)
        assert_resume_identity({"identity": identity}, identity)
        with self.assertRaises(ValueError):
            assert_resume_identity({"identity": identity}, training_identity([row], 49))
        row["text"] += " Later."
        with self.assertRaises(ValueError):
            assert_resume_identity({"identity": identity}, training_identity([row], 48))

    def test_train_refuses_test_rows(self):
        row = self.example()
        row["split"] = "test"
        with self.assertRaises(ValueError):
            training_identity([row], 48)

    def test_cursor_replay_is_identical(self):
        uninterrupted = [sample_indices(23, step) for step in range(20)]
        resumed = [sample_indices(23, step) for step in range(8)] + [sample_indices(23, step) for step in range(8, 20)]
        self.assertEqual(uninterrupted, resumed)
        self.assertEqual(sorted(i for step in range(6) for i in sample_indices(23, step))[:23].count(0), 1)

    def test_final_test_requires_lock(self):
        with patch("src.news_selection.extraction_inputs", return_value=([], {}, [], EXTRACTION_SCHEMA_V3, {})), \
             patch("src.news_selection.verify_selection_lock", side_effect=ValueError("missing lock")):
            with self.assertRaisesRegex(ValueError, "missing lock"):
                score_extraction("some-run", "test")

    def test_gate_does_not_trade_arabic_regression_for_total_f1(self):
        base = {"schema_version": EXTRACTION_SCHEMA_V3,
                "metrics": {"schema_validity": .8, "micro_f1": .2,
                            "by_language": {"en": {"micro_f1": .3}, "ar": {"micro_f1": .1}}}}
        candidate = {"schema_version": EXTRACTION_SCHEMA_V3,
                     "metrics": {"schema_validity": 1, "micro_f1": .4,
                                 "by_language": {"en": {"micro_f1": .5}, "ar": {"micro_f1": .09}}}}
        self.assertFalse(passes_gate(candidate, base))
        candidate["metrics"]["by_language"]["ar"]["micro_f1"] = .1
        self.assertTrue(passes_gate(candidate, base))

    def test_json_parsing_is_not_schema_validity(self):
        row = self.example()
        reference = {**row, "review_status": "human_reviewed"}
        result = extraction_metrics([{"article_id": row["article_id"], "raw_outputs": ['{"countries":[]}']}], [reference])
        self.assertEqual(result["json_parse_validity"], 1)
        self.assertEqual(result["schema_validity"], 0)
        self.assertEqual(result["micro_f1"], 0)

    def test_cpu_checkpoint_restores_optimizer_rng_and_update(self):
        import torch
        class TinyAdapter(torch.nn.Linear):
            def save_pretrained(self, path):
                path.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), path / "adapter_model.safetensors")
                (path / "adapter_config.json").write_text("{}", encoding="ascii")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            torch.manual_seed(42)
            model = TinyAdapter(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            def update():
                optimizer.zero_grad()
                model(torch.rand(1, 2)).sum().backward()
                optimizer.step()
            update()
            pointer = save_checkpoint(directory, model, {"step": 1, "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()})
            update()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            checkpoint = check_checkpoint(directory, pointer)
            state = torch.load(checkpoint / "state.pt", weights_only=False)
            model.load_state_dict(torch.load(checkpoint / "adapter/adapter_model.safetensors", weights_only=True))
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            update()
            for key in expected:
                self.assertTrue(torch.equal(expected[key], model.state_dict()[key]))
            (checkpoint / "adapter/adapter_config.json").write_text('{"changed":true}', encoding="ascii")
            with self.assertRaises(ValueError):
                check_checkpoint(directory, pointer)


if __name__ == "__main__":
    unittest.main()
