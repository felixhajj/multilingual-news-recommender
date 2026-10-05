import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.portfolio_config import read_json, write_json, write_jsonl
from src.release_processing import recover_interrupted_run, remaining_allowance, retryable_failure


class ReleaseBudgetTests(unittest.TestCase):
    def test_resource_failures_remain_retryable_without_retrying_schema_failures(self):
        self.assertTrue(retryable_failure({"error_type": "RuntimeError", "error": "CUDA out of memory"}))
        self.assertTrue(retryable_failure({"error_type": "TimeoutError", "error": "deadline"}))
        self.assertFalse(retryable_failure({"error_type": "ExtractionOutputError", "error": "Invalid JSON"}))

    def budget(self):
        return {"consumed_seconds": 100, "limit_seconds": 10000, "allocations": [],
                "completion_budget": {"allocation_id": "completion-one", "sections": {"index": 2000}}}

    def test_index_allocation_is_shared_across_restarts_and_preserves_other_sections(self):
        budget = self.budget()
        budget["allocations"] = [
            {"completion_allocation_id": "completion-one", "budget_section": "index", "elapsed_seconds": 700},
            {"completion_allocation_id": "completion-one", "budget_section": "application_checks", "elapsed_seconds": 400},
            {"completion_allocation_id": "historical", "budget_section": "index", "elapsed_seconds": 900}]
        self.assertEqual(remaining_allowance(budget, 3600), 1300)
        budget["consumed_seconds"] = 9900
        self.assertEqual(remaining_allowance(budget, 3600), 100)

    def test_recovery_is_charged_once_and_reduces_index_allowance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "budget.json"
            budget = self.budget()
            manifest = {"status": "preparing", "elapsed_seconds": 400,
                        "in_flight_article_id": "unfinished", "in_flight_reserve_seconds": 300,
                        "completion_allocation_id": "completion-one"}
            recover_interrupted_run(budget, manifest, path)
            recover_interrupted_run(budget, manifest, path)
            self.assertEqual(budget["consumed_seconds"], 800)
            self.assertEqual(len(budget["recovered_release_runs"]), 1)
            self.assertEqual(remaining_allowance(budget, 3600), 1300)
            self.assertEqual(read_json(path), budget)

    def test_worker_skips_existing_inference_and_charges_final_exports(self):
        from scripts import build_news_release as worker
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            clock = SimpleNamespace(value=0)
            articles = [{"article_id": str(i), "title": "News", "body": "News reports diplomacy. " * 20,
                         "language": "ar" if i == 2 else "en", "split": "validation", "domain_hits": 1}
                        for i in range(1, 4)]
            rows = [{"article": articles[0], "status": "ready"}]
            processed = []
            def collection(*args):
                return list(rows), np.zeros((len(rows), 768))
            def analyze(article, persist=False):
                processed.append(article["article_id"])
                clock.value += 10
                result = {"article": article, "status": "ready", "evidence": []}
                rows.append(result)
                return result
            def recommend(*args):
                clock.value += 7
                return {"results": []}
            pipeline = SimpleNamespace(identity="selected", embedding_version="e5", linker=SimpleNamespace(version="linker"),
                                       backend=SimpleNamespace(identity={"run_id": "extraction-only-v3"}),
                                       store=SimpleNamespace(collection=collection), analyze_article=analyze,
                                       recommend=recommend)
            write_json(folder / "processing_budget.json", self.budget())
            write_jsonl(folder / "corpus.jsonl", articles)
            with patch.object(worker, "ARTIFACTS", folder), patch.object(worker, "ROOT", folder), \
                 patch.object(worker, "NewsPipeline", return_value=pipeline), \
                 patch.object(worker, "check_storage"), patch.object(worker, "publish_evidence_snapshot"), \
                 patch.object(worker.time, "monotonic", side_effect=lambda: clock.value), \
                 patch("sys.argv", ["build_news_release", "--limit", "3", "--hours", "1"]):
                worker.main()
            self.assertEqual(set(processed), {"2", "3"})
            budget = read_json(folder / "processing_budget.json")
            self.assertEqual(budget["consumed_seconds"], 141)
            self.assertTrue(budget["allocations"][-1]["includes_final_export"])
            release = read_json(folder / "release_manifest.json")
            self.assertEqual(release["status"], "prepared")
            self.assertEqual(release["language_counts"], {"en": 2, "ar": 1})


class GenerationDeadlineTests(unittest.TestCase):
    def adapter(self):
        import torch
        from src.llm_extractor import QwenExtractionAdapter
        class Inputs(dict):
            def to(self, device):
                return self
        labels = {name: [] for name in ("countries", "companies", "organizations", "profiles",
                                       "systems", "topics", "relationships")}
        import json
        raw = json.dumps(labels)
        adapter = object.__new__(QwenExtractionAdapter)
        tokenizer = Mock(pad_token_id=0, eos_token_id=4)
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = Inputs(input_ids=torch.tensor([[1, 2]]))
        tokenizer.decode.return_value = raw
        adapter.tokenizer = tokenizer
        adapter.model = SimpleNamespace(device="cpu", generate=Mock(return_value=torch.tensor([[1, 2, 3, 4]])))
        return adapter, raw

    def test_expired_budget_does_not_start_generation(self):
        adapter, _ = self.adapter()
        with patch("time.monotonic", return_value=11), self.assertRaises(TimeoutError):
            adapter.extract("Article", deadline=10)
        adapter.model.generate.assert_not_called()

    def test_generation_is_bounded_without_changing_successful_output(self):
        adapter, raw = self.adapter()
        with patch("time.monotonic", return_value=0):
            _, output = adapter.extract("Article", deadline=10)
        self.assertEqual(output, raw)
        self.assertEqual(adapter.model.generate.call_args.kwargs["max_time"], 10)
        self.assertFalse(adapter.model.generate.call_args.kwargs["do_sample"])

    def test_deadline_output_is_retained_as_failure_instead_of_success(self):
        adapter, raw = self.adapter()
        with patch("time.monotonic", side_effect=[0, 11]), self.assertRaises(TimeoutError) as raised:
            adapter.extract("Article", deadline=10)
        self.assertEqual(raised.exception.raw_output, raw)


class EncoderChunkTests(unittest.TestCase):
    def test_actual_pinned_tokenizer_chunks_fit_model_limit(self):
        snapshot = Path("D:/hf-cache/hub/models--intfloat--multilingual-e5-base/snapshots/"
                        "d128750597153bb5987e10b1c3493a34e5a4502a")
        if not snapshot.exists():
            self.skipTest("Pinned E5 tokenizer is not cached locally; no network downloads in tests")
        from transformers import AutoTokenizer
        from src.learned_linker import encode_batch
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        chunks = []
        def encode(texts, **kwargs):
            chunks.extend(texts)
            vectors = np.zeros((len(texts), 768), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors
        model = SimpleNamespace(tokenizer=tokenizer, encode=encode)
        texts = ["passage: " + "Officials discussed regional diplomacy and elections. " * 160,
                 "query: " + "ناقش المسؤولون التعاون الإقليمي والمفاوضات السياسية. " * 160]
        with patch("src.embeddings._get_model", return_value=model):
            vectors = encode_batch(texts)
        self.assertEqual(vectors.shape, (2, 768))
        self.assertGreater(len(chunks), len(texts))
        for text in chunks:
            ids = tokenizer(text, truncation=False, verbose=False)["input_ids"]
            self.assertLessEqual(len(ids), tokenizer.model_max_length)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)


if __name__ == "__main__":
    unittest.main()
