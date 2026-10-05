import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from scripts.run_phase3_job import completed_training, isolated_stage
from src.portfolio_config import file_digest, write_json


class IsolationTests(unittest.TestCase):
    def test_completed_training_is_verified_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            run = path / "runs/candidate"
            adapter = run / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "adapter_config.json").write_text("{}")
            write_json(run / "manifest.json", {"status": "completed", "optimizer_steps": 48,
                "changed_tensors": 288, "adapter_sha256": file_digest(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": file_digest(adapter / "adapter_config.json")})
            with patch("scripts.run_phase3_job.ARTIFACTS", path):
                self.assertTrue(completed_training("candidate"))
                (adapter / "adapter_model.safetensors").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    completed_training("candidate")

    def test_child_failure_is_recorded_and_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            child = Mock(pid=123)
            child.wait.return_value = 1
            write_json(Path(folder) / "job.json", {"deadline": "2100-01-01T00:00:00+00:00"})
            with patch("scripts.run_phase3_job.FOLDER", Path(folder)), \
                 patch("scripts.run_phase3_job.subprocess.Popen", return_value=child):
                with self.assertRaisesRegex(RuntimeError, "exited 1"):
                    isolated_stage("train:domain-extraction-v3")
            self.assertTrue((Path(folder) / "stage_history.jsonl").exists())

    def test_expired_stage_does_not_start_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder) / "job.json", {"deadline": "2000-01-01T00:00:00+00:00"})
            with patch("scripts.run_phase3_job.FOLDER", Path(folder)), \
                 patch("scripts.run_phase3_job.subprocess.Popen") as launch:
                with self.assertRaises(TimeoutError):
                    isolated_stage("train:domain-extraction-v3")
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
