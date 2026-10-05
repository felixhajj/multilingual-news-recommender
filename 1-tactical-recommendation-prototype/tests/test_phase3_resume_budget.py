import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_phase3_job import reserve_budget
from src.portfolio_config import read_json, write_json


class Phase3ResumeBudgetTests(unittest.TestCase):
    def fixture(self, folder, consumed=63113):
        write_json(folder / "processing_budget.json", {"consumed_seconds": consumed,
                   "limit_seconds": 86400, "allocations": []})
        write_json(folder / "phase3/job.json", {"status": "needs_attention", "charged_seconds": 10800,
                   "deadline": "2026-09-30T14:07:36+00:00"})

    def test_same_extension_id_is_charged_once_and_deadline_does_not_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            with patch("scripts.run_phase3_job.ARTIFACTS", folder), patch("scripts.run_phase3_job.FOLDER", folder / "phase3"):
                first = reserve_budget(3, 2, "resume-one")
                second = reserve_budget(3, 2, "resume-one")
                self.assertEqual(first, second)
                self.assertEqual(second["charged_seconds"], 18000)
                self.assertEqual(read_json(folder / "processing_budget.json")["consumed_seconds"], 70313)

    def test_insufficient_remaining_budget_refuses_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder, consumed=86000)
            before = read_json(folder / "processing_budget.json")
            with patch("scripts.run_phase3_job.ARTIFACTS", folder), patch("scripts.run_phase3_job.FOLDER", folder / "phase3"):
                with self.assertRaisesRegex(ValueError, "Cumulative"):
                    reserve_budget(3, 2, "resume-one")
            self.assertEqual(before, read_json(folder / "processing_budget.json"))

    def test_allocation_id_cannot_change_allowance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            with patch("scripts.run_phase3_job.ARTIFACTS", folder), patch("scripts.run_phase3_job.FOLDER", folder / "phase3"):
                reserve_budget(3, 2, "resume-one")
                with self.assertRaisesRegex(ValueError, "different allowance"):
                    reserve_budget(3, 1, "resume-one")


if __name__ == "__main__":
    unittest.main()
