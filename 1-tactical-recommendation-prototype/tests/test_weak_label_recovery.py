import json
import unittest

from scripts.run_phase3_job import accept_weak_label
from src.extraction_data import CURRENT_FILTER_CATEGORIES
from src.portfolio_config import digest
from src.weak_label_recovery import recover_case_only


class WeakRecoveryTests(unittest.TestCase):
    def fixtures(self):
        labels = {c: [] for c in CURRENT_FILTER_CATEGORIES}
        labels.update(countries=["Lebanon"], locations=["Beirut"], relationships=[])
        labels["Countries"] = labels.pop("countries")
        candidate = {"text": "Officials in Lebanon met in Beirut."}
        attempt = {"outcome": "rejected", "error": "schema keys", "source_hash": digest(candidate["text"]),
                   "raw_outputs": [json.dumps(labels)]}
        return candidate, attempt

    def test_recovers_casing_without_claiming_valid_raw_output(self):
        candidate, attempt = self.fixtures()
        original = json.dumps(attempt)
        labels, provenance = recover_case_only(candidate, attempt, 1, accept_weak_label)
        self.assertEqual(labels["countries"], ["Lebanon"])
        self.assertFalse(provenance["raw_schema_valid"])
        self.assertEqual(original, json.dumps(attempt))

    def test_missing_key_is_not_padded(self):
        candidate, attempt = self.fixtures()
        value = json.loads(attempt["raw_outputs"][0])
        del value["locations"]
        attempt["raw_outputs"] = [json.dumps(value)]
        with self.assertRaises(ValueError):
            recover_case_only(candidate, attempt, 1, accept_weak_label)

    def test_colliding_keys_refused(self):
        candidate, attempt = self.fixtures()
        value = json.loads(attempt["raw_outputs"][0])
        value["countries"] = []
        attempt["raw_outputs"] = [json.dumps(value)]
        with self.assertRaises(ValueError):
            recover_case_only(candidate, attempt, 1, accept_weak_label)

    def test_nonliteral_entities_still_fail(self):
        candidate, attempt = self.fixtures()
        value = json.loads(attempt["raw_outputs"][0])
        value["locations"] = ["Paris"]
        attempt["raw_outputs"] = [json.dumps(value)]
        with self.assertRaisesRegex(ValueError, "nonliteral"):
            recover_case_only(candidate, attempt, 1, accept_weak_label)

    def test_incomplete_chunks_are_not_full_article_labels(self):
        candidate, attempt = self.fixtures()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            recover_case_only(candidate, attempt, 2, accept_weak_label)

    def test_modified_source_refused(self):
        candidate, attempt = self.fixtures()
        candidate["text"] += " Edited."
        with self.assertRaisesRegex(ValueError, "Source"):
            recover_case_only(candidate, attempt, 1, accept_weak_label)


if __name__ == "__main__":
    unittest.main()
