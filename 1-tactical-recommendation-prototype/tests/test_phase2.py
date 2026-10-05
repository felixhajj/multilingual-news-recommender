import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.news_evaluation import (ensure_review_integrity_manifests, ranking_review_identity,
                                 extraction_metrics, validate_ranking_review, validate_review)
from src.news_review import empty_labels, save_extraction_review
from src.extraction_data import (CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V2,
                                 EXTRACTION_SCHEMA_V3)
from src.llm_extractor import validate_extraction_response
from src.news_training import (build_labeling_queue, quarantine_invalid_additions,
                               validate_grounded_labels)
from scripts.prepare_review_candidates import suggestion, surface_present
from scripts.review_news import highlighted_article, parse_entity_list
from src.portfolio_config import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


class Phase2DataTests(unittest.TestCase):
    def article(self, article_id, language, group):
        body = "A company tested a missile system in public. " * 8
        return {"article_id": article_id, "title": "Defense company news", "body": body,
                "language": language, "content_hash": digest("A company tested a missile system in public."),
                "group_id": group, "split": "train", "mentions": [{"text": "company"}], "domain_hits": 3,
                "source_url": "https://example.test/" + article_id, "source_revision": "1",
                "license_id": "CC", "license_url": "https://example.test/license"}

    def test_balanced_queue_keeps_unequal_language_tail_and_excludes_frozen_groups(self):
        records = [self.article("en-1", "en", "g1"), self.article("en-2", "en", "g2"),
                   self.article("en-3", "en", "g3"), self.article("ar-1", "ar", "g4"),
                   self.article("ar-frozen", "ar", "heldout")]
        queue = build_labeling_queue(records, [], {"heldout"}, limit=10)
        self.assertEqual(len(queue), 4)
        self.assertEqual([row["language"] for row in queue[:2]], ["en", "ar"])
        self.assertEqual(sum(row["language"] == "en" for row in queue), 3)
        self.assertNotIn("ar-frozen", {row["article_id"] for row in queue})

    def test_relationship_endpoints_must_be_literal_before_acceptance(self):
        labels = {key: [] for key in
                  ("countries", "companies", "organizations", "profiles", "systems", "topics")}
        labels["companies"] = ["Acme"]
        labels["relationships"] = [
            {"subject": "Acme", "relation": "built", "object": "invented system"}
        ]
        with self.assertRaisesRegex(ValueError, "nonliteral relationship endpoint"):
            validate_grounded_labels(labels, "Acme published a statement.")

    def test_saved_invalid_addition_is_quarantined_and_attempt_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels = {key: [] for key in
                      ("countries", "companies", "organizations", "profiles", "systems", "topics")}
            labels["companies"] = ["Acme"]
            labels["relationships"] = [
                {"subject": "Acme", "relation": "built", "object": "invented system"}
            ]
            additions = [{"article_id": "a", "text": "Acme published a statement.", "labels": labels}]
            attempts = [{"article_id": "a", "outcome": "accepted", "attempt_id": "one"}]

            valid, corrected = quarantine_invalid_additions(additions, attempts, directory)

            self.assertEqual(valid, [])
            self.assertEqual(corrected[0]["outcome"], "rejected")
            self.assertEqual(corrected[0]["original_outcome"], "accepted")
            quarantined = json.loads((directory / "quarantined.jsonl").read_text().splitlines()[0])
            self.assertEqual(quarantined["article_id"], "a")
            self.assertIn("nonliteral relationship endpoint", quarantined["quarantine_reason"])

    def test_annotation_aid_does_not_call_roles_or_events_people_or_countries(self):
        role = {"instance_of": [], "descriptions": {"en": "head of state of Republic of Poland"}}
        event = {"instance_of": [], "descriptions": {"en": "massacre of military officers in 1940"}}
        person = {"instance_of": ["Q5"], "descriptions": {"en": "Polish politician"}}
        self.assertEqual(suggestion(role)[0], "ignore")
        self.assertEqual(suggestion(event)[0], "ignore")
        self.assertEqual(suggestion(person)[0], "profiles")
        self.assertEqual(suggestion({"instance_of": [], "descriptions": {"en": "central government"}},
                                    "government")[0], "ignore")

    def test_annotation_aid_separates_locations_from_countries(self):
        country = {"instance_of": ["Q6256"], "descriptions": {"en": "sovereign state"}}
        location = {"instance_of": ["Q515"], "descriptions": {"en": "city in Palestine"}}
        territory = {"instance_of": [], "descriptions": {"en": "Palestinian territory"}}
        adjective = {"instance_of": ["Q82794"], "labels": {"en": "Africa"},
                     "descriptions": {"en": "continent"}}
        self.assertEqual(suggestion(country, "Lebanon")[0], "countries")
        self.assertEqual(suggestion(location, "Gaza City")[0], "locations")
        self.assertEqual(suggestion(territory, "Gaza Strip")[0], "locations")
        self.assertEqual(suggestion(adjective, "African"),
                         ("ignore", "nationality_or_regional_adjective"))
        self.assertEqual(suggestion(territory, "الفلسطينيين"),
                         ("ignore", "nationality_or_regional_adjective"))
        self.assertEqual(suggestion(location, "العاصمة"),
                         ("ignore", "generic_role_or_common_word"))

    def test_global_surface_matching_respects_word_boundaries(self):
        self.assertTrue(surface_present("Iran", "Talks with Iran continued."))
        self.assertFalse(surface_present("Iran", "The Iranian delegation arrived."))

    def test_simple_review_lists_accept_pipes_or_lines_and_remove_duplicates(self):
        self.assertEqual(parse_entity_list("Iran | Lebanon\nIran"), ["Iran", "Lebanon"])

    def test_article_highlighter_has_categories_and_recovery_controls(self):
        labels = {key: [] for key in CURRENT_FILTER_CATEGORIES if key != "topics"}
        labels["countries"] = ["Lebanon"]
        markup = highlighted_article(
            {"article_id": "a", "title": "Lebanon update", "text": "News from Lebanon."}, labels)
        for category in labels:
            self.assertIn(f'data-category="{category}"', markup)
        for action in ("remove", "undo", "reset"):
            self.assertIn(f'data-action="{action}"', markup)
        self.assertIn("entity-country", markup)

    def test_article_highlighter_does_not_match_inside_larger_words(self):
        labels = {key: [] for key in CURRENT_FILTER_CATEGORIES if key != "topics"}
        labels["countries"] = ["Russia"]
        markup = highlighted_article(
            {"article_id": "a", "title": "Russia and Russian officials", "text": "Russia spoke."}, labels)
        self.assertEqual(markup.count(">Russia</mark>"), 2)
        self.assertNotIn(">Russia</mark>n", markup)

    def test_v2_cannot_masquerade_as_location_aware_v3(self):
        legacy = {key: [] for key in
                  ("countries", "companies", "organizations", "profiles", "systems", "topics")}
        legacy["relationships"] = []
        validate_extraction_response(legacy, EXTRACTION_SCHEMA_V2)
        with self.assertRaisesRegex(ValueError, "locations"):
            validate_extraction_response(legacy, EXTRACTION_SCHEMA_V3)


class Phase2ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.artifacts = Path(self.temp.name)
        self.extraction = [{"article_id": "a", "source_url": "https://example.test/a", "title": "Iran news",
                            "text": "Iran met diplomats in Lebanon.", "language": "en",
                            "content_hash": digest("Iran met diplomats in Lebanon."), "group_id": "g",
                            "review_status": "pending", "reviewed_by": None, "labels": None,
                            "instructions": "Extract literal entities."}]
        self.ranking = [{"query_id": "q", "interest": "Iran diplomacy",
                         "article": {"article_id": "a", "title": "Iran news", "body": "Iran met diplomats."},
                         "required_filters": {"countries": ["Iran"]}, "role": "validation", "relevance": None,
                         "review_status": "pending", "reviewed_by": None,
                         "instructions": "Grade 0, 1 or 2."}]
        review = self.artifacts / "review"
        write_jsonl(review / "extraction.jsonl", self.extraction)
        write_json(review / "extraction_roles.json", {"a": "validation"})
        write_json(review / "frozen_extraction_manifest.json", {
            "items": {"a": digest(self.extraction[0]["text"])}, "group_ids": ["g"], "selection_version": "old"})
        write_jsonl(review / "ranking.jsonl", self.ranking)
        old_ranking = digest([{key: row[key] for key in ("query_id", "interest", "article", "required_filters", "role")}
                              for row in self.ranking])
        write_json(review / "frozen_ranking_manifest.json", {"items": old_ranking})
        self.patches = [patch("src.news_evaluation.ARTIFACTS", self.artifacts),
                        patch("src.news_review.ARTIFACTS", self.artifacts)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        ensure_review_integrity_manifests()

    def test_title_and_role_are_frozen_by_integrity_manifest(self):
        validate_review(self.extraction)
        with self.assertRaisesRegex(ValueError, "titles, bodies, metadata or roles"):
            validate_review([{**self.extraction[0], "title": "Changed"}])

    def test_extraction_review_requires_literal_labels_and_records_attribution(self):
        labels = empty_labels()
        labels["countries"] = ["Iran", "Lebanon"]
        self.assertEqual(save_extraction_review("a", labels, "reviewer-1"), 1)
        saved = json.loads((self.artifacts / "review/extraction.jsonl").read_text().splitlines()[0])
        self.assertEqual(saved["review_status"], "human_reviewed")
        self.assertTrue(saved["reviewed_at"])
        with self.assertRaisesRegex(ValueError, "already reviewed by reviewer-1"):
            save_extraction_review("a", labels, "reviewer-2")
        labels["countries"].append("Invented")
        with self.assertRaisesRegex(ValueError, "copied exactly"):
            save_extraction_review("a", labels, "reviewer-1")

    def test_location_review_must_be_literal(self):
        labels = empty_labels()
        labels["countries"] = ["Iran"]
        labels["locations"] = ["Invented place"]
        with self.assertRaisesRegex(ValueError, "copied exactly"):
            save_extraction_review("a", labels, "reviewer-1")

    def test_boolean_is_not_a_valid_ranking_grade(self):
        row = {**self.ranking[0], "relevance": True, "review_status": "human_reviewed",
               "reviewed_by": "reviewer-1", "reviewed_at": "2026-01-01T00:00:00Z"}
        self.assertEqual(ranking_review_identity([row]), ranking_review_identity(self.ranking))
        with self.assertRaisesRegex(ValueError, "integer 0, 1 or 2"):
            validate_ranking_review([row])

    def test_primary_f1_scores_named_entities_not_subjective_topics(self):
        expected = empty_labels()
        expected["countries"] = ["Iran"]
        expected["topics"] = ["diplomacy"]
        predicted = empty_labels()
        predicted["countries"] = ["Iran"]
        predicted["topics"] = ["different topic"]
        metrics = extraction_metrics(
            [{"article_id": "a", "raw_output": json.dumps(predicted)}],
            [{"article_id": "a", "labels": expected, "review_status": "human_reviewed"}],
        )
        self.assertEqual(metrics["micro_f1"], 1)
        self.assertNotIn("topics", metrics["scored_categories"])


if __name__ == "__main__":
    unittest.main()
