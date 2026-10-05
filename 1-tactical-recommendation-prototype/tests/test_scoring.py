import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data_loader import (
    load_articles,
    load_corpus_source_manifest,
    load_domain_corpus,
    load_entity_catalogue,
    load_extraction_dataset,
    load_filter_vocabulary,
    load_incoming_articles,
    load_users,
)
from src.corpus_ingestion import (
    corpus_summary,
    select_approved_training_documents,
    stable_split,
    training_text,
    write_training_corpus,
)
from src.embeddings import (
    article_content_text,
    embedding_map_for_user_article,
    normalize,
    user_interest_text,
)
from src.entity_enrichment import enrich_articles
from src.entity_catalogue_pipeline import promote_reviewed_entities
from src.extraction_data import FILTER_CATEGORIES, build_extraction_messages, examples_by_split
from src.extraction_outputs import apply_extraction_outputs, load_extraction_outputs
from src.llm_extractor import (
    article_extraction_text,
    merge_extraction_with_catalogue,
    parse_first_json_object,
    validate_extraction_response,
)
from src.review_pipeline import promote_reviewed_examples
from scripts.discover_gdelt_articles import discover
from scripts.discover_wikidata_entities import bindings_to_candidates, build_query
from src.scoring import (
    find_embedding_signals,
    find_exact_matches,
    rank_articles_for_user,
    score_article_for_user,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ALLOWED_CATEGORIES = ("countries", "companies", "organizations", "profiles", "systems", "topics")


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.users = load_users(DATA_DIR)
        self.articles = load_articles(DATA_DIR)
        self.vocabulary = load_filter_vocabulary(DATA_DIR)
        self.catalogue = load_entity_catalogue(DATA_DIR)

    def allowed_values(self):
        allowed = set()
        for values in self.vocabulary.values():
            allowed.update(values)
        return allowed

    def test_user_interests_use_only_visible_filter_values(self):
        allowed = self.allowed_values()
        for user in self.users:
            self.assertEqual(set(user["interests"]), set(ALLOWED_CATEGORIES))
            for category, values in user["interests"].items():
                with self.subTest(user=user["user_id"], category=category):
                    self.assertTrue(set(values).issubset(set(self.vocabulary[category])))
                    self.assertTrue(set(values).issubset(allowed))

    def test_article_tags_use_only_visible_filter_values(self):
        allowed = self.allowed_values()
        for article in self.articles:
            self.assertEqual(set(article["tags"]), set(ALLOWED_CATEGORIES))
            for category, values in article["tags"].items():
                with self.subTest(article=article["article_id"], category=category):
                    self.assertTrue(set(values).issubset(set(self.vocabulary[category])))
                    self.assertTrue(set(values).issubset(allowed))

    def test_every_user_has_an_explicit_interest_statement(self):
        for user in self.users:
            with self.subTest(user=user["user_id"]):
                self.assertTrue(user["query"].strip())

    def test_embedding_input_keeps_semantics_separate_from_exact_filters(self):
        user_text = user_interest_text(self.users[0])
        article = self.articles[0]
        article_text = article_content_text(article)

        self.assertEqual(user_text, f"query: {self.users[0]['query']}")
        self.assertNotIn("Countries:", user_text)
        self.assertNotIn("Companies:", user_text)
        self.assertIn(article["title"], article_text)
        for field in ("summary", "body"):
            if article.get(field):
                self.assertIn(article[field], article_text)
        self.assertNotIn("Countries:", article_text)
        self.assertNotIn("Companies:", article_text)
        self.assertNotIn(article["date"], article_text)
        self.assertNotIn(article["product_type"], article_text)

    def test_unseen_future_term_remains_in_embedding_input(self):
        article = {
            "title": "AeroShield Dynamics opens regional talks",
            "summary": "The new supplier presented a counter-drone platform.",
            "tags": {category: [] for category in ALLOWED_CATEGORIES},
        }

        article_text = article_content_text(article)

        self.assertIn("AeroShield Dynamics", article_text)
        self.assertIn("counter-drone platform", article_text)

    def test_exact_matches_contribute_to_hybrid_score(self):
        user = self.users[0]
        article = next(item for item in self.articles if item["article_id"] == "article_010")

        with patch("src.scoring.embedding_similarity", return_value=0.0), patch(
            "src.scoring.find_semantic_matches", return_value=[]
        ):
            result = score_article_for_user(user, article, self.articles)

        self.assertGreater(len(result["exact_matches"]), 0)
        self.assertGreater(result["metadata_score"], 0)
        self.assertEqual(result["score"], round(result["metadata_score"] * 0.25))

    def test_article_score_equals_embedding_cosine_similarity_scaled_to_100(self):
        user = self.users[0]
        article = self.articles[0]

        with patch("src.scoring.embedding_similarity", return_value=0.876), patch(
            "src.scoring.exact_match_score", return_value=0
        ), patch(
            "src.scoring.find_semantic_matches", return_value=[]
        ):
            result = score_article_for_user(user, article, self.articles)

        self.assertEqual(result["embedding_similarity"], 0.876)
        self.assertEqual(result["embedding_score"], 88)
        self.assertEqual(result["score"], 66)

    def test_semantic_matches_exclude_exact_matches(self):
        user = self.users[0]
        article = next(item for item in self.articles if item["article_id"] == "article_010")
        exact_values = {normalize(match["value"]) for match in find_exact_matches(user, article)}

        def fake_encode(texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        with patch("src.embeddings.encode_texts", side_effect=fake_encode):
            signals = find_embedding_signals(user, article, self.articles)

        for signal in signals:
            self.assertNotIn(normalize(signal["interest"]), exact_values)
            self.assertNotIn(normalize(signal["article_tag"]), exact_values)

    def test_3d_map_returns_labeled_points(self):
        user = self.users[0]
        article = self.articles[0]

        def fake_encode(texts):
            return [[float(index), float(index + 1), float(index + 2)] for index, _ in enumerate(texts)]

        def fake_reduce(vectors):
            return [[float(index), float(index + 1), float(index + 2)] for index, _ in enumerate(vectors)]

        with patch("src.embeddings.encode_texts", side_effect=fake_encode), patch(
            "src.embeddings.reduce_vectors_to_3d", side_effect=fake_reduce
        ):
            points = embedding_map_for_user_article(user, article)

        self.assertGreater(len(points), 0)
        for point in points:
            self.assertIn(point["source"], ("user", "article"))
            self.assertIn(point["category"], ALLOWED_CATEGORIES)
            for field in ("label", "x", "y", "z", "is_exact_match"):
                self.assertIn(field, point)

    def test_ranking_sorts_by_embedding_score(self):
        user = self.users[0]

        with patch("src.scoring.embedding_similarity", side_effect=[0.2, 0.8, 0.5] + [0.0] * 20), patch(
            "src.scoring.exact_match_score", return_value=0
        ), patch(
            "src.scoring.find_semantic_matches", return_value=[]
        ):
            ranked = rank_articles_for_user(user, self.articles[:3])

        self.assertEqual(ranked[0]["score"], 60)
        self.assertEqual(ranked[-1]["score"], 15)

    def test_required_company_and_country_create_direct_match(self):
        user = next(item for item in self.users if item["user_id"] == "user_004")
        direct_article = next(item for item in self.articles if item["article_id"] == "article_002")
        related_article = next(item for item in self.articles if item["article_id"] == "article_007")

        with patch("src.scoring.embedding_similarity", return_value=0.5), patch(
            "src.scoring.find_semantic_matches", return_value=[]
        ):
            direct = score_article_for_user(user, direct_article)
            related = score_article_for_user(user, related_article)

        self.assertTrue(direct["direct_match"])
        self.assertFalse(related["direct_match"])

    def test_incoming_arabic_article_links_to_canonical_entities(self):
        incoming = load_incoming_articles(DATA_DIR)
        enriched = enrich_articles(incoming, self.catalogue)
        article = next(item for item in enriched if item["article_id"] == "incoming_001")

        self.assertIn("Lockheed Martin", article["tags"]["companies"])
        self.assertIn("Saudi Arabia", article["tags"]["countries"])
        self.assertIn("Saudi Ministry of Defense", article["tags"]["organizations"])
        self.assertIn("Air defense systems", article["tags"]["systems"])

    def test_unknown_entity_is_kept_for_review(self):
        incoming = load_incoming_articles(DATA_DIR)
        enriched = enrich_articles(incoming, self.catalogue)
        article = next(item for item in enriched if item["article_id"] == "incoming_003")

        candidates = {item["value"] for item in article["unknown_candidates"]}
        self.assertIn("AeroShield Dynamics", candidates)

    def test_known_alias_is_not_duplicated_as_unknown(self):
        incoming = load_incoming_articles(DATA_DIR)
        enriched = enrich_articles(incoming, self.catalogue)
        article = next(item for item in enriched if item["article_id"] == "incoming_002")

        candidates = {item["value"] for item in article["unknown_candidates"]}
        self.assertNotIn("The Saudi MoD", candidates)

    def test_extraction_dataset_has_fixed_train_validation_and_test_splits(self):
        examples = load_extraction_dataset(DATA_DIR)
        splits = examples_by_split(examples)

        self.assertGreaterEqual(len(splits["train"]), 20)
        self.assertEqual(len(splits["validation"]), 5)
        self.assertEqual(len(splits["test"]), 5)

    def test_extraction_training_messages_end_with_valid_structured_json(self):
        example = load_extraction_dataset(DATA_DIR)[0]
        messages = build_extraction_messages(example)
        response = json.loads(messages[-1]["content"])

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(set(response), set(FILTER_CATEGORIES) | {"relationships"})

    def test_unseen_test_companies_are_not_leaked_into_training_examples(self):
        splits = examples_by_split(load_extraction_dataset(DATA_DIR))
        train_text = " ".join(example["text"] for example in splits["train"])

        self.assertNotIn("AeroShield Dynamics", train_text)
        self.assertNotIn("شركة درع الخليج", train_text)

    def test_domain_corpus_only_selects_reviewed_authorized_documents(self):
        manifest = load_corpus_source_manifest(DATA_DIR)
        documents = load_domain_corpus(DATA_DIR)

        approved, rejected = select_approved_training_documents(documents, manifest)

        self.assertEqual(len(approved), 8)
        self.assertEqual(rejected, [])
        self.assertEqual(
            corpus_summary(approved, rejected)["splits"],
            {"train": 6, "validation": 1, "test": 1},
        )

    def test_discovery_metadata_cannot_enter_training_even_if_record_requests_it(self):
        manifest = load_corpus_source_manifest(DATA_DIR)
        document = {
            **load_domain_corpus(DATA_DIR)[0],
            "document_id": "blocked_gdelt_document",
            "source_id": "gdelt_discovery",
            "allowed_for_training": True,
        }

        approved, rejected = select_approved_training_documents([document], manifest)

        self.assertEqual(approved, [])
        self.assertIn("source rights_status is discovery_only", rejected[0]["reasons"])
        self.assertIn("source does not allow model_training", rejected[0]["reasons"])

    def test_unreviewed_document_cannot_enter_training(self):
        manifest = load_corpus_source_manifest(DATA_DIR)
        document = {
            **load_domain_corpus(DATA_DIR)[0],
            "document_id": "pending_document",
            "review_status": "pending",
        }

        approved, rejected = select_approved_training_documents([document], manifest)

        self.assertEqual(approved, [])
        self.assertIn("document review_status is pending", rejected[0]["reasons"])

    def test_domain_corpus_deduplicates_article_content(self):
        manifest = load_corpus_source_manifest(DATA_DIR)
        original = load_domain_corpus(DATA_DIR)[0]
        duplicate = {
            **original,
            "document_id": "duplicate_content_document",
            "source_url": "https://example.invalid/duplicate-content-document",
        }

        approved, rejected = select_approved_training_documents([original, duplicate], manifest)

        self.assertEqual(len(approved), 1)
        self.assertIn("duplicate article content", rejected[0]["reasons"])

    def test_discovery_copy_does_not_block_authorized_copy_of_same_text(self):
        manifest = load_corpus_source_manifest(DATA_DIR)
        authorized = load_domain_corpus(DATA_DIR)[0]
        discovered = {
            **authorized,
            "document_id": "unapproved_discovery_copy",
            "source_id": "gdelt_discovery",
            "allowed_for_training": False,
            "review_status": "pending",
        }

        approved, rejected = select_approved_training_documents([discovered, authorized], manifest)

        self.assertEqual([document["document_id"] for document in approved], [authorized["document_id"]])
        self.assertEqual(rejected[0]["document_id"], "unapproved_discovery_copy")

    def test_generated_domain_corpus_keeps_text_and_provenance(self):
        manifest = load_corpus_source_manifest(DATA_DIR)
        approved, _ = select_approved_training_documents(load_domain_corpus(DATA_DIR), manifest)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "corpus.jsonl"
            write_training_corpus(approved, output_path)
            output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertIn(approved[0]["title"], training_text(approved[0]))
        self.assertEqual(output[0]["provenance"]["license_id"], "CC0-1.0")
        self.assertEqual(output[0]["provenance"]["source_url"], approved[0]["source_url"])

    def test_stable_corpus_split_does_not_change_between_calls(self):
        self.assertEqual(stable_split("future_article_123"), stable_split("future_article_123"))
        self.assertIn(stable_split("future_article_123"), {"train", "validation", "test"})

    def test_only_analyst_approved_extractions_are_promoted(self):
        approved_review = json.loads(
            (DATA_DIR / "reviewed_extraction_queue.sample.jsonl").read_text(encoding="utf-8")
        )
        pending_review = {
            **approved_review,
            "example_id": "pending_future_001",
            "text": "A separate pending example that must not enter training.",
            "review_status": "pending",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "extraction_examples.jsonl"
            queue_path = Path(temp_dir) / "review_queue.jsonl"
            shutil.copyfile(DATA_DIR / "extraction_examples.jsonl", dataset_path)
            queue_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in (approved_review, pending_review))
                + "\n",
                encoding="utf-8",
            )

            promoted, skipped = promote_reviewed_examples(queue_path, dataset_path)

        self.assertEqual([example["example_id"] for example in promoted], ["reviewed_future_001"])
        self.assertEqual(skipped[0]["reason"], "review_status is pending")

    def test_gdelt_discovery_results_are_never_marked_for_training(self):
        payload = {
            "articles": [
                {
                    "url": "https://publisher.example/future-report",
                    "title": "Future regional defence report",
                    "seendate": "20260715T120000Z",
                    "language": "Arabic",
                    "sourcecountry": "Lebanon",
                    "domain": "publisher.example",
                }
            ]
        }

        with patch("scripts.discover_gdelt_articles._request_json", return_value=payload):
            candidates = discover("defence", 10, "1week")

        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0]["article_text"])
        self.assertFalse(candidates[0]["allowed_for_training"])
        self.assertEqual(candidates[0]["rights_review_status"], "pending")

    def test_wikidata_candidates_preserve_english_and_arabic_aliases_for_review(self):
        bindings = [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q123"},
                "labelEn": {"value": "Example Defence Company"},
                "labelAr": {"value": "شركة الدفاع التجريبية"},
                "aliasesEn": {"value": "EDC|||Example Defence"},
                "aliasesAr": {"value": "الدفاع التجريبية"},
            }
        ]

        candidates = bindings_to_candidates(bindings, "company")

        self.assertEqual(candidates[0]["canonical_name"], "Example Defence Company")
        self.assertIn("شركة الدفاع التجريبية", candidates[0]["aliases"])
        self.assertIn("EDC", candidates[0]["aliases"])
        self.assertEqual(candidates[0]["review_status"], "pending")
        self.assertIn("Q1934969", build_query("company", 100))

    def test_only_reviewed_wikidata_entities_are_merged_into_catalogue(self):
        approved = {
            "entity_id": "wikidata.Q7240",
            "wikidata_id": "Q7240",
            "type": "company",
            "canonical_name": "Lockheed Martin",
            "aliases": ["LockMart", "لوكهيد-مارتن"],
            "review_status": "approved",
            "reviewed_by": "prototype_analyst",
            "reviewed_at": "2026-07-15T18:00:00Z",
        }
        pending = {
            **approved,
            "entity_id": "wikidata.Q999",
            "wikidata_id": "Q999",
            "canonical_name": "Pending Company",
            "review_status": "pending",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            catalogue_path = Path(temp_dir) / "entities.json"
            candidates_path = Path(temp_dir) / "candidates.jsonl"
            shutil.copyfile(DATA_DIR / "entities.json", catalogue_path)
            candidates_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in (approved, pending)) + "\n",
                encoding="utf-8",
            )

            promoted, skipped = promote_reviewed_entities(candidates_path, catalogue_path)
            catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))

        lockheed = next(entity for entity in catalogue if entity["canonical_name"] == "Lockheed Martin")
        self.assertEqual(promoted, ["wikidata.Q7240"])
        self.assertIn("LockMart", lockheed["aliases"])
        self.assertNotIn("Pending Company", {entity["canonical_name"] for entity in catalogue})
        self.assertEqual(skipped[0]["reason"], "not approved")

    def test_adapter_output_parser_ignores_surrounding_text_and_requires_schema(self):
        labels = {
            **{category: [] for category in FILTER_CATEGORIES},
            "relationships": [],
        }
        raw_output = f"Here is the result:\n{json.dumps(labels)}\nFinished."

        parsed = validate_extraction_response(parse_first_json_object(raw_output))

        self.assertEqual(parsed, labels)
        with self.assertRaises(ValueError):
            validate_extraction_response({"companies": ["Lockheed Martin"]})

    def test_article_extraction_text_combines_title_summary_and_body_once(self):
        article = {
            "title": "Regional air-defence update",
            "summary": "Officials discussed radar integration.",
            "body": "The meeting also covered training and maintenance.",
            "text": "The meeting also covered training and maintenance.",
        }

        text = article_extraction_text(article)

        self.assertIn(article["title"], text)
        self.assertIn(article["summary"], text)
        self.assertEqual(text.count(article["body"]), 1)

    def test_catalogue_aliases_fill_entities_missed_by_model_extraction(self):
        article = load_incoming_articles(DATA_DIR)[0]
        model_labels = {
            **{category: [] for category in FILTER_CATEGORIES},
            "companies": ["Lockheed Martin"],
            "countries": ["Saudi Arabia"],
            "relationships": [],
        }

        labels, entities, _ = merge_extraction_with_catalogue(
            article,
            model_labels,
            self.catalogue,
        )

        self.assertIn("Saudi Ministry of Defense", labels["organizations"])
        self.assertIn("Air defense systems", labels["systems"])
        self.assertIn("Lockheed Martin", {entity["canonical_name"] for entity in entities})

    def test_saved_adapter_output_is_attached_to_matching_article(self):
        articles = enrich_articles(load_incoming_articles(DATA_DIR), self.catalogue)
        labels = {
            **{category: [] for category in FILTER_CATEGORIES},
            "companies": ["Lockheed Martin"],
            "organizations": ["Saudi Ministry of Defense"],
            "relationships": [],
        }
        record = {
            "article_id": "incoming_001",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "labels": labels,
            "review_status": "pending",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "extractions.jsonl"
            output_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            records = load_extraction_outputs(output_path)

        enriched = apply_extraction_outputs(articles, records)
        article = next(item for item in enriched if item["article_id"] == "incoming_001")

        self.assertEqual(article["llm_extraction"]["review_status"], "pending")
        self.assertIn("Lockheed Martin", article["tags"]["companies"])
        self.assertIn("Saudi Ministry of Defense", article["tags"]["organizations"])

    def test_pending_adapter_output_can_be_excluded_from_production_tags(self):
        articles = [{"article_id": "future_1", "tags": {category: [] for category in FILTER_CATEGORIES}}]
        labels = {
            **{category: [] for category in FILTER_CATEGORIES},
            "companies": ["Unreviewed Company"],
            "relationships": [],
        }
        records = [{"article_id": "future_1", "labels": labels, "review_status": "pending"}]

        enriched = apply_extraction_outputs(articles, records, include_pending=False)

        self.assertNotIn("llm_extraction", enriched[0])
        self.assertEqual(enriched[0]["tags"]["companies"], [])


if __name__ == "__main__":
    unittest.main()
