from functools import lru_cache
from pathlib import Path
import json


@lru_cache(maxsize=None)
def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_users(data_dir):
    return _load_json(str(Path(data_dir) / "users.json"))


def load_articles(data_dir):
    return _load_json(str(Path(data_dir) / "articles.json"))


def load_filter_vocabulary(data_dir):
    return _load_json(str(Path(data_dir) / "filter_vocabulary.json"))


def load_entity_catalogue(data_dir):
    return _load_json(str(Path(data_dir) / "entities.json"))


def load_incoming_articles(data_dir):
    return _load_json(str(Path(data_dir) / "incoming_articles.json"))


def load_evaluation_cases(data_dir):
    return _load_json(str(Path(data_dir) / "evaluation_cases.json"))


def load_extraction_dataset(data_dir):
    from src.extraction_data import load_extraction_examples

    return load_extraction_examples(Path(data_dir) / "extraction_examples.jsonl")


def load_corpus_source_manifest(data_dir):
    from src.corpus_ingestion import load_source_manifest

    return load_source_manifest(Path(data_dir) / "corpus_sources.json")


def load_domain_corpus(data_dir):
    from src.corpus_ingestion import load_corpus_documents

    return load_corpus_documents(Path(data_dir) / "domain_corpus_sample.jsonl")
