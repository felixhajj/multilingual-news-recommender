import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_ingestion import (  # noqa: E402
    corpus_summary,
    load_corpus_documents,
    load_source_manifest,
    select_approved_training_documents,
    write_training_corpus,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a deduplicated domain corpus from reviewed, training-authorized records."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "domain_corpus_sample.jsonl",
        help="Input JSONL containing reviewed article records.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "corpus_sources.json",
        help="Source rights and permitted-use manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "corpus" / "approved_domain_corpus.jsonl",
        help="Output JSONL used by a domain-adaptation job.",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=ROOT / "output" / "corpus" / "rejected_documents.json",
        help="Audit log for records excluded from training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_source_manifest(args.manifest)
    documents = load_corpus_documents(args.input)
    approved, rejected = select_approved_training_documents(documents, manifest)

    write_training_corpus(approved, args.output)
    args.rejections.parent.mkdir(parents=True, exist_ok=True)
    args.rejections.write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = {
        **corpus_summary(approved, rejected),
        "training_corpus": str(args.output),
        "rejection_audit": str(args.rejections),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
