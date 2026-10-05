"""Audit/save a distinct case-only derived weak dataset without rerunning Qwen."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finish_phase2 import verify_completion
from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3
from src.portfolio_config import (ARTIFACTS, LEGACY_ADAPTER, configure_cache, digest,
                                  file_digest, read_json, read_jsonl, write_json, write_jsonl)
from src.weak_label_recovery import POLICY, recover_case_only, validate_weak_label


def collect_recovery():
    configure_cache()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(LEGACY_ADAPTER / "tokenizer.json"))
    old = ARTIFACTS / "phase3/weak_labels_v3"
    queue = {r["article_id"]: r for r in read_jsonl(old / "queue.jsonl")}
    attempts = read_jsonl(old / "attempts.jsonl")
    teacher = read_json(old / "identity.json")["teacher"]
    if file_digest(LEGACY_ADAPTER / "tokenizer.json") != teacher["tokenizer_files"]["tokenizer.json"]:
        raise ValueError("Original teacher tokenizer changed")
    examples, audit = [], []
    for attempt in attempts:
        candidate = queue[attempt["article_id"]]
        if attempt["outcome"] == "accepted":
            example = attempt["accepted_example"]
            validate_weak_label(example["labels"], example["text"])
            examples.append(example)
            continue
        tokens = tokenizer.encode(candidate["text"], add_special_tokens=False).ids
        expected_chunks = len(range(0, len(tokens), teacher["chunk_stride"]))
        try:
            labels, provenance = recover_case_only(candidate, attempt, expected_chunks, validate_weak_label)
            examples.append({**candidate, "split": "train", "schema_version": EXTRACTION_SCHEMA_V3,
                "labels": labels, "label_source": "machine_assisted_case_only_derived_not_human_gold",
                "teacher_identity": teacher, "exhaustiveness": "unverified", "derivation": provenance,
                "field_checks": {c: "literal_spans_checked_category_accuracy_not_human_verified" for c in CURRENT_FILTER_CATEGORIES}})
            audit.append({"article_id": candidate["article_id"], "recovered": True, "derivation": provenance})
        except ValueError as exc:
            audit.append({"article_id": candidate["article_id"], "recovered": False, "reason": str(exc)})
    languages = dict(Counter(r["language"] for r in examples))
    report = {"policy": POLICY, "source_attempts_sha256": file_digest(old / "attempts.jsonl"),
              "source_queue_sha256": file_digest(old / "queue.jsonl"), "examples": len(examples),
              "languages": languages, "raw_accepted": sum(a["outcome"] == "accepted" for a in attempts),
              "derived": sum(a["recovered"] for a in audit),
              "nonempty_fields": {c: sum(bool(r["labels"][c]) for r in examples) for c in CURRENT_FILTER_CATEGORIES},
              "dataset_hash": digest(examples), "implementation_sha256": file_digest(__file__),
              "scope": "Weak training only; strict inference and validation unchanged; no human completeness claim"}
    return examples, audit, report


def recover(save=False):
    verify_completion()
    examples, audit, report = collect_recovery()
    if save:
        languages = report["languages"]
        if len(examples) < 20 or min(languages.get("en", 0), languages.get("ar", 0)) < 4:
            raise ValueError(f"Recovered dataset still insufficient: {len(examples)}, {languages}")
        folder = ARTIFACTS / "phase3/weak_labels_v3_case_recovery_v1"
        path = folder / "examples.jsonl"
        previous = read_json(folder / "manifest.json")
        if previous and previous["dataset_hash"] != report["dataset_hash"]:
            raise ValueError("Derived dataset is immutable; use a new version")
        write_jsonl(path, examples)
        write_jsonl(folder / "recovery_audit.jsonl", audit)
        report["examples_sha256"] = file_digest(path)
        write_json(folder / "manifest.json", report)
        write_json(ARTIFACTS / "phase3/training_dataset.json", {
            "path": str(path.relative_to(ARTIFACTS)), "sha256": file_digest(path),
            "manifest_sha256": file_digest(folder / "manifest.json"), "policy": POLICY})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.save), indent=2))
