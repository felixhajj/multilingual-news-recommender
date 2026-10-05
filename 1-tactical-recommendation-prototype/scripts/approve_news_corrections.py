"""Validate attributed corrections and create an immutable, explicit training version."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm_extractor import validate_extraction_response
from src.extraction_data import CURRENT_FILTER_CATEGORIES, EXTRACTION_SCHEMA_V3
from src.portfolio_config import ARTIFACTS, digest, read_jsonl, write_json, write_jsonl


def approve(path):
    rows = read_jsonl(path)
    heldout = [r for r in read_jsonl(ARTIFACTS / "corpus.jsonl") if r["split"] != "train"]
    forbidden_ids = {r["article_id"] for r in heldout}
    forbidden_hashes = {digest(r["body"]) for r in heldout} | {digest(r["title"] + "\n\n" + r["body"]) for r in heldout}
    forbidden_groups = {r["group_id"] for r in heldout}
    for row in rows:
        if row.get("review_status") != "approved_for_training" or not row.get("reviewed_by"):
            raise ValueError("Every correction requires explicit approval and reviewer attribution")
        if row.get("article_id") in forbidden_ids or digest(row["text"]) in forbidden_hashes or row.get("group_id") in forbidden_groups:
            raise ValueError("Held-out evaluation content cannot be recycled into training")
        validate_extraction_response(row["labels"], EXTRACTION_SCHEMA_V3)
        if any(mention not in row["text"] for category in CURRENT_FILTER_CATEGORIES for mention in row["labels"][category]):
            raise ValueError("Corrected mentions must be literal source passages")
        row.update(split="train", label_source="human_approved_correction",
                   schema_version=EXTRACTION_SCHEMA_V3)
    version = digest(rows)
    directory = ARTIFACTS / "datasets" / version
    if (directory / "examples.jsonl").exists():
        return directory
    write_jsonl(directory / "examples.jsonl", rows)
    write_json(directory / "manifest.json", {"version": version, "examples": len(rows),
        "status": "approved_not_trained", "retraining": "Explicit run only; promotion still requires validation"})
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrections", type=Path)
    print(approve(parser.parse_args().corrections))
