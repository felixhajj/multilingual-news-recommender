"""Audit saved validation predictions only; never claim retroactive inference binding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.learned_linker import linker_artifact_identity, normalized
from src.portfolio_config import ARTIFACTS, file_digest, read_json, read_jsonl, write_json

VALIDATION_POLICY = {"accepted_precision_floor": .9, "minimum_accepted": 10,
                     "unknown_abstention_floor": .9,
                     "rule": "Also require overall correct-link accuracy above unique-alias baseline"}


def metrics(rows, entities, threshold):
    aliases = {}
    qids = {e["qid"] for e in entities}
    for entity in entities:
        for alias in entity["normalized_aliases"]:
            aliases.setdefault(alias, set()).add(entity["qid"])
    for row in rows:
        expected = row["predicted_qid"] if row["score"] >= threshold else None
        if expected != row["accepted_qid"] or row["gold_in_catalogue"] != (row["qid"] in qids):
            raise ValueError("Saved validation decision disagrees with model threshold/catalogue")
    accepted = [r for r in rows if r["accepted_qid"] is not None]
    known = [r for r in rows if r["gold_in_catalogue"]]
    unknown = [r for r in rows if not r["gold_in_catalogue"]]
    unseen = [r for r in known if normalized(r["text"]) not in aliases]
    ambiguous = [r for r in known if len(aliases.get(normalized(r["text"]), ())) > 1]
    ratio = lambda count, total: count / total if total else None
    accuracy = lambda subset: ratio(sum(r["accepted_qid"] == r["qid"] for r in subset), len(subset))
    return {"mentions": len(rows), "accepted": len(accepted),
            "accepted_precision": accuracy(accepted), "accepted_coverage": ratio(len(accepted), len(rows)),
            "correct_link_accuracy": accuracy(rows),
            "unique_alias_accuracy": ratio(sum(r["alias_prediction"] == r["qid"] for r in rows), len(rows)),
            "catalogue_coverage": ratio(len(known), len(rows)),
            "candidate_recall_at_10_known": ratio(sum(r["gold_retrieved"] for r in known), len(known)),
            "out_of_catalogue_mentions": len(unknown),
            "unknown_abstention": ratio(sum(r["accepted_qid"] is None for r in unknown), len(unknown)),
            "unseen_known_alias_mentions": len(unseen), "unseen_known_alias_accuracy": accuracy(unseen),
            "ambiguous_known_alias_mentions": len(ambiguous), "ambiguous_known_alias_accuracy": accuracy(ambiguous)}


def main():
    directory = ARTIFACTS / "linker"
    config = read_json(directory / "model.json")
    rows = read_jsonl(directory / "validation_predictions.jsonl")
    entities = read_json(directory / "entities.json")
    result = metrics(rows, entities, config["threshold"])
    passed = (result["accepted"] >= VALIDATION_POLICY["minimum_accepted"]
              and (result["accepted_precision"] or 0) >= VALIDATION_POLICY["accepted_precision_floor"]
              and (result["unknown_abstention"] if result["unknown_abstention"] is not None else 1)
              >= VALIDATION_POLICY["unknown_abstention_floor"]
              and result["correct_link_accuracy"] > result["unique_alias_accuracy"])
    report = {"role": "validation", "policy": VALIDATION_POLICY, "metric_gate_passed": passed,
              "prediction_file_sha256": file_digest(directory / "validation_predictions.jsonl"),
              "current_artifact_audit": linker_artifact_identity(directory), "metrics": result,
              "by_language": {lang: metrics([r for r in rows if r["language"] == lang], entities, config["threshold"])
                              for lang in ("en", "ar")},
              "deployment_eligible": False,
              "limitations": ["Published hyperlink IDs are partial annotations, not human-gold extraction labels.",
                              "Historical predictions did not bind vector bytes and feature implementation at inference time.",
                              "Current artifact hashes are an audit, not proof these bytes generated historical predictions.",
                              "Fresh identity-bound validation is required before deployment; final-test files were not read."]}
    path = ARTIFACTS / "phase3/reports/linker_historical_validation.json"
    existing = read_json(path)
    if existing and existing != report:
        raise ValueError("Saved audit changed; preserve it and use a new report version")
    write_json(path, report)
    print(result)
    print("Metric gate:", passed, "; deployment eligible: False (historical provenance)")


if __name__ == "__main__":
    main()
