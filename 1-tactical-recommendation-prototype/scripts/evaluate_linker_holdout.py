"""Freeze a new holdout after candidate-ranker development, then evaluate fixed weights."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learned_linker import LearnedEntityLinker, context_at, normalized
from src.portfolio_config import ARTIFACTS, digest, read_jsonl, read_json, write_jsonl, write_json


def metrics(rows):
    accepted = [r for r in rows if r["prediction"]["entity_id"]]
    unknown = [r for r in rows if not r["gold_in_catalogue"]]
    unseen = [r for r in rows if not r["alias_prediction"]]
    return {"mentions": len(rows), "accepted": len(accepted),
        "accepted_precision": sum(r["prediction"]["entity_id"] == r["qid"] for r in accepted)/max(1,len(accepted)),
        "coverage": len(accepted)/max(1,len(rows)),
        "link_accuracy_with_abstention": sum(r["prediction"]["entity_id"] == r["qid"] for r in rows)/max(1,len(rows)),
        "alias_accuracy": sum(r["alias_prediction"] == r["qid"] for r in rows)/max(1,len(rows)),
        "out_of_catalogue_mentions": len(unknown),
        "out_of_catalogue_abstention_rate": sum(r["prediction"]["entity_id"] is None for r in unknown)/max(1,len(unknown)),
        "unseen_or_ambiguous_alias_mentions": len(unseen),
        "unseen_or_ambiguous_alias_accuracy": sum(r["prediction"]["entity_id"] == r["qid"] for r in unseen)/max(1,len(unseen))}


def main(limit):
    folder = ARTIFACTS / "linker"
    linker = LearnedEntityLinker()
    frozen_path = folder / "final_holdout.jsonl"
    if not frozen_path.exists():
        previously_seen = {r["article_id"] for r in read_jsonl(folder / "test_predictions.jsonl")}
        articles = sorted([r for r in read_jsonl(ARTIFACTS / "corpus.jsonl")
                           if r["split"] == "test" and r["article_id"] not in previously_seen],
                          key=lambda r: digest(r["article_id"]))
        rows = []
        for language in ("en", "ar"):
            subset = [{**m, "article_id": a["article_id"], "language": language,
                       "context": context_at(a["body"], m["start"], m["end"])}
                      for a in articles if a["language"] == language for m in a["mentions"][:5]][:limit//2]
            rows.extend(subset)
        write_jsonl(frozen_path, rows)
        write_json(folder / "final_holdout_manifest.json", {"model_version": linker.version,
            "data_hash": digest(rows), "excluded_development_articles": sorted(previously_seen),
            "scope": "Fresh article-disjoint holdout frozen after two development iterations; published mention IDs, not exhaustive extraction labels"})
    rows = read_jsonl(frozen_path)
    manifest = read_json(folder / "final_holdout_manifest.json")
    if manifest["model_version"] != linker.version or manifest["data_hash"] != digest(rows):
        raise ValueError("Frozen holdout or model changed. Do not silently reevaluate a tuned model.")
    output_path = folder / "final_predictions.jsonl"
    predictions = read_jsonl(output_path) if output_path.exists() else []
    aliases = {}
    qids = {e["qid"] for e in linker.entities}
    for entity in linker.entities:
        for alias in entity["normalized_aliases"]:
            aliases.setdefault(alias, set()).add(entity["qid"])
    for row in rows[len(predictions):]:
        candidates = aliases.get(normalized(row["text"]), set())
        predictions.append({**row, "prediction": linker.link(row["text"], row["context"]),
                            "gold_in_catalogue": row["qid"] in qids,
                            "alias_prediction": next(iter(candidates)) if len(candidates) == 1 else None})
        write_jsonl(output_path, predictions)
    report = {"status": "evaluated", "manifest": manifest, "overall": metrics(predictions),
        "by_language": {lang: metrics([p for p in predictions if p["language"] == lang]) for lang in ("en", "ar")}}
    write_json(folder / "final_metrics.json", report)
    print(report["overall"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mentions", type=int, default=200)
    main(parser.parse_args().mentions)
