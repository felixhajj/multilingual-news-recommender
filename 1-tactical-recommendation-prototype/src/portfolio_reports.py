"""Evidence pages are generated from artifacts, not manually entered scores."""
from src.portfolio_config import ARTIFACTS, ROOT, read_json, read_jsonl, write_json


def evidence_report():
    runs = []
    for path in sorted((ARTIFACTS / "runs").glob("*/manifest.json")):
        value = read_json(path)
        runs.append({key: value.get(key) for key in (
            "run_id", "stage", "status", "dataset_examples", "optimizer_steps", "unique_articles_seen",
            "supervised_tokens", "maximum_weight_change", "elapsed_seconds", "promotion_status", "label_source")})
    review_path = ARTIFACTS / "review" / "extraction.jsonl"
    reviews = read_jsonl(review_path) if review_path.exists() else []
    report = {"corpus": read_json(ARTIFACTS / "corpus_manifest.json", {}),
            "linker": read_json(ARTIFACTS / "linker" / "final_metrics.json", {"status": "final_holdout_pending"}),
            "linker_development": read_json(ARTIFACTS / "linker" / "metrics.json", {"status": "not_trained"}),
            "active_model": read_json(ARTIFACTS / "active_model.json", {"run_id": "legacy-20-mock-examples", "status": "historical_baseline_not_validated_on_real_news"}),
            "extraction_experiments": runs,
            "extraction_evaluation": read_json(ARTIFACTS / "extraction_evaluation.json", {"status": "awaiting_human_review", "metrics": None}),
            "review": {"total": len(reviews), "human_reviewed": sum(r["review_status"] == "human_reviewed" for r in reviews)},
            "recommendation_evaluation": read_json(ARTIFACTS / "recommendation_evaluation.json", {"status": "awaiting_human_judgments", "metrics": None}),
            "release": read_json(ARTIFACTS / "release_manifest.json", {"status": "not_prepared", "articles": 0}),
            "processing_budget": read_json(ARTIFACTS / "processing_budget.json", {}),
            "limitations": ["Historical open news; no enterprise affiliation",
                            "Machine-assisted extraction labels are not human gold",
                            "Only completed inference records enter the release collection",
                            "A pretrained model may already have encountered historical test articles"]}
    completion = read_json(ARTIFACTS / "phase3/completion.json", {})
    selected = completion.get("selected_extractor")
    if completion.get("status") == "evaluated" and selected:
        report["active_model"] = {"run_id": selected, "status": "validation_selected_research_release",
                                  "selection_lock_hash": completion.get("selection_lock_hash")}
        report["historical_extraction_evaluation"] = report["extraction_evaluation"]
        report["extraction_evaluation"] = {"status": "evaluated", **read_json(
            ARTIFACTS / "phase3/reports" / f"{selected}_test.json", {})}
        report["historical_recommendation_evaluation"] = report["recommendation_evaluation"]
        report["recommendation_evaluation"] = read_json(
            ARTIFACTS / "phase4/selected_ranking/report_v2.json", report["recommendation_evaluation"])
        report["selected_linker_evaluation"] = read_json(ARTIFACTS / "phase3/reports/linker_v3_test.json", {})
        report["phase4"] = read_json(ARTIFACTS / "phase4/completion.json", {})
    return report


def publish_evidence_snapshot():
    report = evidence_report()
    write_json(ROOT / "data" / "release" / "evidence.json", report)
    return report
