"""Verify Phase 4 artifacts and write a compact, reproducible completion record."""
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_selected_ranking import report_from_saved_evaluation
from scripts.validate_news_notebooks import NOTEBOOKS, validation_fingerprint
from src.portfolio_config import ARTIFACTS, file_digest, read_json, read_jsonl, write_json


PHASE = ARTIFACTS / "phase4"
APP_CHECK = PHASE / "live_app_check" / "report.json"
APP_PROGRESS = PHASE / "live_app_check" / "progress.json"
TEST_LOG = PHASE / "live_app_check" / "unit-tests-20261002-final.log"
RANKING_REPORT = PHASE / "selected_ranking" / "report_v2.json"
RANKING_MANIFEST = PHASE / "selected_ranking" / "manifest.json"
NOTEBOOK_REPORT = ARTIFACTS / "notebook_validation" / "report.json"
OUTPUT = PHASE / "completion.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    app = read_json(APP_CHECK, {})
    progress = read_json(APP_PROGRESS, {})
    require(app.get("status") == "completed_with_model_failures", "Live-app smoke report is not complete")
    require(app.get("open_checks") == [], "Live-app report still has open checks")
    require(all(value is True for value in app.get("completion_checks", {}).values()),
            "At least one live-app completion check is not true")
    require(app.get("completion_checks", {}).get("distinct_interests_recompute_rankings") is True,
            "Distinct-interest ranking behavior is not verified")
    require(app.get("recommendation_smoke", {}).get("top_five_order_changed") is True,
            "Saved recommendation smoke did not produce changed top-five orders")
    for key in ("recommend_english", "recommend_arabic"):
        require(progress.get("ui", {}).get(key, {}).get("visible_cards_match_pipeline") is True,
                f"Live UI cards do not match pipeline output for {key}")
    require(progress.get("cache_checks", {}).get("same_input_hit") is True
            and progress.get("cache_checks", {}).get("edited_title_miss") is True,
            "Cache identity checks are incomplete")

    smoke = app.get("article_smoke", {})
    analyses = read_jsonl(PHASE / "live_app_check" / "analyses.jsonl")
    failures = read_jsonl(PHASE / "live_app_check" / "failures.jsonl")
    require(smoke.get("selected") == 30 and smoke.get("successful") == len(analyses)
            and smoke.get("recorded_failures") == len(failures)
            and len(analyses) + len(failures) == 30,
            "The 30-article smoke batch is not fully accounted for")
    require(all(row.get("raw_model_outputs") for row in failures),
            "A failed smoke inference has no preserved raw model output")
    require(smoke.get("not_a_quality_evaluation") is True,
            "Unlabeled smoke results must not be reported as quality metrics")

    release = read_json(ARTIFACTS / "release_manifest.json", {})
    database = ARTIFACTS / "release.sqlite"
    database_hash = file_digest(database)
    require(release.get("articles") == 300, "Selected release manifest does not contain 300 articles")
    require(app.get("release_index", {}).get("database_sha256_before") == database_hash
            and app.get("release_index", {}).get("database_sha256_after") == database_hash,
            "The release database changed during the live checks")
    with sqlite3.connect(database) as db:
        indexed_count = db.execute(
            "SELECT COUNT(DISTINCT article_id) FROM articles WHERE pipeline_identity=?",
            (release["pipeline_identity"],),
        ).fetchone()[0]
    require(indexed_count == 300, "SQLite selected index does not contain exactly 300 articles")

    saved_ranking = read_json(RANKING_REPORT, {})
    ranking_manifest = read_json(RANKING_MANIFEST, {})
    recalculated = report_from_saved_evaluation()
    compared_fields = ("pipeline_identity", "model_identity", "selection_lock_hash", "reference_sha256",
                       "ranking_implementation_sha256", "human_reviewed_judgments", "unique_articles",
                       "validation_averages", "test_averages", "locked_selected_method", "extraction_failure_ids")
    for field in compared_fields:
        require(saved_ranking.get(field) == recalculated.get(field),
                f"Saved selected-pipeline ranking report does not reproduce: {field}")
    require(saved_ranking.get("human_reviewed_judgments") == 30
            and ranking_manifest.get("reference_sha256") == saved_ranking.get("reference_sha256"),
            "Ranking metrics are not bound to the frozen 30 judgments")

    notebook_rows = {row.get("notebook"): row for row in read_json(NOTEBOOK_REPORT, [])}
    notebook_evidence = {}
    for name in NOTEBOOKS:
        row = notebook_rows.get(name, {})
        executed_path = ARTIFACTS / "notebook_validation" / name
        require(row.get("status") == "passed" and executed_path.is_file()
                and file_digest(executed_path) == row.get("executed_sha256")
                and validation_fingerprint(name) == row.get("input_fingerprint"),
                f"Notebook execution is missing or stale: {name}")
        notebook_evidence[name] = {"status": "passed", "executed_sha256": row["executed_sha256"],
                                   "scope": row.get("scope")}

    test_log = TEST_LOG.read_text(encoding="utf-8")
    match = re.search(r"Ran (\d+) tests in ([0-9.]+)s", test_log)
    require(match is not None and "\nOK" in test_log, "Final complete regression test log did not pass")
    test_evidence = {"status": "passed", "tests": int(match.group(1)),
                     "elapsed_seconds": float(match.group(2)), "log_sha256": file_digest(TEST_LOG)}

    process = read_json(PHASE / "live_app_check_process.json", {})
    require(process.get("status") == "exited" and process.get("return_code") == 0,
            "Supervised live-app process did not exit successfully")
    budget = read_json(ARTIFACTS / "processing_budget.json", {})
    require(budget.get("consumed_seconds", 0) <= budget.get("limit_seconds", 0),
            "The cumulative processing cap was exceeded")

    evidence = {
        "status": "phase4_complete_with_model_failures_preserved",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_identity": release.get("pipeline_identity"),
        "selected_run_id": release.get("model", {}).get("run_id"),
        "release_index": {"articles": indexed_count, "database_sha256": database_hash},
        "live_app": {"status": app["status"], "open_checks": [],
                     "completion_checks": app["completion_checks"],
                     "interest_order_changed": app["recommendation_smoke"]["top_five_order_changed"],
                     "smoke_inputs": 30, "successful": len(analyses), "model_failures_preserved": len(failures),
                     "failure_ids": [row.get("article_id") for row in failures]},
        "selected_pipeline_ranking": {"status": saved_ranking.get("status"),
                                      "judgments": saved_ranking.get("human_reviewed_judgments"),
                                      "unique_articles": saved_ranking.get("unique_articles"),
                                      "validation_averages": saved_ranking.get("validation_averages"),
                                      "test_averages": saved_ranking.get("test_averages"),
                                      "locked_selected_method": saved_ranking.get("locked_selected_method"),
                                      "extraction_failure_ids": saved_ranking.get("extraction_failure_ids"),
                                      "recalculated_from_saved_artifacts": True},
        "notebooks": notebook_evidence,
        "regression_tests": test_evidence,
        "budget": {"consumed_seconds": budget["consumed_seconds"],
                   "limit_seconds": budget["limit_seconds"],
                   "remaining_seconds": budget["limit_seconds"] - budget["consumed_seconds"]},
        "next_phase": "Phase 5: package, restore, publish, and verify the portfolio release",
        "public_deployment_verified": False,
    }
    write_json(OUTPUT, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
