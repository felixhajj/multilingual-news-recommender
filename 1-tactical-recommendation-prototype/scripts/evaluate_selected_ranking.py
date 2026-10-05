"""Re-evaluate frozen recommendation judgments with the selected Phase 4 pipeline."""
import argparse
import atexit
import json
import math
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news_evaluation import ranking_metrics, validate_ranking_review
from src.news_pipeline import NewsPipeline, rank_records
from src.portfolio_config import (ARTIFACTS, GPU_LOCK, digest, file_digest, read_json,
                                  read_jsonl, write_json, write_jsonl)

MAX_SECTION_SECONDS = 3600
ALLOCATION_ID = "phase4-selected-ranking-20261001"
OUTPUT = ARTIFACTS / "phase4" / "selected_ranking"


class BudgetMeter:
    """Charge actual elapsed time incrementally against the approved phase budget."""

    def __init__(self, requested_seconds):
        self.started = time.monotonic()
        self.last_saved = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        budget = read_json(ARTIFACTS / "processing_budget.json")
        if not budget or budget["consumed_seconds"] > budget["limit_seconds"]:
            raise RuntimeError("Processing budget is missing or already over its cap")
        allocations = budget.setdefault("allocations", [])
        matches = [row for row in allocations if row.get("allocation_id") == ALLOCATION_ID]
        if len(matches) > 1:
            raise RuntimeError("Duplicate recommendation evaluation budget allocation")
        if matches:
            self.allocation = matches[0]
        else:
            self.allocation = {"phase": 4, "operation": "selected_pipeline_recommendation_evaluation",
                               "allocation_id": ALLOCATION_ID, "allowed_seconds": MAX_SECTION_SECONDS,
                               "charged_seconds": 0.0, "accounting": "actual elapsed time, incrementally persisted"}
            allocations.append(self.allocation)
            write_json(ARTIFACTS / "processing_budget.json", budget)
        if self.allocation.get("allowed_seconds") != MAX_SECTION_SECONDS:
            raise RuntimeError("Recommendation evaluation allocation has unexpected limits")
        self.base_charge = float(self.allocation.get("charged_seconds", 0.0))
        if self.base_charge > MAX_SECTION_SECONDS:
            raise RuntimeError("Recommendation evaluation exceeded its one-hour allocation")
        self.session_allowance = min(requested_seconds, MAX_SECTION_SECONDS - self.base_charge,
                                     budget["limit_seconds"] - budget["consumed_seconds"])
        if self.session_allowance <= 0:
            raise RuntimeError("No processing budget remains for recommendation evaluation")

    def elapsed(self):
        return time.monotonic() - self.started

    def deadline(self):
        return self.started + self.session_allowance

    def check(self):
        self.charge()
        if self.elapsed() >= self.session_allowance:
            raise TimeoutError("The approved Phase 4 recommendation-evaluation budget is exhausted")

    def charge(self):
        with self.lock:
            elapsed = min(self.elapsed(), self.session_allowance)
            delta = elapsed - self.last_saved
            if delta <= 0:
                return
            budget = read_json(ARTIFACTS / "processing_budget.json")
            allocation = next(row for row in budget["allocations"]
                              if row.get("allocation_id") == ALLOCATION_ID)
            if allocation.get("charged_seconds", 0.0) + delta > MAX_SECTION_SECONDS + 1e-6:
                raise TimeoutError("The one-hour recommendation-evaluation allocation is exhausted")
            if budget["consumed_seconds"] + delta > budget["limit_seconds"] + 1e-6:
                raise TimeoutError("The cumulative 32-hour processing budget is exhausted")
            allocation["charged_seconds"] = round(allocation.get("charged_seconds", 0.0) + delta, 3)
            budget["consumed_seconds"] += delta
            write_json(ARTIFACTS / "processing_budget.json", budget)
            self.last_saved = elapsed

    def _heartbeat(self):
        while not self.stop_event.wait(15):
            self.charge()

    def start(self):
        thread = threading.Thread(target=self._heartbeat, name="phase4-budget-meter", daemon=True)
        thread.start()
        return thread

    def close(self):
        self.charge()
        self.stop_event.set()


def unique_judged_articles(rows):
    articles = {}
    for row in rows:
        article = row["article"]
        article_id = article["article_id"]
        previous = articles.setdefault(article_id, article)
        if previous != article:
            raise ValueError(f"Frozen pools disagree about article {article_id}")
    return articles


def check_analysis(record, article, pipeline_identity, model_identity):
    if (record.get("status") not in {"ready", "extraction_failed"}
            or record.get("pipeline_identity") != pipeline_identity
            or record.get("model", {}).get("run_id") != model_identity.get("run_id")
            or record.get("model", {}).get("adapter_sha256") != model_identity.get("adapter_sha256")
            or record.get("article", {}).get("article_id") != article["article_id"]
            or record["article"].get("title") != article["title"]
            or record["article"].get("body") != article["body"]):
        raise ValueError(f"Analysis identity or frozen article changed: {article['article_id']}")
    expected_key = digest({"title": article["title"], "body": article["body"], "pipeline": pipeline_identity})
    if record.get("cache_key") != expected_key:
        raise ValueError(f"Analysis content cache key is invalid: {article['article_id']}")
    if record["status"] == "extraction_failed" and not record.get("failure", {}).get("raw_outputs"):
        raise ValueError(f"Failed extraction has no preserved raw output: {article['article_id']}")


def failed_analysis(article, pipeline, exc):
    raw_outputs = getattr(exc, "raw_outputs", None)
    if raw_outputs is None:
        raw_outputs = [getattr(exc, "raw_output", "")]
    if not raw_outputs or not any(isinstance(value, str) and value.strip() for value in raw_outputs):
        raise exc
    return {"status": "extraction_failed", "article": article,
            "cache_key": digest({"title": article["title"], "body": article["body"],
                                 "pipeline": pipeline.identity}),
            "pipeline_identity": pipeline.identity, "model": pipeline.backend.identity,
            "failure": {"error": str(exc), "raw_outputs": raw_outputs,
                        "chunk_metadata": getattr(exc, "chunk_metadata", [])},
            "tags": {}, "review_status": "failed_not_indexed"}


def load_selected_record(db, article, pipeline, release_identity):
    rows = db.execute("SELECT payload, embedding_version, vector FROM articles "
                      "WHERE pipeline_identity=? AND article_id=?",
                      (release_identity, article["article_id"])).fetchall()
    if len(rows) != 1:
        return None
    payload, embedding_version, raw_vector = rows[0]
    record = json.loads(payload)
    check_analysis(record, article, pipeline.identity, pipeline.backend.identity)
    if embedding_version != pipeline.embedding_version:
        raise ValueError("Selected article embedding version differs from the frozen evaluation")
    vector = np.frombuffer(raw_vector, dtype=np.float32).copy()
    if vector.shape != (768,) or not np.isfinite(vector).all() or not math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=1e-3):
        raise ValueError(f"Selected article has an invalid stored vector: {article['article_id']}")
    return record, vector


def report_from_saved_evaluation():
    """Recompute metrics from saved model outputs/vectors, separated by frozen role."""
    manifest = read_json(OUTPUT / "manifest.json")
    if not manifest or not manifest.get("status", "").startswith("completed"):
        raise RuntimeError("There is no completed Phase 4 evaluation to summarize")
    rows = read_jsonl(ARTIFACTS / "review" / "ranking.jsonl")
    if validate_ranking_review(rows) != 30 or file_digest(ARTIFACTS / "review" / "ranking.jsonl") != manifest["reference_sha256"]:
        raise ValueError("Frozen recommendation judgments changed")
    analyses_path, vectors_path = OUTPUT / "analyses.jsonl", OUTPUT / "vectors.npz"
    if (file_digest(analyses_path) != manifest.get("analysis_sha256")
            or file_digest(vectors_path) != manifest.get("vectors_sha256")):
        raise ValueError("Saved evaluation analyses or vectors changed")
    analyses_rows = read_jsonl(analyses_path)
    analyses = {row["article_id"]: row["analysis"] for row in analyses_rows}
    articles = unique_judged_articles(rows)
    if len(analyses) != len(analyses_rows) or set(analyses) != set(articles):
        raise ValueError("Saved analyses do not cover exactly the frozen article IDs")
    for article_id, article in articles.items():
        check_analysis(analyses[article_id], article, manifest["pipeline_identity"], manifest["model_identity"])

    with np.load(vectors_path, allow_pickle=False) as saved:
        vector_ids = saved["article_ids"].tolist()
        vectors = np.asarray(saved["article_vectors"], dtype=np.float32)
        query_ids = saved["query_ids"].tolist()
        query_vectors = np.asarray(saved["query_vectors"], dtype=np.float32)
    expected_article_ids = sorted(articles)
    expected_query_ids = sorted({row["query_id"] for row in rows})
    if (vector_ids != expected_article_ids or query_ids != expected_query_ids
            or vectors.shape != (len(articles), 768)
            or query_vectors.shape != (len(query_ids), 768)
            or not np.isfinite(vectors).all() or not np.isfinite(query_vectors).all()):
        raise ValueError("Saved article/query vector identities or shapes are invalid")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3) or not np.allclose(np.linalg.norm(query_vectors, axis=1), 1, atol=1e-3):
        raise ValueError("Saved recommendation vectors are not normalized")
    vector_by_id = {article_id: vectors[index] for index, article_id in enumerate(vector_ids)}
    query_vector_by_id = {query_id: query_vectors[index] for index, query_id in enumerate(query_ids)}
    weight = read_json(OUTPUT / "report.json")["semantic_weight"]
    results = {}
    for query_id in expected_query_ids:
        pool = [row for row in rows if row["query_id"] == query_id]
        if len(pool) != 10 or len({row["role"] for row in pool}) != 1:
            raise ValueError(f"Frozen query pool is inconsistent: {query_id}")
        relevance = {row["article"]["article_id"]: row["relevance"] for row in pool}
        query = pool[0]
        methods = {}
        for method in ("keyword", "e5", "hybrid"):
            candidate_ids = ([article_id for article_id in relevance
                              if analyses[article_id]["status"] == "ready"] if method == "hybrid"
                             else list(relevance))
            candidates = []
            for article_id in candidate_ids:
                analysis = analyses[article_id]
                candidates.append(analysis if analysis["status"] == "ready" else
                                  {"article": analysis["article"], "cache_key": analysis["cache_key"],
                                   "tags": {}, "status": "text_only_after_extraction_failure"})
            candidate_vectors = np.stack([vector_by_id[article_id] for article_id in candidate_ids])
            ranked = rank_records(candidates, candidate_vectors, query["interest"],
                                  query["required_filters"], method, weight,
                                  query_vector_by_id[query_id] if method != "keyword" else None)
            ranked_ids = [item["article"]["article_id"] for item in ranked]
            methods[method] = {"ranked_ids": ranked_ids, "available_candidates": len(candidate_ids),
                               "metrics": ranking_metrics(ranked_ids, relevance)}
        results[query_id] = {"role": query["role"], "methods": methods}

    roles = sorted({result["role"] for result in results.values()})
    averages_by_role = {}
    for role in roles:
        role_queries = [result for result in results.values() if result["role"] == role]
        averages_by_role[role] = {method: {metric: float(np.mean([
            result["methods"][method]["metrics"][metric] for result in role_queries]))
            for metric in ("recall_at_5", "ndcg_at_5")}
            for method in ("keyword", "e5", "hybrid")}
    if "test" not in averages_by_role:
        raise ValueError("Frozen recommendation review has no test-role query")
    failures = sorted(article_id for article_id, analysis in analyses.items()
                      if analysis["status"] == "extraction_failed")
    historical_path = ARTIFACTS / "phase3" / "reports" / "ranking_test.json"
    historical = read_json(historical_path)
    first_report = read_json(OUTPUT / "report.json")
    report = {"evaluation_version": 2,
              "status": "evaluated_with_extraction_failures" if failures else "evaluated",
              "pipeline_identity": manifest["pipeline_identity"],
              "model_identity": manifest["model_identity"],
              "selection_lock_hash": manifest["selection_lock_hash"],
              "reference_sha256": manifest["reference_sha256"],
              "ranking_implementation_sha256": manifest["ranking_implementation_sha256"],
              "human_reviewed_judgments": len(rows), "unique_articles": len(articles),
              "validation_averages": averages_by_role.get("validation"),
              "test_averages": averages_by_role["test"],
              "averages": averages_by_role["test"],
              "averages_by_role": averages_by_role,
              "queries": results, "semantic_weight": weight,
              "locked_selected_method": first_report["locked_selected_method"],
              "extraction_failure_ids": failures,
              "failure_handling": first_report["failure_handling"],
              "scope": "Top-level averages cover only the unchanged held-out test-role query pools; validation is reported separately. All judgments are pre-existing human labels. Failed Qwen extraction outputs are excluded from hybrid candidates and retained as failures; no tags are fabricated.",
              "historical_phase3_test_report": ({"path": str(historical_path.relative_to(ARTIFACTS)),
                  "sha256": file_digest(historical_path), "averages": historical.get("averages"),
                  "scope_note": "Historical report used the legacy extractor; comparison is descriptive, not a controlled single-variable comparison."}
                  if historical else None),
              "limitations": first_report["limitations"]}
    path = OUTPUT / "report_v2.json"
    if path.exists():
        existing = read_json(path)
        comparable = dict(existing)
        historical_record = comparable.get("historical_phase3_test_report")
        if historical_record:
            comparable["historical_phase3_test_report"] = {**historical_record,
                "path": historical_record["path"].replace("\\", "/")}
            report["historical_phase3_test_report"]["path"] = report["historical_phase3_test_report"]["path"].replace("\\", "/")
        if comparable != report:
            raise ValueError("Role-separated Phase 4 report already exists with different contents")
        # Keep the original report bytes and checksum across host platforms.
        return existing
    write_json(path, report)
    manifest.setdefault("report_versions", [])
    for name in ("report.json", "report_v2.json"):
        file_path = OUTPUT / name
        if file_path.exists() and not any(item.get("path") == name for item in manifest["report_versions"]):
            manifest["report_versions"].append({"path": name, "sha256": file_digest(file_path)})
    manifest["latest_report"] = "report_v2.json"
    manifest["report_sha256"] = file_digest(path)
    write_json(OUTPUT / "manifest.json", manifest)
    return report


def main(hours=1.0):
    if hours <= 0 or hours > 1:
        raise ValueError("This operation is capped at the approved one hour")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    budget_meter = BudgetMeter(int(hours * 3600))
    atexit.register(budget_meter.close)
    frozen_path = ARTIFACTS / "review" / "ranking.jsonl"
    rows = read_jsonl(frozen_path)
    if validate_ranking_review(rows) != 30:
        raise ValueError("All 30 frozen recommendation judgments must remain human-reviewed")
    articles = unique_judged_articles(rows)
    reference_hash = file_digest(frozen_path)
    manifest_path = OUTPUT / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest and manifest.get("reference_sha256") != reference_hash:
        raise ValueError("Frozen review identity changed; refusing to reuse evaluation progress")

    with GPU_LOCK.acquire(timeout=0):
        pipeline = NewsPipeline()
        release = read_json(ARTIFACTS / "release_manifest.json", {})
        if (pipeline.backend.identity.get("run_id") != "extraction-only-v3"
                or pipeline.identity != release.get("pipeline_identity")
                or pipeline.index_identity() != pipeline.identity):
            raise ValueError("Application pipeline is not the locked Phase 4 release pipeline")
        completion = read_json(ARTIFACTS / "phase3" / "completion.json", {})
        if completion.get("selected_extractor") != pipeline.backend.identity.get("run_id"):
            raise ValueError("Selected model differs from the Phase 3 locked extractor")
        identity = {"pipeline_identity": pipeline.identity,
                    "model_identity": pipeline.backend.identity,
                    "linker_identity": pipeline.linker.version,
                    "embedding_version": pipeline.embedding_version,
                    "selection_lock_hash": completion.get("selection_lock_hash"),
                    "release_manifest_sha256": file_digest(ARTIFACTS / "release_manifest.json"),
                    "reference_sha256": reference_hash,
                    "ranking_implementation_sha256": digest(__import__("inspect").getsource(rank_records)),
                    "scope": "Three unchanged ten-article human-judged pools; all 27 unique articles; ranking choices remain locked; not corpus-wide recall"}
        if manifest and any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Pipeline or evaluation identity changed; start a separately versioned evaluation")
        if not manifest:
            manifest = {**identity, "status": "in_progress", "created_at": time.time(),
                        "source_policy": "Reuse only exact selected-pipeline rows; freshly infer absent rows without persisting them to release.sqlite"}
            write_json(manifest_path, manifest)

        saved_path = OUTPUT / "analyses.jsonl"
        saved_rows = read_jsonl(saved_path) if saved_path.exists() else []
        analyses = {row["article_id"]: row["analysis"] for row in saved_rows}
        if len(analyses) != len(saved_rows) or set(analyses) - set(articles):
            raise ValueError("Resumable evaluation analyses contain duplicate or unexpected IDs")
        if any(row.get("source") not in {"fresh_selected_pipeline_inference", "fresh_selected_pipeline_inference_failed",
                                          "verified_selected_release_index"}
               for row in saved_rows):
            raise ValueError("Resumable evaluation contains an unknown analysis source")
        db_path = ARTIFACTS / "release.sqlite"
        db_uri = db_path.resolve().as_uri() + "?mode=ro"
        db = sqlite3.connect(db_uri, uri=True)
        generated_ids = [row["article_id"] for row in saved_rows if row["source"].startswith("fresh_selected_pipeline_inference")]
        failed_ids = [row["article_id"] for row in saved_rows
                      if row["analysis"].get("status") == "extraction_failed"]
        meter_thread = budget_meter.start()
        pipeline.backend.processing_deadline = budget_meter.deadline() - min(
            60, max(1, budget_meter.session_allowance * 0.05))
        try:
            for article_id, article in articles.items():
                budget_meter.check()
                if article_id in analyses:
                    check_analysis(analyses[article_id], article, pipeline.identity, pipeline.backend.identity)
                    continue
                selected = load_selected_record(db, article, pipeline, pipeline.identity)
                if selected is None:
                    source = {"article_id": article_id, "title": article["title"],
                              "body": article["body"], **{k: v for k, v in article.items()
                                                        if k not in {"article_id", "title", "body"}}}
                    try:
                        record = pipeline.analyze_article(source, persist=False, force=True)
                    except ValueError as exc:
                        record = failed_analysis(source, pipeline, exc)
                        failed_ids.append(article_id)
                    generated_ids.append(article_id)
                else:
                    record, _ = selected
                check_analysis(record, article, pipeline.identity, pipeline.backend.identity)
                analyses[article_id] = record
                source_kind = ("fresh_selected_pipeline_inference_failed" if record["status"] == "extraction_failed"
                               else "fresh_selected_pipeline_inference" if selected is None
                               else "verified_selected_release_index")
                saved_rows.append({"article_id": article_id, "analysis": record, "source": source_kind})
                write_jsonl(saved_path, saved_rows)
                budget_meter.charge()
                print(f"Prepared {len(analyses)}/{len(articles)} judged articles; newly analyzed {len(generated_ids)}", flush=True)
        finally:
            db.close()
            budget_meter.charge()

        missing = set(articles) - set(analyses)
        if missing:
            raise RuntimeError(f"Isolated evaluation incomplete; missing {sorted(missing)}")

        vectors_by_id = {}
        db = sqlite3.connect(db_uri, uri=True)
        try:
            for article_id in articles:
                selected = load_selected_record(db, articles[article_id], pipeline, pipeline.identity)
                if selected is not None:
                    vectors_by_id[article_id] = selected[1]
        finally:
            db.close()
        to_encode = [article_id for article_id in articles if article_id not in vectors_by_id]
        if to_encode:
            budget_meter.check()
            texts = ["passage: " + articles[article_id]["title"] + "\n\n" + articles[article_id]["body"]
                     for article_id in to_encode]
            encoded = pipeline.encoder(texts)
            for article_id, vector in zip(to_encode, encoded):
                vectors_by_id[article_id] = np.asarray(vector, dtype=np.float32)
            budget_meter.charge()
        budget_meter.check()
        query_rows = {}
        for query_id in sorted({row["query_id"] for row in rows}):
            pool = [row for row in rows if row["query_id"] == query_id]
            if len(pool) != 10 or len({row["interest"] for row in pool}) != 1 or len({json.dumps(row["required_filters"], sort_keys=True) for row in pool}) != 1:
                raise ValueError(f"Frozen query pool is inconsistent: {query_id}")
            query_rows[query_id] = pool[0]
        query_texts = ["query: " + query_rows[qid]["interest"].strip() for qid in sorted(query_rows)]
        query_vectors = pipeline.encoder(query_texts)
        budget_meter.charge()
        article_ids = sorted(articles)
        vectors = np.stack([vectors_by_id[article_id] for article_id in article_ids])
        if vectors.shape != (len(articles), 768) or not np.isfinite(vectors).all():
            raise ValueError("Evaluation vectors are invalid")
        np.savez_compressed(OUTPUT / "vectors.npz", article_ids=np.asarray(article_ids),
                            article_vectors=vectors, query_ids=np.asarray(sorted(query_rows)),
                            query_vectors=query_vectors)

        semantic_weight = pipeline.semantic_weight
        results = {}
        for query_index, query_id in enumerate(sorted(query_rows)):
            pool = [row for row in rows if row["query_id"] == query_id]
            relevance = {row["article"]["article_id"]: row["relevance"] for row in pool}
            query = query_rows[query_id]
            methods = {}
            for method in ("keyword", "e5", "hybrid"):
                query_vector = query_vectors[query_index] if method != "keyword" else None
                candidate_ids = ([article_id for article_id in relevance
                                  if analyses[article_id]["status"] == "ready"] if method == "hybrid"
                                 else list(relevance))
                candidates = []
                for article_id in candidate_ids:
                    analysis = analyses[article_id]
                    if analysis["status"] == "ready":
                        candidates.append(analysis)
                    else:
                        candidates.append({"article": analysis["article"], "cache_key": analysis["cache_key"],
                                           "tags": {}, "status": "text_only_after_extraction_failure"})
                pool_vectors = np.stack([vectors_by_id[article_id] for article_id in candidate_ids])
                ranked = rank_records(candidates, pool_vectors, query["interest"],
                                      query["required_filters"], method, semantic_weight, query_vector)
                ranked_ids = [item["article"]["article_id"] for item in ranked]
                methods[method] = {"ranked_ids": ranked_ids,
                                   "available_candidates": len(candidate_ids),
                                   "metrics": ranking_metrics(ranked_ids, relevance)}
            results[query_id] = {"role": query["role"], "methods": methods}
        averages = {method: {metric: float(np.mean([results[q]["methods"][method]["metrics"][metric]
                                                    for q in results]))
                             for metric in ("recall_at_5", "ndcg_at_5")}
                    for method in ("keyword", "e5", "hybrid")}
        historical_path = ARTIFACTS / "phase3" / "reports" / "ranking_test.json"
        historical = read_json(historical_path)
        report = {"status": "evaluated_with_extraction_failures" if failed_ids else "evaluated", **identity,
                  "human_reviewed_judgments": len(rows), "unique_articles": len(articles),
                  "newly_analyzed_article_ids": sorted(generated_ids),
                  "successful_new_inference_ids": sorted(set(generated_ids) - set(failed_ids)),
                  "extraction_failure_ids": sorted(failed_ids),
                  "reused_selected_index_article_ids": sorted(set(articles) - set(generated_ids)),
                  "semantic_weight": semantic_weight, "locked_selected_method": pipeline.ranking_method,
                  "queries": results, "averages": averages,
                  "historical_phase3_test_report": ({"path": str(historical_path.relative_to(ARTIFACTS)),
                      "sha256": file_digest(historical_path), "averages": historical.get("averages"),
                      "scope_note": "Historical test rows use the legacy extractor index; not a like-for-like selected-pipeline comparison."}
                      if historical else None),
                  "failure_handling": ("Malformed model output is preserved verbatim and that article is excluded from hybrid candidates; keyword/E5 use article text only. No tags are fabricated."
                                       if failed_ids else "No model-output failures occurred."),
                  "limitations": ["Only three fixed ten-article pools were judged.",
                                  "All 30 judgments are human-reviewed, but this is not corpus-wide Recall@5.",
                                  "The test query pool remains a small post-selection check; no model or ranking choice is changed by these scores."]}
        report_path = OUTPUT / "report.json"
        existing_report = read_json(report_path)
        if existing_report and existing_report != report:
            raise ValueError("Existing Phase 4 report differs; preserve it and start a new version")
        write_json(report_path, report)
        manifest["status"] = "completed_with_model_failures" if failed_ids else "completed"
        manifest["completed_at"] = time.time()
        manifest["analysis_sha256"] = file_digest(saved_path)
        manifest["vectors_sha256"] = file_digest(OUTPUT / "vectors.npz")
        manifest["report_sha256"] = file_digest(report_path)
        manifest["new_inference_count"] = len(generated_ids)
        write_json(manifest_path, manifest)
        budget_meter.close()
        role_report = report_from_saved_evaluation()
        print(json.dumps({"status": role_report["status"], "test_averages": role_report["test_averages"],
                          "validation_averages": role_report["validation_averages"],
                          "newly_analyzed": sorted(generated_ids),
                          "charged_seconds": next(row["charged_seconds"] for row in
                              read_json(ARTIFACTS / "processing_budget.json")["allocations"]
                              if row.get("allocation_id") == ALLOCATION_ID)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--report-only", action="store_true",
                        help="Recalculate role-separated scores from saved outputs without loading models")
    args = parser.parse_args()
    if args.report_only:
        print(json.dumps(report_from_saved_evaluation(), ensure_ascii=False), flush=True)
    else:
        with GPU_LOCK.acquire(timeout=0):
            main(args.hours)
