"""Run real inference and indexing, then export fast, verifiable example results."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.news_pipeline import NewsPipeline
from src.portfolio_config import ARTIFACTS, ROOT, GPU_LOCK, PIPELINE_VERSION, digest, read_json, read_jsonl, write_json, write_jsonl
from src.portfolio_reports import publish_evidence_snapshot
from src.phase3_resources import check_storage
from src.release_processing import (FINALIZATION_RESERVE_SECONDS, collection_progress,
                                    recover_interrupted_run, remaining_allowance, retryable_failure)


def export(pipeline, with_recommendation=True):
    rows, _ = pipeline.store.collection(pipeline.embedding_version, pipeline.identity)
    # Keep the explicitly historical demo intact until at least one article has
    # been genuinely processed with the selected Phase 3 configuration.
    if rows:
        write_json(ROOT / "data" / "release" / "example_analyses.json", rows[:3])
        if with_recommendation:
            result = pipeline.recommend("Iran nuclear diplomacy and international sanctions")
            if result["results"]:
                write_json(ROOT / "data" / "release" / "example_ranking.json", result)
        publish_evidence_snapshot()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        export(NewsPipeline())
        return
    if args.limit <= 0:
        raise ValueError("Article limit must be positive")
    check_storage(ARTIFACTS)
    records = read_jsonl(ARTIFACTS / "corpus.jsonl")
    frozen = read_json(ARTIFACTS / "review" / "frozen_extraction_manifest.json", {}).get("items", {})
    candidates = [r for r in records if r["split"] != "train" and r["article_id"] not in frozen and
                  250 <= len(r["body"]) <= 2800 and r["domain_hits"] > 0]
    candidates.sort(key=lambda r: (-r["domain_hits"], digest(r["article_id"])))
    language_groups = [[r for r in candidates if r["language"] == language] for language in ("ar", "en")]
    ordered = [r for pair in zip(*language_groups) for r in pair]
    remaining = [r for r in candidates if r not in ordered]
    prior_release = read_json(ARTIFACTS / "release_manifest.json", {})
    budget_path = ARTIFACTS / "processing_budget.json"
    budget = read_json(budget_path, {"consumed_seconds": 0, "limit_seconds": 86400})
    recover_interrupted_run(budget, prior_release, budget_path)
    allowance = remaining_allowance(budget, args.hours * 3600)
    if allowance <= FINALIZATION_RESERVE_SECONDS:
        raise RuntimeError("Insufficient indexing allocation, including finalization reserve")
    completion_id = budget.get("completion_budget", {}).get("allocation_id")
    started = time.monotonic()
    pipeline = NewsPipeline()
    deadline = started + allowance
    pipeline.backend.processing_deadline = deadline - FINALIZATION_RESERVE_SECONDS
    old_run = prior_release.get("model", {}).get("run_id")
    legacy_failures = (prior_release.get("failed_attempts", 0)
                       if old_run != pipeline.backend.identity.get("run_id")
                       else prior_release.get("legacy_failed_attempts", 0))
    failure_path = ARTIFACTS / "phase4" / f"release_failures-{pipeline.identity[:12]}.jsonl"
    failures = read_jsonl(failure_path) if failure_path.exists() else []
    nonretryable_failures = {row["article_id"] for row in failures
                             if row.get("pipeline_identity") == pipeline.identity
                             and not retryable_failure(row)}
    fatal_error = None
    stop_reason = "candidate_queue_exhausted"
    rows, _ = pipeline.store.collection(pipeline.embedding_version, pipeline.identity)
    completed_ids = {row["article"]["article_id"] for row in rows}
    print(f"Resuming {len(completed_ids)}/{args.limit} existing articles; allowance {allowance/3600:.2f} hours", flush=True)

    def write_progress(in_flight_article_id=None):
        rows, _ = pipeline.store.collection(pipeline.embedding_version, pipeline.identity)
        write_json(ARTIFACTS / "release_manifest.json", {"status": "preparing", **collection_progress(rows),
            "target": args.limit, "legacy_failed_attempts": legacy_failures,
            "failed_attempts": len(failures), "model": pipeline.backend.identity,
            "pipeline_identity": pipeline.identity, "linker_version": pipeline.linker.version,
            "embedding_version": pipeline.embedding_version, "pipeline_version": PIPELINE_VERSION,
            "elapsed_seconds": time.monotonic()-started,
            "in_flight_article_id": in_flight_article_id,
            "in_flight_reserve_seconds": 300 if in_flight_article_id else 0,
            "completion_allocation_id": completion_id,
            "session_allowance_seconds": allowance,
            "failure_log": str(failure_path.relative_to(ARTIFACTS))})

    try:
        write_progress()
        for article in ordered + remaining:
            if article["article_id"] in nonretryable_failures or article["article_id"] in completed_ids:
                continue
            rows, _ = pipeline.store.collection(pipeline.embedding_version, pipeline.identity)
            if len(rows) >= args.limit:
                stop_reason = "target_reached"
                break
            if time.monotonic() + 300 >= pipeline.backend.processing_deadline:
                stop_reason = "index_allocation_limit"
                break
            check_storage(ARTIFACTS)
            write_progress(article["article_id"])
            try:
                analysis = pipeline.analyze_article(article, persist=True)
                completed_ids.add(article["article_id"])
                print(f"Processed {article['article_id']}: {len(analysis['evidence'])} grounded mentions", flush=True)
            except (RuntimeError, ValueError, OSError, MemoryError) as exc:
                failures.append({"article_id": article["article_id"],
                                 "pipeline_identity": pipeline.identity,
                                 "error_type": type(exc).__name__, "error": str(exc),
                                 "content_hash": digest(article["title"] + "\n\n" + article["body"]),
                                 "raw_outputs": getattr(exc, "raw_outputs", [getattr(exc, "raw_output", "")]),
                                 "chunk_metadata": getattr(exc, "chunk_metadata", []),
                                 "retryable": isinstance(exc, (OSError, MemoryError)) or "out of memory" in str(exc).casefold()})
                write_jsonl(failure_path, failures)
                if isinstance(exc, (OSError, MemoryError)) or "out of memory" in str(exc).casefold():
                    fatal_error = exc
                    stop_reason = "processing_deadline" if isinstance(exc, TimeoutError) else "resource_failure"
                    print(f"Stopping after a resource failure: {type(exc).__name__}: {exc}", flush=True)
                    break
            rows, _ = pipeline.store.collection(pipeline.embedding_version, pipeline.identity)
            write_progress()
            if (len(rows) <= 3 or len(rows) % 10 == 0) and time.monotonic() + 30 < pipeline.backend.processing_deadline:
                export(pipeline)
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        fatal_error = exc
        stop_reason = "worker_failure"
        print(f"Stopping safely: {type(exc).__name__}: {exc}", flush=True)
    finally:
        export_error = None
        try:
            export(pipeline, with_recommendation=fatal_error is None and time.monotonic() + 30 < deadline)
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            export_error = str(exc)
            print(f"Example export needs attention: {type(exc).__name__}: {exc}", flush=True)
        elapsed = time.monotonic()-started
        budget["consumed_seconds"] += elapsed
        budget.setdefault("allocations", []).append({
            "phase": 4, "operation": "selected_release_index_rebuild",
            "pipeline_identity": pipeline.identity, "target": args.limit,
            "elapsed_seconds": elapsed, "failed_attempts": len(failures),
            "completion_allocation_id": completion_id, "budget_section": "index",
            "includes_final_export": True,
        })
        write_json(budget_path, budget)
        write_jsonl(failure_path, failures)
        rows, _ = pipeline.store.collection(pipeline.embedding_version, pipeline.identity)
        write_json(ARTIFACTS / "release_manifest.json", {"status": "prepared" if len(rows) >= args.limit else "partial",
            **collection_progress(rows), "target": args.limit, "failed_attempts": len(failures),
            "legacy_failed_attempts": legacy_failures,
            "pipeline_identity": pipeline.identity, "model": pipeline.backend.identity,
            "linker_version": pipeline.linker.version, "embedding_version": pipeline.embedding_version, "pipeline_version": PIPELINE_VERSION,
            "elapsed_seconds": elapsed, "failure_log": str(failure_path.relative_to(ARTIFACTS)),
            "completion_allocation_id": completion_id, "stop_reason": stop_reason,
            "export_error": export_error})
        publish_evidence_snapshot()
    if fatal_error:
        raise RuntimeError("Selected-model indexing stopped after a resource failure; the partial index is resumable") from fatal_error


if __name__ == "__main__":
    with GPU_LOCK:
        main()
