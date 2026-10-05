"""Resumable release processing limits without resetting earlier charges."""
from collections import Counter

from src.portfolio_config import digest, write_json

FINALIZATION_RESERVE_SECONDS = 60


def retryable_failure(row):
    return (row.get("retryable") is True
            or row.get("error_type") in {"OSError", "MemoryError", "TimeoutError"}
            or "out of memory" in row.get("error", "").casefold())


def remaining_allowance(budget, requested_seconds):
    if requested_seconds <= 0:
        raise ValueError("Processing hours must be positive")
    remaining = max(0, budget["limit_seconds"] - budget["consumed_seconds"])
    completion = budget.get("completion_budget", {})
    if completion:
        spent = sum(row.get("elapsed_seconds", 0) for row in budget.get("allocations", [])
                    if row.get("completion_allocation_id") == completion["allocation_id"]
                    and row.get("budget_section") == "index")
        remaining = min(remaining, max(0, completion["sections"]["index"] - spent))
    return min(requested_seconds, remaining)


def recover_interrupted_run(budget, manifest, budget_path):
    if manifest.get("status") != "preparing":
        return
    recovery_id = digest(manifest)
    if recovery_id in budget.get("recovered_release_runs", []):
        return
    elapsed = manifest.get("elapsed_seconds", 0)
    if manifest.get("in_flight_article_id"):
        elapsed += manifest.get("in_flight_reserve_seconds", 300)
    budget["consumed_seconds"] += elapsed
    budget.setdefault("recovered_release_runs", []).append(recovery_id)
    budget["recovery_note"] = "Recovered interrupted release indexing from its last durable checkpoint"
    budget.setdefault("allocations", []).append({
        "phase": 4, "operation": "selected_release_index_recovery",
        "completion_allocation_id": manifest.get("completion_allocation_id"),
        "budget_section": "index", "elapsed_seconds": elapsed,
        "recovery_id": recovery_id,
    })
    write_json(budget_path, budget)


def collection_progress(rows):
    return {"articles": len(rows),
            "language_counts": dict(Counter(row["article"].get("language", "unknown") for row in rows))}
