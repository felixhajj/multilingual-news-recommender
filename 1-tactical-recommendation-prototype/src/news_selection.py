"""Validation reports and locked final-test reporting with file-bound evidence."""
from datetime import datetime, timezone
from pathlib import Path

from src.extraction_data import EXTRACTION_SCHEMA_V2, EXTRACTION_SCHEMA_V3
from src.news_evaluation import extraction_metrics, review_roles, validate_review
from src.portfolio_config import (ARTIFACTS, BASE_REVISION, LEGACY_ADAPTER, digest,
                                  file_digest, read_json, read_jsonl, write_json)

EXTRACTION_GATE = {
    "schema_floor": 0.8,
    "rule": "v3 schema >= max(0.8, base), entity micro-F1 strictly above base, neither language F1 below base",
    "required_schema": EXTRACTION_SCHEMA_V3,
}


def plain_id(value):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("Expected a plain immutable run ID")
    return value


def extraction_inputs(run_id):
    plain_id(run_id)
    references = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    if validate_review(references) != 30:
        raise ValueError("Extraction review is incomplete")
    roles = review_roles(references)
    folder = ARTIFACTS / "evaluation" / run_id
    manifest = read_json(folder / "manifest.json")
    if not manifest:
        raise ValueError(f"No prediction manifest for {run_id}")
    schema = manifest["model"].get("schema_version", EXTRACTION_SCHEMA_V2)
    expected_ids = set(manifest["frozen_ids"])
    if expected_ids != set(roles) or any(manifest["article_hashes"].get(r["article_id"]) != r["content_hash"] for r in references):
        raise ValueError("Prediction manifest disagrees with frozen inputs")
    predictions = read_jsonl(folder / "predictions.jsonl")
    ids = [p["article_id"] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError(f"Incomplete or duplicated predictions for {run_id}")
    evidence = {
        "references": file_digest(ARTIFACTS / "review/extraction.jsonl"),
        "roles": file_digest(ARTIFACTS / "review/extraction_roles.json"),
        "manifest": file_digest(folder / "manifest.json"),
        "predictions": file_digest(folder / "predictions.jsonl"),
        "model": manifest["model"],
    }
    if manifest["model"].get("adapter_enabled"):
        adapter = ARTIFACTS / "runs" / run_id / "adapter"
        for field, name in (("adapter_sha256", "adapter_model.safetensors"),
                            ("adapter_config_sha256", "adapter_config.json")):
            if file_digest(adapter / name) != manifest["model"][field]:
                raise ValueError("Evaluated adapter files changed")
        evidence["training_manifest"] = file_digest(ARTIFACTS / "runs" / run_id / "manifest.json")
    if schema == EXTRACTION_SCHEMA_V3:
        model = manifest["model"]
        if model.get("revision") != BASE_REVISION or model.get("tokenizer_revision") != BASE_REVISION:
            raise ValueError("v3 base and tokenizer revisions must be pinned")
        expected_inputs = {r["article_id"]: digest(r["title"] + "\n\n" + r["text"]) for r in references}
        if manifest.get("full_input_hashes") != expected_inputs:
            raise ValueError("v3 title/body prediction inputs changed")
        adapter = ARTIFACTS / "runs" / run_id / "adapter" if model.get("adapter_enabled") else LEGACY_ADAPTER
        if not model.get("tokenizer_files"):
            raise ValueError("Missing tokenizer identity")
        for name, checksum in model["tokenizer_files"].items():
            if Path(name).name != name or file_digest(adapter / name) != checksum:
                raise ValueError("Evaluated tokenizer changed")
        if model.get("adapter_enabled"):
            training = read_json(ARTIFACTS / "runs" / run_id / "manifest.json")
            if training["status"] != "completed" or not training.get("changed_tensors"):
                raise ValueError("Incomplete v3 optimization cannot be a deployment candidate")
            identity = training["identity"]
            for key, value in (("base_revision", model["revision"]), ("tokenizer_revision", model["tokenizer_revision"]),
                               ("schema_version", schema), ("prompt_sha256", model["prompt_sha256"])):
                if identity.get(key) != value:
                    raise ValueError("Training and evaluated configuration disagree")
            # Old manifests use Windows separators; keep their bytes frozen and
            # resolve the same relative artifact on Linux-hosted releases.
            dataset = ARTIFACTS / training["dataset_artifact"].replace("\\", "/")
            if dataset.resolve().is_relative_to(ARTIFACTS.resolve()) is False:
                raise ValueError("Dataset path outside artifact directory")
            if digest(read_jsonl(dataset)) != identity["dataset_hash"]:
                raise ValueError("Training dataset changed")
            evidence["dataset_file"] = file_digest(dataset)
            if training["tokenizer_files"] != model["tokenizer_files"]:
                raise ValueError("Training tokenizer changed before evaluation")
    return references, roles, predictions, schema, evidence


def metrics_by_language(predictions, references, schema):
    result = extraction_metrics(predictions, references, schema)
    result["by_language"] = {}
    for language in ("en", "ar"):
        gold = [r for r in references if r["language"] == language]
        ids = {r["article_id"] for r in gold}
        result["by_language"][language] = extraction_metrics([p for p in predictions if p["article_id"] in ids], gold, schema)
    return result


def score_extraction(run_id, role="validation"):
    if role not in {"validation", "test"}:
        raise ValueError("Choose validation or test")
    references, roles, predictions, schema, evidence = extraction_inputs(run_id)
    if role == "test":
        lock = verify_selection_lock()
        if run_id not in lock["run_evidence"]:
            raise ValueError("Run was not included in the fixed selection decision")
    gold = [r for r in references if roles[r["article_id"]] == role]
    ids = {r["article_id"] for r in gold}
    subset = [p for p in predictions if p["article_id"] in ids]
    report = {
        "run_id": run_id, "role": role, "evidence": evidence,
        "metrics": metrics_by_language(subset, gold, schema),
        "schema_version": schema, "deployment_eligible_schema": schema == EXTRACTION_SCHEMA_V3,
        "scope": "Held out from our training; small human-reviewed sample, exact entity surface matching",
    }
    if schema == EXTRACTION_SCHEMA_V2:
        report["v3_metrics"] = metrics_by_language(subset, gold, EXTRACTION_SCHEMA_V3)
        report["limitation"] = "Historical v2 scores exclude locations. Strict v3 schema validity is separate; this run cannot be promoted as location-aware."
    path = ARTIFACTS / "phase3/reports" / f"{run_id}_{role}.json"
    existing = read_json(path)
    if existing and existing != report:
        raise ValueError("Report evidence changed; use a new evaluation run ID")
    if existing:
        # Sealed byte hashes include host line endings; replay must be read-only.
        return existing
    write_json(path, report)
    return report


def passes_gate(candidate, baseline):
    if candidate["schema_version"] != EXTRACTION_SCHEMA_V3 or baseline["schema_version"] != EXTRACTION_SCHEMA_V3:
        return False
    c, b = candidate["metrics"], baseline["metrics"]
    return (c["schema_validity"] >= max(EXTRACTION_GATE["schema_floor"], b["schema_validity"])
            and c["micro_f1"] > b["micro_f1"]
            and all(c["by_language"][lang]["micro_f1"] >= b["by_language"][lang]["micro_f1"] for lang in ("en", "ar")))


def lock_selection(base_id, candidate_ids, extra_evidence=None):
    if not candidate_ids:
        raise ValueError("Evaluate at least one candidate before locking selection")
    path = ARTIFACTS / "phase3/selection_lock.json"
    if path.exists():
        existing = verify_selection_lock()
        if (existing["base_id"] != base_id or existing["candidate_ids"] != list(candidate_ids)
                or existing["extra_evidence"] != (extra_evidence or {})):
            raise ValueError("Selection already locked for different runs")
        return existing
    baseline = score_extraction(base_id)
    candidates = {run: score_extraction(run) for run in candidate_ids}
    eligible = [run for run, report in candidates.items() if passes_gate(report, baseline)]
    selected = max(eligible, key=lambda run: (candidates[run]["metrics"]["micro_f1"], run)) if eligible else None
    run_evidence = {base_id: baseline["evidence"], **{run: report["evidence"] for run, report in candidates.items()}}
    payload = {
        "version": 1, "locked_at": datetime.now(timezone.utc).isoformat(), "gate": EXTRACTION_GATE,
        "base_id": base_id, "candidate_ids": list(candidate_ids), "selected_run": selected,
        "decision": "promote_candidate" if selected else "no_candidate_passed_research_preview",
        "run_evidence": run_evidence, "extra_evidence": extra_evidence or {},
        "policy": "Validation decisions fixed before final-test metrics; failed gates remain unmet release requirements.",
    }
    payload["lock_hash"] = digest(payload)
    write_json(path, payload)
    return verify_selection_lock()


def verify_selection_lock():
    lock = read_json(ARTIFACTS / "phase3/selection_lock.json")
    if not lock:
        raise ValueError("Final-test scoring requires a selection lock")
    if digest({k: v for k, v in lock.items() if k != "lock_hash"}) != lock["lock_hash"]:
        raise ValueError("Selection lock changed")
    for run_id, expected in lock["run_evidence"].items():
        if extraction_inputs(run_id)[-1] != expected:
            raise ValueError("Evidence changed after selection was locked")
    for name, checksum in lock["extra_evidence"].items():
        path = ARTIFACTS / name
        if not path.resolve().is_relative_to(ARTIFACTS.resolve()):
            raise ValueError("Linked evidence path outside artifact directory")
        if file_digest(path) != checksum:
            raise ValueError("Linked selection evidence changed")
    return lock
