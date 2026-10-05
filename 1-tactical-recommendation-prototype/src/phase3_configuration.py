"""Resolve the validation-locked Phase 3 choices for the shared application."""
from functools import lru_cache

from src.learned_linker import linker_artifact_identity
from src.news_selection import plain_id, verify_selection_lock
from src.portfolio_config import ARTIFACTS, file_digest, read_json


@lru_cache(maxsize=1)
def selected_configuration():
    completion = read_json(ARTIFACTS / "phase3/completion.json", {})
    decisions = completion.get("component_decisions", {})
    run_id = completion.get("selected_extractor")
    linker_relative = decisions.get("linker_selected")
    if (completion.get("status") != "evaluated" or completion.get("release_eligible") is not True
            or not run_id or not linker_relative or decisions.get("ranking_method") not in {"keyword", "e5", "hybrid"}):
        return None

    plain_id(run_id)
    lock = verify_selection_lock()
    locked_decisions = read_json(ARTIFACTS / "phase3/component_decisions.json", {})
    if decisions != locked_decisions:
        raise ValueError("Completion component choices differ from locked validation decisions")
    weight = decisions.get("semantic_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
        raise ValueError("Selected semantic weight must be between zero and one")
    if lock.get("lock_hash") != completion.get("selection_lock_hash") or lock.get("selected_run") != run_id:
        raise ValueError("Phase 3 completion does not match its verified selection lock")
    for name, expected in completion.get("final_reports", {}).items():
        path = ARTIFACTS / name
        if not path.resolve().is_relative_to(ARTIFACTS.resolve()) or file_digest(path) != expected:
            raise ValueError("A locked Phase 3 final report changed")

    evidence = lock["run_evidence"][run_id]
    model = evidence["model"]
    run_directory = ARTIFACTS / "runs" / run_id
    manifest = read_json(run_directory / "manifest.json", {})
    if manifest.get("status") != "completed" or manifest.get("optimizer_steps") != 48:
        raise ValueError("Selected extraction run is not a completed training run")
    adapter = run_directory / "adapter"
    if (file_digest(adapter / "adapter_model.safetensors") != model["adapter_sha256"]
            or file_digest(adapter / "adapter_config.json") != model["adapter_config_sha256"]):
        raise ValueError("Selected extraction adapter bytes changed")

    linker_path = (ARTIFACTS / linker_relative).resolve()
    if not linker_path.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Selected linker path escapes the artifacts directory")
    linker_identity = linker_artifact_identity(linker_path)
    if not linker_identity["training_identity_complete"]:
        raise ValueError("Selected linker is missing training provenance")

    return {"run_id": run_id, "adapter_path": adapter, "model": model,
            "schema_version": model["schema_version"], "max_new_tokens": model["max_new_tokens"],
            "chunk_tokens": model["chunk_tokens"], "chunk_stride": model["chunk_stride"],
            "linker_path": linker_path, "linker_identity": linker_identity,
            "ranking_method": decisions["ranking_method"],
            "semantic_weight": decisions["semantic_weight"],
            "selection_lock_hash": lock["lock_hash"]}
