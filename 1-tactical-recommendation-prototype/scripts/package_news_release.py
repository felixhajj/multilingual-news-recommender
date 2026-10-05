"""Create a checksummed portable artifact bundle, excluding base models and caches."""
import argparse
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, ROOT, LEGACY_ADAPTER, digest, file_digest, read_json, write_json
from src.portfolio_reports import evidence_report
from src.news_selection import verify_selection_lock
from src.learned_linker import linker_artifact_identity
from src.phase3_configuration import selected_configuration


def legacy_release_gates():
    report = evidence_report()
    active = read_json(ARTIFACTS / "active_model.json", {})
    release = report["release"]
    locked, bound_active, phase3_evaluated = False, False, False
    try:
        lock = verify_selection_lock()
        locked = True
        completion = read_json(ARTIFACTS / "phase3/completion.json", {})
        phase3_evaluated = completion.get("selection_lock_hash") == lock["lock_hash"] and completion.get("release_eligible") is True
        for name, sha in completion.get("final_reports", {}).items():
            if file_digest(ARTIFACTS / name) != sha:
                phase3_evaluated = False
        selected = lock["selected_run"]
        linker = active.get("linker")
        bound_active = (selected is not None and active.get("run_id") == selected
                        and active.get("gate", {}).get("selection_lock_hash") == lock["lock_hash"]
                        and active.get("model") == lock["run_evidence"][selected]["model"]
                        and release.get("model") == active["model"]
                        and bool(linker) and (ARTIFACTS / linker).resolve().is_relative_to(ARTIFACTS.resolve())
                        and release.get("linker_version") == digest(linker_artifact_identity(ARTIFACTS / linker)))
    except (ValueError, OSError, KeyError, TypeError):
        pass
    return {"reviewed_extraction": report["review"]["human_reviewed"] == 30,
            "selection_lock_verified": locked, "phase3_evaluated": phase3_evaluated,
            "identity_bound_active_index": bound_active,
            "extraction_evaluated": report["extraction_evaluation"].get("status") == "evaluated",
            "retrieval_evaluated": report["recommendation_evaluation"].get("status") == "evaluated",
            "validated_active_adapter": active.get("gate", {}).get("passed") is True,
            "trained_linker_evaluated": report["linker"].get("status") == "evaluated",
            "release_articles": release.get("articles", 0) >= 300 and release.get("status") == "prepared",
            "index_matches_active_model": release.get("model", {}).get("run_id") == active.get("run_id"),
            "training_runs": all(read_json(ARTIFACTS / "runs" / name / "manifest.json", {}).get("status") in {"completed", "time_limited"}
                                 for name in ("extraction-only-v1", "domain-v1", "domain-extraction-v1"))}


def release_gates():
    release = read_json(ARTIFACTS / "release_manifest.json", {})
    completion = read_json(ARTIFACTS / "phase4/completion.json", {})
    gates = {"selected_configuration_verified": False, "selected_index_identity": False,
             "phase4_complete": completion.get("status") == "phase4_complete_with_model_failures_preserved",
             "release_articles": release.get("articles") == 300 and release.get("status") == "prepared",
             "database_verified": False, "notebooks_current": False}
    try:
        import sqlite3
        selected_configuration.cache_clear()
        selected = selected_configuration()
        gates["selected_configuration_verified"] = selected is not None
        gates["selected_index_identity"] = bool(selected and release.get("model") == selected["model"]
            and release.get("linker_version") == digest(linker_artifact_identity(selected["linker_path"]))
            and completion.get("pipeline_identity") == release.get("pipeline_identity"))
        database = ARTIFACTS / "release.sqlite"
        with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as db:
            count = db.execute("SELECT count(*) FROM articles WHERE pipeline_identity=?",
                               (release.get("pipeline_identity"),)).fetchone()[0]
        actual_hash = file_digest(database)
        original_hash = completion.get("release_index", {}).get("database_sha256")
        portable = read_json(ARTIFACTS / "runtime_database_manifest.json", {})
        gates["database_verified"] = count == 300 and (actual_hash == original_hash or (
            portable.get("source_sha256") == original_hash and portable.get("snapshot_sha256") == actual_hash
            and portable.get("pipeline_identity") == release.get("pipeline_identity")))
        from scripts.validate_news_notebooks import NOTEBOOKS, validation_fingerprint
        reports = read_json(ARTIFACTS / "notebook_validation/report.json", [])
        by_name = {row["notebook"]: row for row in reports}
        gates["notebooks_current"] = all(by_name.get(name, {}).get("status") == "passed"
            and by_name[name].get("input_fingerprint") == validation_fingerprint(name)
            and file_digest(ARTIFACTS / "notebook_validation" / name) == by_name[name].get("executed_sha256")
            for name in NOTEBOOKS)
    except (ValueError, OSError, KeyError, TypeError):
        pass
    return gates


def runtime_artifact_paths():
    lock = verify_selection_lock()
    completion = read_json(ARTIFACTS / "phase3/completion.json")
    names = {"phase3/selection_lock.json", "phase3/completion.json", "release_manifest.json",
             "processing_budget.json", "review/extraction.jsonl", "review/extraction_roles.json",
             "review/recommendation.jsonl", "phase4/completion.json"}
    names.update(lock["extra_evidence"])
    names.update(completion["final_reports"])
    for run_id in lock["run_evidence"]:
        names.update(f"evaluation/{run_id}/{name}" for name in ("manifest.json", "predictions.jsonl"))
        folder = ARTIFACTS / "runs" / run_id
        if (folder / "manifest.json").exists():
            names.add(f"runs/{run_id}/manifest.json")
            training = read_json(folder / "manifest.json")
            if training.get("dataset_artifact"):
                names.add(training["dataset_artifact"].replace("\\", "/"))
        names.update(path.relative_to(ARTIFACTS).as_posix() for path in (folder / "adapter").glob("*") if path.is_file())
    for folder in ("review", "phase4/selected_ranking", "phase3/linker-v3", "notebook_validation"):
        names.update(path.relative_to(ARTIFACTS).as_posix() for path in (ARTIFACTS / folder).rglob("*")
                     if path.is_file() and path.suffix not in {".sqlite", ".zip"})
    return {name for name in names if (ARTIFACTS / name).is_file()}


def package(preview=False, profile="runtime"):
    if profile not in {"runtime", "evidence"}:
        raise ValueError("Unknown bundle profile")
    gates = release_gates()
    if not preview and not all(gates.values()):
        raise ValueError(f"Final release gates remain open: {[name for name, passed in gates.items() if not passed]}")
    from src.news_store import NewsStore
    import sqlite3
    # SQLite backup captures a coherent WAL snapshot without copying a live database file.
    snapshot = ARTIFACTS / "package" / f"{profile}.sqlite"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with NewsStore(ARTIFACTS / "release.sqlite").connect() as source:
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
        finally:
            target.close()
    if profile == "runtime":
        with sqlite3.connect(snapshot) as target:
            target.execute("DELETE FROM articles WHERE pipeline_identity != ?",
                           (read_json(ARTIFACTS / "release_manifest.json")["pipeline_identity"],))
            target.execute("DELETE FROM analyses WHERE cache_key NOT IN (SELECT cache_key FROM articles)")
            target.execute("DROP TABLE IF EXISTS articles_legacy_v1")
            target.commit()
            target.execute("VACUUM")
        write_json(snapshot.with_suffix(".identity.json"), {
            "source_sha256": read_json(ARTIFACTS / "phase4/completion.json")["release_index"]["database_sha256"],
            "snapshot_sha256": file_digest(snapshot),
            "pipeline_identity": read_json(ARTIFACTS / "release_manifest.json")["pipeline_identity"]})
    files = {}
    for name in ("corpus_manifest.json", "active_model.json", "release_manifest.json",
                 "extraction_evaluation.json", "recommendation_evaluation.json", "processing_budget.json",
                 "extraction_training.jsonl", "training_preparation.json", "experiment_status.json"):
        if (ARTIFACTS / name).exists():
            files["output/portfolio/" + name] = ARTIFACTS / name
    paths = (ARTIFACTS / name for name in runtime_artifact_paths()) if profile == "runtime" else ARTIFACTS.rglob("*")
    for path in paths:
        relative = path.relative_to(ARTIFACTS)
        if (path.is_file() and relative.parts[0] != "package" and path.name != "release.sqlite"
                and not path.name.endswith((".tmp", ".lock", "-wal", "-shm"))):
            files["output/portfolio/" + relative.as_posix()] = path
    for folder in ("src", "scripts", "notebooks", "data/release", "deployment", "licenses", "docs", "tests"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".zip"}:
                files[path.relative_to(ROOT).as_posix()] = path
    for path in ROOT.glob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".toml"}:
            files[path.name] = path
    for path in LEGACY_ADAPTER.glob("*"):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / "licenses").glob("*"):
        files["licenses/" + path.name] = path
    files["output/portfolio/release.sqlite"] = snapshot
    if profile == "runtime":
        files["output/portfolio/runtime_database_manifest.json"] = snapshot.with_suffix(".identity.json")
    total = sum(path.stat().st_size for path in files.values())
    limit = 500 * 1024**2 if profile == "runtime" else 5 * 1024**3
    if total >= limit:
        raise ValueError(f"{profile} bundle exceeds its {limit} byte limit")
    manifest = {"kind": "research_preview" if preview else "validated_release", "profile": profile, "gates": gates,
                "uncompressed_bytes": total, "files": {name: file_digest(path) for name, path in sorted(files.items())}}
    manifest["version"] = digest(manifest)
    archive = ARTIFACTS / "package" / f"{profile}-{manifest['version'][:16]}.zip"
    import json
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as source, output.open(info, "w") as sink:
                import shutil
                shutil.copyfileobj(source, sink)
        info = zipfile.ZipInfo("bundle_manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        output.writestr(info, json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    write_json(archive.with_suffix(".manifest.json"), {**manifest, "archive_sha256": file_digest(archive)})
    print(f"{archive}: {archive.stat().st_size/1024**2:.1f} MiB; {manifest['kind']}")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Explicit unvalidated local preview, never accepted by final publisher")
    parser.add_argument("--profile", choices=("runtime", "evidence"), default="runtime")
    args = parser.parse_args()
    package(args.preview, args.profile)
