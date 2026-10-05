"""Execute the three walkthroughs without opting into GPU generation or training."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat
from nbclient import NotebookClient
from src.portfolio_config import ROOT, ARTIFACTS, digest, file_digest, read_json, write_json

NOTEBOOKS = ("tactical_report_a_to_z_walkthrough.ipynb",
             "tactical_report_entity_extraction_qlora.ipynb",
             "tactical_report_recommendation_demo.ipynb")
RUNTIME_FILES = ("portfolio_app.py", "src/news_pipeline.py", "src/news_store.py",
                 "src/news_evaluation.py", "src/news_selection.py", "src/llm_extractor.py",
                 "src/learned_linker.py", "src/embeddings.py", "src/phase3_configuration.py")


def validation_fingerprint(name):
    release = read_json(ARTIFACTS / "release_manifest.json", {})
    return digest({"notebook_sha256": file_digest(ROOT / "notebooks" / name),
                   "pipeline_identity": release.get("pipeline_identity"),
                   "runtime_sources": {path: file_digest(ROOT / path) for path in RUNTIME_FILES},
                   "selection_lock": read_json(ARTIFACTS / "phase3" / "selection_lock.json", {}).get("lock_hash")})


if __name__ == "__main__":
    report_path = ARTIFACTS / "notebook_validation" / "report.json"
    report = read_json(report_path, [])
    if not isinstance(report, list):
        raise ValueError("Notebook validation report has an unexpected format")
    report_by_name = {row.get("notebook"): row for row in report}
    report_by_name = {name: row for name, row in report_by_name.items() if name in NOTEBOOKS}
    for name in NOTEBOOKS:
        destination = ARTIFACTS / "notebook_validation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = validation_fingerprint(name)
        previous = report_by_name.get(name, {})
        if (previous.get("status") == "passed" and previous.get("input_fingerprint") == fingerprint
                and destination.exists() and file_digest(destination) == previous.get("executed_sha256")):
            print({"notebook": name, "status": "passed", "resumed": "verified existing execution"}, flush=True)
            continue
        started = time.monotonic()
        notebook = None
        try:
            notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
            nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    assert index > 0 and notebook.cells[index-1].cell_type == "markdown"
                    assert cell.metadata.get("stage_id")
            all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            if name == "tactical_report_a_to_z_walkthrough.ipynb" and "RUN_LIVE = False" not in all_code:
                raise ValueError("A-to-Z notebook must keep live generation opt-in")
            if name == "tactical_report_entity_extraction_qlora.ipynb" and "RUN_TRAINING = False" not in all_code:
                raise ValueError("Training notebook must keep training opt-in")
            NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
            status = "passed"
            error = None
        except Exception as exc:
            status = "failed"
            error = str(exc)[-2000:]
        if notebook is not None:
            nbformat.write(notebook, destination)
        result = {"notebook": name, "status": status, "input_fingerprint": fingerprint,
                  "executed_sha256": file_digest(destination) if destination.exists() else None,
                  "elapsed_seconds": round(time.monotonic()-started, 3),
                  "scope": "Default cells; opt-in live generation and training were not run"}
        if error:
            result["error"] = error
        report_by_name[name] = result
        report = [report_by_name[key] for key in NOTEBOOKS if key in report_by_name]
        write_json(report_path, report)
        print(result, flush=True)
    if any(row["status"] != "passed" for row in report):
        raise SystemExit(1)
