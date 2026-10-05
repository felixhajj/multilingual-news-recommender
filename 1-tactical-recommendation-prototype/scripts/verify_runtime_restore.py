"""Verify an isolated runtime using saved evidence only; no model downloads."""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.restore_news_bundle import restore
from src.portfolio_config import file_digest, write_json


def interpreter_path(python=None):
    # POSIX virtual-environment interpreters are often symlinks; keep that path.
    return str(Path(python).expanduser().absolute() if python else sys.executable)


def verify(archive, destination, python=None):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError("Verification requires a separate empty destination")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Verification destination must be empty")
    manifest = restore(archive, destination)
    program = """
import json, sqlite3
from src.phase3_configuration import selected_configuration
from src.news_selection import score_extraction
from scripts.evaluate_selected_ranking import report_from_saved_evaluation
from src.portfolio_config import ARTIFACTS, read_json, file_digest
protected = [ARTIFACTS / 'release.sqlite', ARTIFACTS / 'phase3/selection_lock.json',
             ARTIFACTS / 'phase3/completion.json', ARTIFACTS / 'phase3/component_decisions.json']
protected += [ARTIFACTS / name for name in read_json(ARTIFACTS / 'phase3/completion.json')['final_reports']]
protected += [ARTIFACTS / 'phase4/selected_ranking/report_v2.json',
              ARTIFACTS / 'phase4/selected_ranking/manifest.json']
before = {str(path): file_digest(path) for path in protected}
selected = selected_configuration()
assert selected['run_id'] == 'extraction-only-v3'
release = read_json(ARTIFACTS / 'release_manifest.json')
with sqlite3.connect(ARTIFACTS / 'release.sqlite') as db:
    total, versions = db.execute('select count(*), count(distinct pipeline_identity) from articles').fetchone()
assert total == 300 and versions == 1
extraction = score_extraction(selected['run_id'], 'test')
ranking = report_from_saved_evaluation()
selected_configuration.cache_clear()
assert selected_configuration()['run_id'] == selected['run_id']
assert before == {str(path): file_digest(path) for path in protected}, 'Replay modified protected evidence'
import importlib.metadata
print(json.dumps({'selected_run': selected['run_id'], 'articles': total,
                  'extraction': extraction['metrics'], 'ranking': ranking['averages'],
                  'protected_evidence_unchanged': True,
                  'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'filelock', 'scikit-learn')}}))
"""
    import os
    env = dict(os.environ)
    env.pop("NEWS_ARTIFACTS", None)
    env["PYTHONPATH"] = str(destination)
    env["HF_HUB_OFFLINE"] = "1"
    interpreter = interpreter_path(python)
    result = subprocess.run([interpreter, "-c", program], cwd=destination, env=env,
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    evidence = {"status": "passed", "archive_sha256": file_digest(archive),
                "version": manifest["version"], "destination": str(destination),
                "python": interpreter,
                "scope": ("Isolated source/artifacts and separate dependency environment; offline saved-metric replay, not fresh inference"
                          if python else "Isolated source/artifacts; shared Python dependencies; offline saved-metric replay, not fresh inference"),
                "results": json.loads(result.stdout)}
    write_json(ROOT / "output/portfolio/phase5/runtime_restore.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--python", type=Path, help="Interpreter from a clean requirements-evidence environment")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.destination, args.python), indent=2))
