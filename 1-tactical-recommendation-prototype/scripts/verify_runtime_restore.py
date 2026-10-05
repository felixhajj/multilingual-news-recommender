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


def verify(archive, destination):
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
from src.portfolio_config import ARTIFACTS, read_json
selected = selected_configuration()
assert selected['run_id'] == 'extraction-only-v3'
release = read_json(ARTIFACTS / 'release_manifest.json')
with sqlite3.connect(ARTIFACTS / 'release.sqlite') as db:
    total, versions = db.execute('select count(*), count(distinct pipeline_identity) from articles').fetchone()
assert total == 300 and versions == 1
extraction = score_extraction(selected['run_id'], 'test')
ranking = report_from_saved_evaluation()
print(json.dumps({'selected_run': selected['run_id'], 'articles': total,
                  'extraction': extraction['metrics'], 'ranking': ranking['averages']}))
"""
    import os
    env = dict(os.environ)
    env.pop("NEWS_ARTIFACTS", None)
    env["PYTHONPATH"] = str(destination)
    env["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run([sys.executable, "-c", program], cwd=destination, env=env,
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    evidence = {"status": "passed", "archive_sha256": file_digest(archive),
                "version": manifest["version"], "destination": str(destination),
                "scope": "Isolated source/artifacts; shared Python dependencies; offline saved-metric replay, not fresh inference",
                "results": json.loads(result.stdout)}
    write_json(ROOT / "output/portfolio/phase5/runtime_restore.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.destination), indent=2))
