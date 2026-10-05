"""Build a lightweight, shareable copy of the Tactical Report prototype.

Run this file from the ORIGINAL full project environment, where the E5 model,
Qwen tokenizer/adapter artifacts, and project dependencies are already available.
It precomputes every user/article result and creates a sibling folder that no
longer loads any ML model at runtime.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from app import load_article_corpus, load_enriched_incoming_articles, load_notebook_steps
from src.data_loader import load_users
from src.demo_pipeline import build_pipeline_demo
from src.extraction_outputs import load_extraction_outputs
from src.scoring import rank_articles_for_user


ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / "tactical-recommendation-showcase"
DATA_DIR = DEST / "data"
STATIC_DIR = DEST / "static"
EXTRACTION_PATH = ROOT / "output" / "extractions" / "incoming_article_filters.jsonl"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def lightweight_server_source() -> str:
    return r'''from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import mimetypes
import os
import sys

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA = json.loads((ROOT / "data" / "showcase_data.json").read_text(encoding="utf-8"))


class ShowcaseHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stdout.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), format % args)
        )

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            return self._serve_file(STATIC_DIR / "index.html")
        if parsed.path == "/api/users":
            return self._json_response(DATA["users"])
        if parsed.path == "/api/articles":
            return self._json_response(DATA["articles"])
        if parsed.path == "/api/notebook-steps":
            return self._json_response(DATA["notebook_steps"])
        if parsed.path == "/api/enrichment-preview":
            return self._json_response(DATA["enriched_incoming_articles"])
        if parsed.path == "/api/recommendations":
            return self._recommendations(parsed.query)
        if parsed.path == "/api/pipeline-demo":
            return self._pipeline(parsed.query)

        requested = (STATIC_DIR / parsed.path.lstrip("/")).resolve()
        if STATIC_DIR in requested.parents and requested.exists() and requested.is_file():
            return self._serve_file(requested)
        self.send_error(404, "Not found")

    def _recommendations(self, query):
        params = parse_qs(query)
        user_id = params.get("user_id", ["user_001"])[0]
        payload = DATA["rankings"].get(user_id)
        if payload is None:
            return self._json_response({"error": "Unknown user_id."}, status=404)
        return self._json_response(payload)

    def _pipeline(self, query):
        params = parse_qs(query)
        user_id = params.get("user_id", ["user_004"])[0]
        article_id = params.get("article_id", ["incoming_001"])[0]
        payload = DATA["pipelines"].get(f"{user_id}|{article_id}")
        if payload is None:
            return self._json_response(
                {"error": "This user/article combination was not exported."},
                status=404,
            )
        return self._json_response(payload)

    def _serve_file(self, path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json_response(self, payload, status=200):
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8501"))
    server = ThreadingHTTPServer((host, port), ShowcaseHandler)
    print(f"Tactical Report showcase running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
'''


def main() -> None:
    print("Preparing lightweight showcase...")
    if not (ROOT / "app.py").exists() or not (ROOT / "src").exists():
        raise SystemExit(
            "Place this script in the original tactical-recommendation-prototype folder first."
        )

    if DEST.exists():
        shutil.rmtree(DEST)
    DATA_DIR.mkdir(parents=True)
    shutil.copytree(ROOT / "static", STATIC_DIR)

    users = load_users(ROOT / "data")
    articles = load_article_corpus()
    extraction_records = load_extraction_outputs(EXTRACTION_PATH)

    pipelines: dict[str, object] = {}
    total = len(users) * len(articles)
    completed = 0
    for user in users:
        for article in articles:
            completed += 1
            key = f"{user['user_id']}|{article['article_id']}"
            print(f"[{completed}/{total}] Exporting {key}")
            pipelines[key] = build_pipeline_demo(
                ROOT,
                user,
                article,
                extraction_records,
                articles,
            )

    rankings = {}
    for user in users:
        print(f"Exporting ranking for {user['user_id']}")
        ranked = rank_articles_for_user(user, articles)
        rankings[user["user_id"]] = {
            "user": user,
            "recommendations": ranked,
            "total_articles": len(articles),
            "min_score": 0,
        }

    showcase_data = {
        "build_note": (
            "Presentation build: model outputs were generated previously by the complete "
            "local pipeline and saved for lightweight deployment."
        ),
        "users": users,
        "articles": articles,
        "notebook_steps": load_notebook_steps(),
        "enriched_incoming_articles": load_enriched_incoming_articles(),
        "pipelines": pipelines,
        "rankings": rankings,
    }
    write_json(DATA_DIR / "showcase_data.json", showcase_data)

    (DEST / "app.py").write_text(lightweight_server_source(), encoding="utf-8")
    (DEST / "requirements.txt").write_text(
        "# No third-party runtime packages are required.\n",
        encoding="utf-8",
    )
    (DEST / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n.venv/\n.env\n*.log\n.DS_Store\n",
        encoding="utf-8",
    )
    (DEST / "README.md").write_text(
        "# Tactical Report Recommendation Showcase\n\n"
        "A lightweight presentation build of the local learning prototype. The displayed "
        "tokenizer, QLoRA, extraction, multilingual E5, and recommendation outputs were "
        "precomputed by the complete local project and stored as JSON. No model is downloaded "
        "or loaded by this deployment.\n\n"
        "## Run locally\n\n"
        "```bash\npython app.py\n```\n\n"
        "Then open http://127.0.0.1:8501.\n",
        encoding="utf-8",
    )

    size_bytes = sum(path.stat().st_size for path in DEST.rglob("*") if path.is_file())
    print("\nShowcase created successfully:")
    print(DEST)
    print(f"Size: {size_bytes / 1024 / 1024:.2f} MB")
    print("Run it with:")
    print(f'  cd "{DEST}"')
    print("  python app.py")


if __name__ == "__main__":
    main()
