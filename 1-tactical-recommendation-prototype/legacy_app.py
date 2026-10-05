"""Historical manual-catalogue showcase; not the portfolio release entrypoint."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import mimetypes
import os
import sys

if Path("D:/hf-cache").exists():
    os.environ.setdefault("HF_HOME", "D:/hf-cache")

from src.data_loader import load_articles, load_entity_catalogue, load_incoming_articles, load_users
from src.demo_pipeline import build_pipeline_demo
from src.embeddings import embedding_map_for_user_article
from src.entity_enrichment import enrich_articles
from src.extraction_outputs import apply_extraction_outputs, load_extraction_outputs
from src.scoring import rank_articles_for_user


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DEFAULT_PORT = 8501
WALKTHROUGH_NOTEBOOK = ROOT / "notebooks" / "tactical_report_a_to_z_walkthrough.ipynb"
NOTEBOOK_STAGE_CELLS = {
    "01": [4],
    "02": [6],
    "03": [8, 10, 12],
    "04": [16],
    "05": [18],
    "06": [20, 22, 24],
    "07": [26, 28, 30],
}


def load_notebook_steps():
    """Return the exact walkthrough code shown by each prototype stage."""
    notebook = json.loads(WALKTHROUGH_NOTEBOOK.read_text(encoding="utf-8-sig"))
    cells = notebook["cells"]
    stages = {}

    for stage_id, cell_indexes in NOTEBOOK_STAGE_CELLS.items():
        stage_cells = []
        for cell_index in cell_indexes:
            cell = cells[cell_index]
            if cell.get("cell_type") != "code":
                raise ValueError(
                    f"Notebook cell {cell_index} mapped to stage {stage_id} is not code."
                )
            stage_cells.append(
                {
                    "cell_index": cell_index,
                    "execution_count": cell.get("execution_count"),
                    "source": "".join(cell.get("source", [])),
                }
            )
        stages[stage_id] = stage_cells

    return {"notebook": WALKTHROUGH_NOTEBOOK.name, "stages": stages}


def load_article_corpus():
    published = load_articles(ROOT / "data")
    return published + load_enriched_incoming_articles()


def load_enriched_incoming_articles():
    incoming = enrich_articles(
        load_incoming_articles(ROOT / "data"),
        load_entity_catalogue(ROOT / "data"),
    )
    extraction_path = ROOT / "output" / "extractions" / "incoming_article_filters.jsonl"
    return apply_extraction_outputs(incoming, load_extraction_outputs(extraction_path))


class PrototypeHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return

        if parsed.path == "/api/users":
            self._json_response(load_users(ROOT / "data"))
            return

        if parsed.path == "/api/articles":
            self._json_response(load_article_corpus())
            return

        if parsed.path == "/api/notebook-steps":
            self._json_response(load_notebook_steps())
            return

        if parsed.path == "/api/enrichment-preview":
            self._json_response(load_enriched_incoming_articles())
            return

        if parsed.path == "/api/recommendations":
            self._handle_recommendations(parsed.query)
            return

        if parsed.path == "/api/pipeline-demo":
            self._handle_pipeline_demo(parsed.query)
            return

        if parsed.path == "/api/embedding-map":
            self._handle_embedding_map(parsed.query)
            return

        requested = (STATIC_DIR / parsed.path.lstrip("/")).resolve()
        if STATIC_DIR in requested.parents and requested.exists() and requested.is_file():
            self._serve_file(requested)
            return

        self.send_error(404, "Not found")

    def _handle_recommendations(self, query):
        params = parse_qs(query)
        user_id = params.get("user_id", ["user_001"])[0]
        min_score = int(params.get("min_score", ["0"])[0])

        users = load_users(ROOT / "data")
        articles = load_article_corpus()
        user = next((item for item in users if item["user_id"] == user_id), users[0])

        try:
            ranked = rank_articles_for_user(user, articles)
        except RuntimeError as exc:
            self._json_response({"error": str(exc)}, status=503)
            return

        filtered = [item for item in ranked if item["score"] >= min_score]

        self._json_response(
            {
                "user": user,
                "recommendations": filtered,
                "total_articles": len(articles),
                "min_score": min_score,
            }
        )

    def _handle_embedding_map(self, query):
        params = parse_qs(query)
        user_id = params.get("user_id", ["user_001"])[0]
        article_id = params.get("article_id", [""])[0]

        users = load_users(ROOT / "data")
        articles = load_article_corpus()
        user = next((item for item in users if item["user_id"] == user_id), None)
        article = next((item for item in articles if item["article_id"] == article_id), None)

        if not user or not article:
            self._json_response({"error": "Unknown user_id or article_id."}, status=404)
            return

        try:
            points = embedding_map_for_user_article(user, article)
        except RuntimeError as exc:
            self._json_response({"error": str(exc)}, status=503)
            return

        self._json_response(
            {
                "user_id": user_id,
                "article_id": article_id,
                "points": points,
            }
        )

    def _handle_pipeline_demo(self, query):
        params = parse_qs(query)
        user_id = params.get("user_id", ["user_004"])[0]
        article_id = params.get("article_id", ["incoming_001"])[0]

        users = load_users(ROOT / "data")
        articles = load_article_corpus()
        user = next((item for item in users if item["user_id"] == user_id), None)
        article = next((item for item in articles if item["article_id"] == article_id), None)

        if not user or not article:
            self._json_response({"error": "Unknown user_id or article_id."}, status=404)
            return

        extraction_path = ROOT / "output" / "extractions" / "incoming_article_filters.jsonl"
        extraction_records = load_extraction_outputs(extraction_path)
        try:
            payload = build_pipeline_demo(ROOT, user, article, extraction_records, articles)
        except RuntimeError as exc:
            self._json_response({"error": str(exc)}, status=503)
            return

        self._json_response(payload)

    def _serve_file(self, path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json_response(self, payload, status=200):
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run(port=DEFAULT_PORT):
    server = ThreadingHTTPServer(("127.0.0.1", port), PrototypeHandler)
    print(f"Tactical Report recommendation prototype running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT)
