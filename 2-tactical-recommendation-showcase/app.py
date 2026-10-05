from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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
