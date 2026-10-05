"""Record public API integration checks, separate from the frozen quality evaluation."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.portfolio_config import ARTIFACTS, read_json, write_json

SPACE = "felixhajj/multilingual-news-recommender"


def verify(edited_inference=False):
    from gradio_client import Client
    client = Client(SPACE, verbose=False, analytics_enabled=False, download_files=False,
                    httpx_kwargs={"timeout": 60})
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "space": SPACE, "scope": "Public integration smoke tests, not extraction quality or ranking evaluation",
              "checks": {}, "responses": {}}
    output = ARTIFACTS / "phase5/public_api_checks.json"

    def call(name, endpoint, *args):
        started = time.monotonic()
        response = client.submit(*args, api_name=endpoint).result(timeout=240)
        report["responses"][name] = {"elapsed_seconds": time.monotonic() - started, "result": response}
        write_json(output, report)
        return response

    try:
        blank = ("",) * 7
        nuclear = call("nuclear_interest", "/recommend", "Iran nuclear diplomacy and sanctions", *blank)[1]
        french = call("education_interest", "/recommend", "French and Lebanese education cooperation", *blank)[1]
        assert nuclear["status"] == french["status"] == "ready"
        assert nuclear["total_articles"] == french["total_articles"] == 300
        nuclear_ids = [row["article"]["article_id"] for row in nuclear["results"]]
        french_ids = [row["article"]["article_id"] for row in french["results"]]
        assert nuclear_ids != french_ids
        report["checks"]["changed_interest_changes_ranking"] = True
        report["checks"]["current_300_article_index"] = True
        first = french["results"][0]
        explanation = call("selected_explanation", "/explain_selected", first["cache_key"],
                           "French and Lebanese education cooperation", *blank)[1]
        assert explanation["e5"]["dimensions"] == 768 and explanation["qwen"]["generated_json"]
        report["checks"]["selected_result_explanation"] = True
        empty = call("empty_interest", "/recommend", "", *blank)[1]
        assert empty["status"] == "unavailable" and empty.get("error")
        report["checks"]["empty_interest_visible"] = True
        evidence = call("evidence", "/evidence_report")
        assert (isinstance(evidence, dict) and evidence.get("release", {}).get("articles") == 300
                and evidence.get("review", {}).get("human_reviewed") == 30)
        report["checks"]["evidence_endpoint_available"] = True
        if edited_inference:
            original = read_json(ARTIFACTS / "phase5/hosted-english.json")
            article = original["article"]
            response = call("edited_article", "/analyze", article["title"],
                            article["body"] + " The discussion also concerned a school in Paris.",
                            "French and Lebanese education cooperation", *blank)
            trace = response[4]
            assert trace["status"] == "ready", trace
            assert trace["cache_key"] != original["cache_key"] and trace["cache_hit"] is False
            assert trace["model"]["adapter_sha256"] == original["model"]["adapter_sha256"]
            report["checks"]["edited_text_new_inference_identity"] = True
        assert all(report["checks"].values()), "A mandatory hosted check failed"
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output, report)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited-inference", action="store_true")
    verify(parser.parse_args().edited_inference)
