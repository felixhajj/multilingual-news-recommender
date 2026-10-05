"""Resumable, non-evaluative live checks for the selected Phase 4 application."""
import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEWS_EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from src.portfolio_config import ARTIFACTS, digest, file_digest, read_json, read_jsonl, write_json

PHASE = ARTIFACTS / "phase4"
OUT = PHASE / "live_app_check"
REPORT_PATH = OUT / "report.json"
PROGRESS_PATH = OUT / "progress.json"
ANALYSES_PATH = OUT / "analyses.jsonl"
FAILURES_PATH = OUT / "failures.jsonl"
ISOLATED_DB = OUT / "isolated-smoke.sqlite"
ALLOCATION_ID = "phase4-app-notebooks-20261001"
MAX_ARTICLES = 30
LANGUAGES = ("en", "ar")


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def record_id(row):
    return row.get("article_id") or (row.get("article") or {}).get("article_id")


def attempted_article_ids():
    found = set()
    paths = [ARTIFACTS / "release_failures.jsonl", *PHASE.glob("release_failures*.jsonl")]
    for path in paths:
        for row in rows(path):
            article_id = record_id(row)
            if article_id:
                found.add(article_id)
    return found


def sample_articles():
    corpus_path = ARTIFACTS / "corpus.jsonl"
    release_db = ARTIFACTS / "release.sqlite"
    if not corpus_path.is_file() or not release_db.is_file():
        raise RuntimeError("The prepared corpus or release index is missing")
    with sqlite3.connect(release_db) as db:
        indexed_ids = {row[0] for row in db.execute("SELECT DISTINCT article_id FROM articles")}

    roles = read_json(ARTIFACTS / "review" / "extraction_roles.json", {})
    protected_ids = set(roles)
    for path in (ARTIFACTS / "review" / "ranking.jsonl", ARTIFACTS / "extraction_training.jsonl",
                 ROOT / "data" / "extraction_examples.jsonl"):
        for row in rows(path):
            article_id = record_id(row)
            if article_id:
                protected_ids.add(article_id)

    blocked = indexed_ids | protected_ids | attempted_article_ids()
    candidates = {language: [] for language in LANGUAGES}
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            article = json.loads(line)
            body = article.get("body") or ""
            language = article.get("language")
            if (language in candidates and article.get("split") == "train"
                    and not article.get("synthetic", False)
                    and article.get("article_id") not in blocked
                    and 180 <= len(body) <= 6000):
                candidates[language].append(article)

    selected = []
    for language in LANGUAGES:
        ordered = sorted(candidates[language], key=lambda row: hashlib.sha256(
            ("phase4-live-smoke-v1:" + row["article_id"]).encode()).hexdigest())
        selected.extend(ordered[:MAX_ARTICLES // 2])
    if len(selected) != MAX_ARTICLES:
        counts = {key: len(value) for key, value in candidates.items()}
        raise RuntimeError(f"Need 15 eligible articles per language; candidates: {counts}")
    return selected, indexed_ids, protected_ids, len(candidates["en"]), len(candidates["ar"])


def append_result(path, row):
    append_jsonl(path, row)


def save_progress(progress, report):
    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(PROGRESS_PATH, progress)
    report["progress"] = progress
    report["updated_at"] = progress["updated_at"]
    write_json(REPORT_PATH, report)


def available_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def browser_executable():
    options = [Path(os.getenv("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
               Path(os.getenv("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
               Path(os.getenv("PROGRAMFILES", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe"]
    return next((path for path in options if path.is_file()), None)


def check_resource_failure(message):
    text = str(message).lower()
    return any(term in text for term in ("out of memory", "paging file", "error 1455",
                                         "no space left", "insufficient system resources"))


def distinct_interest_rankings_changed(recommendations):
    """Require two distinct interests and two complete, different top-five orders."""
    if len(recommendations) != 2:
        return False
    first, second = recommendations
    first_query = first.get("query", "").strip().casefold()
    second_query = second.get("query", "").strip().casefold()
    if not first_query or not second_query or first_query == second_query:
        return False
    first_ids = [row.get("article_id") for row in first.get("top_results", [])]
    second_ids = [row.get("article_id") for row in second.get("top_results", [])]
    return (len(first_ids) == 5 and len(set(first_ids)) == 5
            and len(second_ids) == 5 and len(set(second_ids)) == 5
            and first_ids != second_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Check inputs and resources without loading models")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    selected, indexed_ids, protected_ids, en_candidates, ar_candidates = sample_articles()
    chrome = browser_executable()
    if chrome is None:
        raise RuntimeError("Chrome or Edge is required for desktop/mobile browser checks")
    free = shutil.disk_usage(ROOT).free
    if free < 4 * 1024**3:
        raise RuntimeError(f"D: has only {free / 1024**3:.2f} GiB free; need at least 4 GiB")
    if args.preflight:
        print(json.dumps({"status": "preflight_passed", "eligible_articles": {"en": en_candidates, "ar": ar_candidates},
                          "selected": {language: sum(row["language"] == language for row in selected) for language in LANGUAGES},
                          "indexed_ids_excluded": len(indexed_ids), "review_and_training_ids_excluded": len(protected_ids),
                          "browser": str(chrome), "free_disk_gib": round(free / 1024**3, 2)}, ensure_ascii=False))
        return 0

    sample_identity = digest([{key: article.get(key) for key in ("article_id", "content_hash", "split", "language")}
                              for article in selected])
    progress = read_json(PROGRESS_PATH, {})
    from src.news_pipeline import NewsPipeline
    from src.news_store import NewsStore
    import portfolio_app

    pipeline = portfolio_app.get_pipeline()
    release_identity = pipeline.index_identity()
    release_manifest = read_json(ARTIFACTS / "release_manifest.json", {})
    release_records, _ = pipeline.store.collection(pipeline.embedding_version, release_identity)
    if release_identity != release_manifest.get("pipeline_identity") or len(release_records) != 300:
        raise RuntimeError(f"Selected release index is not verified as 300 articles ({len(release_records)} found)")
    run_identity = digest({"pipeline": pipeline.identity, "release_index": release_identity,
                           "sample": sample_identity, "allocation": ALLOCATION_ID})
    if progress and progress.get("run_identity") != run_identity:
        raise RuntimeError("Saved smoke-check progress belongs to different inputs or model identities")
    if not progress:
        progress = {"run_identity": run_identity, "ui": {}, "articles": {}, "cache_checks": {},
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    report = read_json(REPORT_PATH, {})
    if report and report.get("run_identity") != run_identity:
        raise RuntimeError("Existing live-check report has a different identity; refusing to overwrite it")
    release_db_before = file_digest(ARTIFACTS / "release.sqlite")
    report.update({"run_identity": run_identity, "status": "running",
        "purpose": "Visitor-flow and multilingual pipeline smoke checks; not a quality evaluation.",
        "pipeline_identity": pipeline.identity, "selected_run_id": pipeline.backend.identity.get("run_id"),
        "release_index": {"pipeline_identity": release_identity, "article_count": len(release_records),
                          "database_sha256_before": release_db_before},
        "input_selection": {"source": "Mewsli-9-derived local corpus", "split": "train",
          "languages": {"en": 15, "ar": 15}, "indexed_ids_excluded": len(indexed_ids),
          "reviewed_and_training_ids_excluded": len(protected_ids), "previous_failure_ids_excluded": len(attempted_article_ids()),
          "sample_identity": sample_identity},
        "browser": {"executable": str(chrome), "screenshots_directory": str(OUT)},
        "budget_allocation_id": ALLOCATION_ID, "quality_metrics": None,
        "quality_note": "Unlabeled integration smoke inputs do not estimate extraction or recommendation quality."})
    save_progress(progress, report)

    # A full regression run must pass before any live model invocation.
    if not progress.get("unit_tests", {}).get("passed"):
        test_env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT,
                               env=test_env, capture_output=True, text=True, timeout=1200)
        test_output = tests.stdout + "\n" + tests.stderr
        (OUT / "unit-tests.log").write_text(test_output, encoding="utf-8")
        progress["unit_tests"] = {"passed": tests.returncode == 0, "return_code": tests.returncode,
                                  "summary": next((line.strip() for line in reversed(test_output.splitlines())
                                                   if line.strip().startswith(("Ran ", "FAILED", "OK"))), "summary unavailable")}
        report["unit_tests"] = progress["unit_tests"]
        save_progress(progress, report)
        if tests.returncode:
            report["status"] = "blocked_unit_tests_failed"
            write_json(REPORT_PATH, report)
            return 1

    port = available_port()
    app = None
    browser = None
    fatal = None
    try:
        app = portfolio_app.build_app().queue(default_concurrency_limit=1)
        app.launch(server_name="127.0.0.1", server_port=port, share=False, quiet=True,
                   prevent_thread_lock=True, show_error=True)
        import requests
        from playwright.sync_api import sync_playwright

        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("Local app did not become ready within 120 seconds")

        sample_en = next(row for row in selected if row["language"] == "en")
        sample_ar = next(row for row in selected if row["language"] == "ar")
        query_pairs = [
            ("English interest", "Iran nuclear diplomacy, sanctions and regional security"),
            ("Arabic interest", "التطورات السياسية في فلسطين وقطاع غزة والمساعدات الإنسانية"),
        ]
        recommendations = []
        for label, query in query_pairs:
            result = pipeline.recommend(query, {}, limit=10)
            recommendations.append({"label": label, "query": query, "status": result["status"],
                "total_articles": result["total_articles"], "top_results": [
                    {"article_id": row["article"]["article_id"], "title": row["article"]["title"],
                     "score": row["score"]} for row in result["results"][:5]]})
        interest_order_changed = distinct_interest_rankings_changed(recommendations)
        report["recommendation_smoke"] = {"queries": recommendations,
                                          "top_five_order_changed": interest_order_changed,
                                          "note": "Functional demonstration only; not judged relevance metrics."}
        save_progress(progress, report)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(chrome), headless=True,
                args=["--no-first-run", "--disable-background-networking", "--disable-features=Translate"])
            page = browser.new_page(viewport={"width": 1365, "height": 950}, device_scale_factor=1)
            page_errors = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
            page.get_by_role("heading", name="Find the news that matters to you.").wait_for(timeout=60000)
            progress["ui"]["app_desktop"] = {"status": "passed", "http_status": response.status_code,
                "title": page.title()}
            page.screenshot(path=str(OUT / "desktop.png"), full_page=True)

            for index, (label, query) in enumerate(query_pairs):
                page.get_by_label("Your interest").fill(query)
                page.get_by_role("button", name="Find relevant articles").click()
                expected = [row["title"] for row in recommendations[index]["top_results"]]
                page.wait_for_function(
                    "expected => { const actual = [...document.querySelectorAll('.news-card h3')].slice(0, expected.length).map(e => e.innerText.trim()); return JSON.stringify(actual) === JSON.stringify(expected); }",
                    arg=expected, timeout=120000)
                headings = [title.strip() for title in page.locator(".news-card h3").all_text_contents()]
                progress["ui"]["recommend_" + label.split()[0].lower()] = {
                    "status": "passed", "visible_cards_match_pipeline": True,
                    "visible_cards": len(headings), "top_titles": headings[:5]}
                save_progress(progress, report)

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(500)
            width = page.evaluate("({viewport: window.innerWidth, content: document.documentElement.scrollWidth})")
            progress["ui"]["mobile_layout"] = {"status": "passed" if width["content"] <= width["viewport"] + 2 else "horizontal_overflow",
                **width}
            page.screenshot(path=str(OUT / "mobile.png"), full_page=True)
            save_progress(progress, report)

            for language, article in (("en", sample_en), ("ar", sample_ar)):
                task = "live_article_" + language
                if progress["ui"].get(task, {}).get("status") in ("generated", "handled_failure"):
                    continue
                page.set_viewport_size({"width": 1365, "height": 950})
                page.goto(url, wait_until="domcontentloaded", timeout=120000)
                page.get_by_role("heading", name="Find the news that matters to you.").wait_for(timeout=60000)
                page.get_by_text("Try your own article: live model analysis", exact=False).click()
                page.get_by_label("Article title").fill(article["title"])
                page.get_by_label("Article text").fill(article["body"])
                page.get_by_role("button", name="Analyze this article").click()
                page.wait_for_function("() => /Analysis (generated|unavailable)/.test(document.body.innerText)",
                                        timeout=600000)
                body_text = page.locator("body").inner_text()
                generated = "Analysis generated" in body_text
                failure_visible = "Analysis unavailable" in body_text
                if not generated and not failure_visible:
                    raise RuntimeError(f"Live {language} article returned no visible success or failure")
                if failure_visible:
                    page.get_by_text("Full inference trace", exact=False).click()
                    page.wait_for_timeout(300)
                    body_text = page.locator("body").inner_text()
                progress["ui"][task] = {"status": "generated" if generated else "handled_failure",
                    "article_id": article["article_id"], "language": language,
                    "source_url": article.get("source_url"), "license_id": article.get("license_id"),
                    "visible_raw_trace_on_failure": ("raw_model_outputs" in body_text) if failure_visible else None,
                    "visible_summary": " ".join(body_text.split())[-1600:]}
                page.screenshot(path=str(OUT / f"live-{language}.png"), full_page=True)
                save_progress(progress, report)
                if failure_visible and check_resource_failure(body_text):
                    fatal = "Live app reported a memory, paging, or storage error; stopped before further inference."
                    break
            if page_errors:
                progress["ui"]["browser_page_errors"] = page_errors[:20]
            if browser:
                browser.close()
                browser = None

        if fatal:
            report["status"] = "paused_resource_failure"
            report["blocking_error"] = fatal
            save_progress(progress, report)
            return 2

        original_store = pipeline.store
        isolated_store = NewsStore(ISOLATED_DB)
        pipeline.store = isolated_store
        completed = {record_id(row) for row in rows(ANALYSES_PATH)} | {record_id(row) for row in rows(FAILURES_PATH)}
        try:
            for index, article in enumerate(selected, 1):
                article_id = article["article_id"]
                if article_id in completed:
                    continue
                try:
                    result = pipeline.analyze_article(article, persist=True)
                    append_result(ANALYSES_PATH, result)
                    row_status = "cache_hit" if result.get("cache_hit") else "generated"
                    progress["articles"][article_id] = {"status": row_status,
                        "language": article["language"], "elapsed_seconds": result.get("elapsed_seconds"),
                        "grounded_mentions": len(result.get("evidence", [])),
                        "entity_link_statuses": {status: sum(link.get("status") == status for link in result.get("entity_links", []))
                                                  for status in sorted({link.get("status") for link in result.get("entity_links", [])})}}
                except Exception as error:
                    failure = {"article_id": article_id, "language": article["language"],
                        "title": article["title"], "source_url": article.get("source_url"),
                        "license_id": article.get("license_id"), "error": str(error),
                        "raw_model_outputs": getattr(error, "raw_outputs", []) or [],
                        "chunk_metadata": getattr(error, "chunk_metadata", []) or []}
                    append_result(FAILURES_PATH, failure)
                    progress["articles"][article_id] = {"status": "visible_or_recorded_failure",
                        "language": article["language"], "error": str(error)}
                    if check_resource_failure(error):
                        fatal = f"Resource error during live batch: {error}"
                        break
                report["article_smoke"] = {"selected": len(selected),
                    "completed": len(progress["articles"]), "last_article": article_id,
                    "last_index": index, "database": str(ISOLATED_DB)}
                save_progress(progress, report)
        finally:
            pipeline.store = original_store

        if not fatal:
            first_article = selected[0]
            if not progress.get("cache_checks", {}).get("same_input_hit"):
                pipeline.store = isolated_store
                try:
                    same = pipeline.analyze_article(first_article, persist=True)
                    progress["cache_checks"]["same_input_hit"] = bool(same.get("cache_hit"))
                    progress["cache_checks"]["same_input_cache_key"] = same.get("cache_key")
                finally:
                    pipeline.store = original_store
            if not progress.get("cache_checks", {}).get("edited_title_miss"):
                pipeline.store = isolated_store
                try:
                    edited = {**first_article, "title": first_article["title"] + " [cache invalidation check]"}
                    edited_analysis = pipeline.analyze_article(edited, persist=True)
                    progress["cache_checks"]["edited_title_miss"] = not edited_analysis.get("cache_hit")
                    progress["cache_checks"]["original_key"] = digest({"title": first_article["title"],
                        "body": first_article["body"], "pipeline": pipeline.identity})
                    progress["cache_checks"]["edited_key"] = edited_analysis.get("cache_key")
                    progress["cache_checks"]["edited_test_input"] = "Title-only edit explicitly marked as a cache test; not corpus evidence."
                except Exception as error:
                    append_result(FAILURES_PATH, {"article_id": first_article["article_id"],
                        "test": "edited_title_cache_miss", "error": str(error),
                        "raw_model_outputs": getattr(error, "raw_outputs", []) or []})
                    progress["cache_checks"]["edited_title_miss"] = False
                    progress["cache_checks"]["edited_title_error"] = str(error)
                finally:
                    pipeline.store = original_store

        release_db_after = file_digest(ARTIFACTS / "release.sqlite")
        successes = len(rows(ANALYSES_PATH))
        failures = len(rows(FAILURES_PATH))
        ui_ok = all(progress.get("ui", {}).get(key, {}).get("status") in accepted for key, accepted in {
            "app_desktop": {"passed"}, "recommend_english": {"passed"},
            "recommend_arabic": {"passed"}, "mobile_layout": {"passed"},
            "live_article_en": {"generated", "handled_failure"},
            "live_article_ar": {"generated", "handled_failure"}}.items())
        cache_ok = (progress.get("cache_checks", {}).get("same_input_hit") is True
                    and progress.get("cache_checks", {}).get("edited_title_miss") is True)
        release_unchanged = release_db_before == release_db_after
        report["release_index"]["database_sha256_after"] = release_db_after
        report["release_index"]["unchanged"] = release_unchanged
        report["article_smoke"] = {"selected": len(selected), "successful": successes,
                                    "recorded_failures": failures, "source_split": "train",
                                    "not_a_quality_evaluation": True}
        report["cache_checks"] = progress.get("cache_checks", {})
        report["ui_checks_passed"] = ui_ok
        report["ranking_order_changed_for_distinct_interests"] = interest_order_changed
        report["completion_checks"] = {"all_30_inputs_accounted_for": successes + failures >= len(selected),
            "release_database_unchanged": release_unchanged, "desktop_mobile_and_ui": ui_ok,
            "edited_content_invalidates_live_cache": cache_ok,
            "distinct_interests_recompute_rankings": interest_order_changed,
            "notebooks_passed": all(row.get("status") == "passed" for row in read_json(
                ARTIFACTS / "notebook_validation" / "report.json", []))}
        blockers = [key for key, value in report["completion_checks"].items() if value is not True]
        report["status"] = "completed_with_model_failures" if failures and not blockers else (
            "passed" if not blockers else "completed_with_open_checks")
        report["open_checks"] = blockers
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save_progress(progress, report)
        return 0 if not blockers else 1
    except Exception as error:
        report["status"] = "interrupted_or_failed"
        report["error"] = str(error)
        report["error_type"] = type(error).__name__
        save_progress(progress, report)
        raise
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if app is not None:
            try:
                app.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
