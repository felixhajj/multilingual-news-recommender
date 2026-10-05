"""Gradio live app; model weights stay on the server, never in the visitor's browser."""
import html
import json
import os
from functools import lru_cache

os.environ.setdefault("NEWS_EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
if os.getenv("SPACE_ID") and os.getenv("NEWS_ZEROGPU") == "1":
    import spaces

import gradio as gr

from src.news_pipeline import NewsPipeline, validate_article
from src.news_evaluation import validate_review, validate_ranking_review
from src.portfolio_config import (ARTIFACTS, ROOT, EMBEDDING_MODEL, EMBEDDING_REVISION,
                                  PIPELINE_VERSION, digest, read_json, read_jsonl, write_jsonl)
from src.portfolio_reports import evidence_report
from src.phase3_configuration import selected_configuration


@lru_cache(maxsize=1)
def get_pipeline():
    return NewsPipeline()


def ranked_html(payload):
    if not payload["results"]:
        if payload.get("status") == "index_rebuild_required":
            return ("<p class='notice'>The selected models are ready for fresh article analysis. "
                    f"The {payload['stale_index_articles']} saved articles still use an older model version; "
                    "they will appear after their new extraction, entity links, and embeddings are processed.</p>")
        return "<p class='notice'>No fully processed articles are available yet. The evidence tab shows preparation status.</p>"
    cards = []
    for position, row in enumerate(payload["results"], 1):
        article = row["article"]
        badges = "Required filters matched" if row["direct_match"] else "Related by meaning"
        cards.append(f"<article class='news-card'><span class='position'>{position:02}</span><div>"
                     f"<small>{html.escape(str(article.get('language') or ''))} / Historical news / {html.escape(str(article.get('date') or ''))}</small>"
                     f"<h3 dir='auto'>{html.escape(article['title'])}</h3><p>{badges} · Ranking score {row['score']:.3f}</p>"
                     f"<p class='excerpt' dir='auto'>{html.escape(article['body'][:260])}...</p></div></article>")
    return "".join(cards)


FILTER_FIELDS = ("countries", "locations", "companies", "organizations", "profiles", "systems", "topics")


def gpu_duration(title, text, *_args, **_kwargs):
    # Reserve less quota for short examples while bounding longer requests.
    return min(120, max(30, 20 + len(text or "") // 120))


def clear_analysis():
    return ("<p class='notice'>Waiting for fresh model analysis. Free GPU queue or quota limits "
            "may block this request; no saved answer is substituted.</p>",
            {}, [], {}, {"status": "requested"})


def analysis_controls(interactive):
    return [gr.update(interactive=interactive) for _ in range(4 + len(FILTER_FIELDS))]


def begin_analysis():
    return (*clear_analysis(), *analysis_controls(False))


def end_analysis():
    return analysis_controls(True)


def filters(*values):
    return {category: [v.strip() for v in (value or "").split(";") if v.strip()]
            for category, value in zip(FILTER_FIELDS, values)}


def index_notice():
    selected = selected_configuration()
    if not selected:
        return "No validation-locked model configuration is active."
    release = read_json(ARTIFACTS / "release_manifest.json", {})
    indexed_run = release.get("model", {}).get("run_id", "unknown")
    linker_version = digest(selected["linker_identity"])
    selected_model = selected["model"]
    indexed_model = release.get("model", {})
    model_fields = ("adapter_sha256", "adapter_config_sha256", "prompt_sha256",
                    "schema_version", "max_new_tokens", "chunk_tokens", "chunk_stride")
    current = (indexed_run == selected["run_id"]
               and all(indexed_model.get(key) == selected_model.get(key) for key in model_fields)
               and release.get("linker_version") == linker_version
               and release.get("embedding_version") == (
                   EMBEDDING_MODEL + ":" + EMBEDDING_REVISION + ":normalized-chunk-mean-v2")
               and release.get("pipeline_version") == PIPELINE_VERSION)
    if current:
        return (f"**Selected models:** `{selected['run_id']}` + learned linker. "
                f"Current-version index: {release.get('articles', 0)}/{release.get('target', 300)} articles "
                f"({html.escape(release.get('status', 'unknown'))}).")
    return (f"**Selected models:** `{selected['run_id']}` + learned linker. "
            f"The saved {release.get('articles', 0)}-article index uses `{html.escape(str(indexed_run))}`; "
            "it will not be presented as if the selected models had processed it. Fresh analysis works now; "
            "indexed recommendations appear as the matching index is rebuilt.")


def recommend(interest, country, location, company, organization, profile, system, topic):
    try:
        if not interest.strip() or len(interest) > 1500:
            raise ValueError("Enter an interest statement between 1 and 1,500 characters")
        result = get_pipeline().recommend(interest, filters(country, location, company,
                                                               organization, profile, system, topic))
        choices = [(r["article"]["title"], r["cache_key"]) for r in result["results"]]
        return ranked_html(result), result, gr.update(choices=choices, value=None)
    except (ValueError, RuntimeError, OSError) as exc:
        return f"<p class='notice'>{html.escape(str(exc))}</p>", {"status": "unavailable", "error": str(exc)}, gr.update(choices=[], value=None)


def explanation_html(explanation):
    escape = lambda value: html.escape(str(value))
    comparisons = explanation["qwen"]["filter_comparison"]
    rows = "".join(f"<tr><td>{escape(c['category'])}</td><td>{escape(', '.join(c['requested']))}</td>"
                   f"<td dir='auto'>{escape(', '.join(c['article']) or 'None extracted')}</td>"
                   f"<td>{'Match' if c['satisfied'] else 'No match'}</td></tr>" for c in comparisons)
    filters_view = ("<table class='comparison'><thead><tr><th>Category</th><th>Your filter</th><th>Article facts</th><th>Result</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table>") if rows else "<p>No required filters. Ranking uses semantic similarity alone.</p>"
    pairs = "".join(f"<tr><td>{escape(p['user_phrase'])}</td><td dir='auto'>{escape(p['article_phrase'])}</td><td>{p['similarity']:.3f}</td></tr>"
                    for p in explanation["e5"]["phrase_pairs"])
    return ("<section class='result-summary'><h3>Qwen + learned linker: article facts versus your filters</h3>" + filters_view +
            f"<h3>E5: how closely the meanings match</h3><p>Cosine similarity: <strong>{explanation['e5']['similarity']:.3f}</strong> (not confidence)</p>"
            "<p>Arabic and English are embedded directly. No manual translation.</p>"
            "<table class='comparison'><thead><tr><th>Your interest phrase</th><th>Article phrase</th><th>E5 similarity</th></tr></thead>"
            f"<tbody>{pairs}</tbody></table><small>Independent phrase probes, not a definitive explanation of the model's reasoning.</small></section>")


def explain_selected(cache_key, interest, country, location, company, organization, profile, system, topic):
    try:
        if not cache_key:
            return "<p>Select a ranked article first.</p>", {}
        pipeline = get_pipeline()
        analysis = pipeline.store.cached(cache_key)
        if not analysis or analysis["pipeline_identity"] != pipeline.index_identity():
            raise ValueError("This saved example belongs to a different model version. Run a fresh search against the current index.")
        result = pipeline.explain(interest, analysis, filters(country, location, company,
                                                               organization, profile, system, topic))
        return explanation_html(result), result
    except (ValueError, RuntimeError, OSError) as exc:
        return f"<p class='notice'>{html.escape(str(exc))}</p>", {"status": "unavailable", "error": str(exc)}


def analysis_failure(exc):
    raw_outputs = getattr(exc, "raw_outputs", None)
    if raw_outputs is None and getattr(exc, "raw_output", None):
        raw_outputs = [exc.raw_output]
    trace = {"status": "failed", "error": str(exc),
             "raw_model_outputs": raw_outputs or [],
             "chunk_metadata": getattr(exc, "chunk_metadata", [])}
    return (f"<p class='notice'>Analysis unavailable: {html.escape(str(exc))}</p>",
            {}, [], {}, trace)


def _analyze_core(title, text, interest, country, location, company, organization, profile, system, topic,
                  progress=gr.Progress()):
    filter_values = (country, location, company, organization, profile, system, topic)
    try:
        progress(0.1, desc="Qwen + LoRA: reading article")
        pipeline = get_pipeline()
        result = pipeline.analyze_article({"title": title, "body": text}, persist=False)
        progress(0.8, desc="E5: comparing your interest with the article")
        explanation = pipeline.explain(interest, result, filters(*filter_values))
        links = [[r["mention"], r.get("canonical_name") or "Unresolved", r.get("entity_id") or "", r["status"]]
                 for r in result["entity_links"]]
        score = explanation["e5"]["similarity"]
        summary = (f"<div class='result-summary'><strong>Analysis generated</strong><p>"
                   f"{len(result['evidence'])} grounded mentions · E5 similarity {score:.3f}</p>"
                   "<small>Similarity is not a probability. Unresolved names stay visible for review.</small></div>")
        progress(1, desc="Complete")
        return summary + explanation_html(explanation), result["extraction"], links, explanation, result
    except (ValueError, RuntimeError, OSError) as exc:
        return analysis_failure(exc)


_run_analysis = _analyze_core
if os.getenv("SPACE_ID") and os.getenv("NEWS_ZEROGPU") == "1":
    import spaces
    # ZeroGPU requires placement during startup, outside the decorated request.
    get_pipeline().backend.load()
    _run_analysis = spaces.GPU(duration=gpu_duration)(_analyze_core)


def analyze(title, text, interest, country, location, company, organization, profile, system, topic,
            progress=gr.Progress()):
    try:
        # Reject invalid inputs before acquiring a lease or consuming GPU quota.
        validate_article({"title": title, "body": text})
        if not isinstance(interest, str) or not interest.strip() or len(interest) > 1500:
            raise ValueError("Enter an interest statement between 1 and 1,500 characters")
        return _run_analysis(title, text, interest, country, location, company,
                             organization, profile, system, topic, progress=progress)
    except (ValueError, RuntimeError, OSError, gr.Error) as exc:
        # Lease rejection occurs outside the decorated function's own handler.
        return analysis_failure(exc)


CSS = """
.gradio-container {max-width:1100px!important;background:#f7f6f1!important;color:#173b42!important}
.hero h1,.hero p,.news-card h3,.notice,.result-summary,.flow {color:#173b42!important}
.hero {padding:36px 0 22px;border-bottom:1px solid #c9d7d5;margin-bottom:22px}
.hero small,.eyebrow {letter-spacing:.14em;text-transform:uppercase;color:#13706e;font-weight:700}
.hero h1 {font:500 clamp(30px,4vw,48px) Georgia,serif;margin:12px 0}
.hero p {font-size:17px;max-width:720px;line-height:1.6}
.news-card {display:flex;gap:20px;padding:24px 10px;border-bottom:1px solid #d4dedc}
.news-card h3 {font:500 23px Georgia,serif;margin:8px 0;line-height:1.4}
.news-card small {color:#546d70}.position {font:26px Georgia,serif;color:#19817d}
.excerpt {color:#52686a;font-size:14px}.notice {padding:18px;border-left:3px solid #bc692e;background:#fff4e8}
.result-summary {padding:20px;background:#e3f0ec;border-left:4px solid #10736b}
.comparison {width:100%;border-collapse:collapse;margin:12px 0 24px}.comparison th,.comparison td {padding:9px;text-align:start;border-bottom:1px solid #bdcfca}
.flow {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}
.flow div {padding:20px;background:#e5edeb;border-top:3px solid #10736b}.flow strong {display:block;margin-bottom:10px}
@media(max-width:650px){.flow{grid-template-columns:1fr}.news-card{gap:12px}.hero{padding-top:18px}}
"""

THEME = gr.themes.Soft(primary_hue="teal").set(
    body_background_fill="#f7f6f1", body_background_fill_dark="#f7f6f1",
    body_text_color="#173b42", body_text_color_dark="#173b42",
    body_text_color_subdued="#52686a", body_text_color_subdued_dark="#52686a",
    background_fill_primary_dark="#ffffff", background_fill_secondary_dark="#edf2ef",
    block_background_fill_dark="#ffffff", block_label_background_fill_dark="#edf2ef",
    block_label_text_color_dark="#173b42", block_title_text_color_dark="#173b42",
    block_info_text_color_dark="#52686a", input_background_fill_dark="#ffffff",
    accordion_text_color_dark="#173b42", table_text_color_dark="#173b42",
    button_secondary_background_fill_dark="#edf2ef", button_secondary_text_color_dark="#173b42",
    code_background_fill_dark="#edf2ef", panel_background_fill_dark="#ffffff")


def review_load(article_id):
    rows = read_jsonl(ARTIFACTS / "review" / "extraction.jsonl")
    record = next(r for r in rows if r["article_id"] == article_id)
    template = {key: [] for key in ("countries", "companies", "organizations", "profiles", "systems", "topics", "relationships")}
    return record["title"] + "\n\n" + record["text"], json.dumps(record.get("labels") or template, ensure_ascii=False, indent=2)


def review_save(article_id, labels_text, reviewer):
    if not reviewer.strip():
        raise gr.Error("Enter your reviewer name")
    rows = read_jsonl(ARTIFACTS / "review" / "extraction.jsonl")
    record = next(r for r in rows if r["article_id"] == article_id)
    record.update(labels=json.loads(labels_text), review_status="human_reviewed", reviewed_by=reviewer.strip())
    count = validate_review(rows)
    write_jsonl(ARTIFACTS / "review" / "extraction.jsonl", rows)
    return f"Saved. {count}/{len(rows)} reviewed. These examples stay outside training."


def ranking_review_load(key):
    rows = read_jsonl(ARTIFACTS / "review" / "ranking.jsonl")
    row = next(r for r in rows if r["query_id"] + ":" + r["article"]["article_id"] == key)
    return (f"Interest: {row['interest']}\nRequired filters: {json.dumps(row['required_filters'])}\n\n"
            f"{row['article']['title']}\n\n{row['article']['body']}"), row.get("relevance")


def ranking_review_save(key, grade, reviewer):
    rows = read_jsonl(ARTIFACTS / "review" / "ranking.jsonl")
    row = next(r for r in rows if r["query_id"] + ":" + r["article"]["article_id"] == key)
    row.update(relevance=grade, reviewed_by=reviewer.strip(), review_status="human_reviewed")
    count = validate_ranking_review(rows)
    write_jsonl(ARTIFACTS / "review" / "ranking.jsonl", rows)
    return f"Saved {count}/30 relevance judgments."


def build_app():
    with gr.Blocks(title="Multilingual News Recommender", css=CSS, theme=THEME) as app:
        gr.HTML("<header class='hero'><small>Applied machine learning / English + Arabic</small>"
                "<h1>Find the news that matters to you.</h1>"
                "<p>Write what you follow. Explore real historical news, or give the models a new article to analyze."
                " Every result has a trace back to its text and model run.</p>"
                "<small>Built with Qwen. Research/evaluation only.</small></header>")
        with gr.Tab("Try it"):
            interest = gr.Textbox(label="Your interest", value="Iran nuclear diplomacy and international sanctions", lines=2)
            filter_inputs = []
            with gr.Accordion("Optional required filters", open=False):
                gr.Markdown("Match any name within a category, and every category you fill. Matching mentions alone does not prove a relationship.")
                for start in range(0, len(FILTER_FIELDS), 3):
                    with gr.Row():
                        for category in FILTER_FIELDS[start:start+3]:
                            field = gr.Textbox(label=f"{category.title()} (alternatives with ;)", value="")
                            filter_inputs.append(field)
            search = gr.Button("Find relevant articles", variant="primary")
            gr.Markdown(index_notice())
            snapshot = read_json(ROOT / "data" / "release" / "example_ranking.json", {"results": []})
            ranking = gr.HTML(ranked_html(snapshot))
            gr.Markdown("The initial card is a recorded historical model result. **Find relevant articles** recalculates ranking against articles processed by the selected models.")
            with gr.Accordion("Ranking details", open=False):
                ranking_json = gr.JSON(label="Actual ranking output", value=snapshot)
            selected_result = gr.Dropdown(choices=[(r["article"]["title"], r["cache_key"]) for r in snapshot["results"]], label="Inspect a result")
            result_explanation = gr.HTML()
            with gr.Accordion("Detailed result evidence", open=False):
                result_details = gr.JSON()
            selected_result.change(explain_selected, [selected_result, interest, *filter_inputs], [result_explanation, result_details])
            search.click(recommend, [interest, *filter_inputs], [ranking, ranking_json, selected_result])
            with gr.Accordion("Try your own article: live model analysis", open=False):
                gr.Markdown("New text runs through Qwen + LoRA and the learned linker, then E5. Models run on the server. Your pasted text is not added to the public corpus or training data.")
                title = gr.Textbox(label="Article title")
                text = gr.Textbox(label="Article text", lines=7, max_lines=20)
                generate = gr.Button("Analyze this article", variant="primary")
                status = gr.HTML()
                gr.Markdown("### 1. Qwen + LoRA: extracted article facts")
                generated_json = gr.JSON(label="Generated JSON")
                gr.Markdown("### 2. Learned linker: mentions to entity IDs")
                links = gr.Dataframe(headers=["Article mention", "Canonical name", "Entity ID", "Status"], interactive=False)
                gr.Markdown("### 3. E5: your interest compared with the article")
                explained = gr.JSON(label="Meaning, matching filters and supporting phrases")
                with gr.Accordion("Full inference trace", open=False):
                    trace = gr.JSON()
                analysis_outputs = [status, generated_json, links, explained, trace]
                controls = [title, text, interest, *filter_inputs, generate]
                generate.click(begin_analysis, outputs=[*analysis_outputs, *controls], queue=False,
                               api_name=False).then(
                    analyze, [title, text, interest, *filter_inputs], analysis_outputs, api_name="analyze").then(
                    end_analysis, outputs=controls, queue=False, api_name=False)
        with gr.Tab("How it works"):
            gr.HTML("<div class='flow'><div><strong>Qwen + LoRA</strong>Article text becomes structured mentions.</div>"
                    "<div><strong>Learned entity linker</strong>Context helps choose an entity ID, or abstain.</div>"
                    "<div><strong>Multilingual E5</strong>Your interest and article become comparable vectors.</div></div>")
            gr.Markdown("E5 places Arabic and English in a shared meaning space. There is no manual translation step. Its similarity score is combined with explicit filter coverage for ranking. Phrase probes illustrate similarities; they are not a claim about the model's internal reasoning.")
            gr.Markdown("Start with `notebooks/tactical_report_a_to_z_walkthrough.ipynb`. Each stage calls the same Python implementation used here. The other notebooks cover extraction training and recommendation evaluation.")
            steps = read_json(ROOT / "data" / "release" / "notebook_stages.json", {})
            for stage, cell in steps.items():
                with gr.Accordion(cell["title"], open=False):
                    gr.Code(cell["source"], language="python", label=stage)
        with gr.Tab("Evidence"):
            gr.Markdown("Training and evaluation reports are read from saved run artifacts. Missing metrics remain pending. Machine-generated training labels and human-reviewed test labels are reported separately.")
            gr.Markdown("Saved extraction metrics measure the local 4-bit execution profile. The hosted demo uses the same selected adapter with float16 base weights and a separate execution identity. Hosted smoke checks prove functionality, not identical quality scores.")
            report = gr.JSON(value=evidence_report(), label="Measured evidence and current status")
            gr.Button("Refresh evidence").click(evidence_report, outputs=report)
        if os.getenv("NEWS_REVIEW_MODE") == "1" and not os.getenv("SPACE_ID"):
            with gr.Tab("Local review"):
                rows = read_jsonl(ARTIFACTS / "review" / "extraction.jsonl")
                gr.Markdown("Review all explicit mentions using the original spelling. This is independent evaluation data. Save only after checking the whole article; an empty array means you checked that field.")
                selected = gr.Dropdown([r["article_id"] for r in rows], label="Frozen article")
                review_text = gr.Textbox(label="Source article", lines=12, interactive=False)
                review_labels = gr.Code(label="Correct JSON", language="json", interactive=True)
                reviewer = gr.Textbox(label="Reviewer name")
                selected.change(review_load, selected, [review_text, review_labels])
                message = gr.Textbox(label="Review status", interactive=False)
                gr.Button("Save reviewed reference").click(review_save, [selected, review_labels, reviewer], message)
                gr.Markdown("### Recommendation judgments\nGrade 0 = irrelevant, 1 = partly relevant, 2 = directly relevant. Judge the article against the interest and required constraints.")
                ranking_path = ARTIFACTS / "review" / "ranking.jsonl"
                judgments = read_jsonl(ranking_path) if ranking_path.exists() else []
                choice = gr.Dropdown([r["query_id"] + ":" + r["article"]["article_id"] for r in judgments], label="Frozen relevance judgment")
                source = gr.Textbox(label="Interest and article", lines=12, interactive=False)
                grade = gr.Radio([0, 1, 2], label="Relevance")
                choice.change(ranking_review_load, choice, [source, grade])
                gr.Button("Save relevance judgment").click(ranking_review_save, [choice, grade, reviewer], message)
    return app


def launch(port=None):
    build_app().queue(default_concurrency_limit=1).launch(server_name=os.getenv("NEWS_HOST", "127.0.0.1"),
        server_port=port or int(os.getenv("PORT", "8502")), show_error=True)


if __name__ == "__main__":
    launch()
