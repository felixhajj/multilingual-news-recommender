"""Generate readable, executable notebooks and stable app-to-code stage links."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys, os, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "news_pipeline.py").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEWS_EMBEDDING_DEVICE", "cpu")
from src.portfolio_config import ARTIFACTS, BASE_MODEL, EMBEDDING_MODEL, configure_cache, read_json, read_jsonl
configure_cache()
from src.portfolio_reports import evidence_report
print("Project:", ROOT)
print("Artifacts:", ARTIFACTS)
'''


def notebook(title, introduction, stages):
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{introduction}")]
    for identifier, heading, explanation, source in stages:
        cells.append(nbf.v4.new_markdown_cell(f'<a id="{identifier}"></a>\n## {heading}\n\n{explanation}'))
        cells.append(nbf.v4.new_code_cell(source, metadata={"stage_id": identifier, "stage_title": heading}))
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (news recommender)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}})


def main():
    stages = [
        ("setup", "1. Locate the project", "Run from the top with the project environment selected. Nothing here retrains automatically. Local notebooks need installed dependencies and downloaded artifacts; website visitors do not.", SETUP),
        ("evidence", "2. Check what actually exists", "Read measured artifact manifests, not a claim that every experiment succeeded. Missing training, review, or deployment remains visible. Historical mock metrics are not real-news results.",
         'report = evidence_report()\ndisplay(report)'),
        ("inputs", "3. One real article and your own interest", "The interest sentence belongs to the user. Required filters are optional structured constraints, not Qwen output. This first view uses a recorded real inference example; its source and model identity remain attached.",
         '''examples = read_json(ROOT / "data" / "release" / "example_analyses.json", [])
if not examples:
    raise RuntimeError("No genuine inference examples yet. Run scripts/build_news_release.py after setup.")
analysis = examples[0]
article = analysis["article"]
interest = "Iran nuclear diplomacy and international sanctions"
required_filters = {"countries": ["Iran"]}
display({"article": article, "user_interest": interest, "required_filters": required_filters})'''),
        ("tokens", "4. Qwen: text becomes token IDs", "This is the same text-to-token step as a text-generation LLM course. Qwen's tokenizer is unchanged. Arabic names may use several subword tokens; that is normal. Fine-tuning changes model behavior, not these token boundaries.",
         '''from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=False)
text = article["title"] + "\\n\\n" + article["body"]
token_ids = tokenizer.encode(text, add_special_tokens=False)
display({"model": BASE_MODEL, "vocabulary_size": len(tokenizer), "article_tokens": len(token_ids)})
display(list(zip(tokenizer.convert_ids_to_tokens(token_ids[:30]), token_ids[:30])))'''),
        ("adapter", "5. QLoRA: what was actually trained", "Qwen is the pretrained generator. LoRA is the small set of trainable matrices attached to it; QLoRA keeps the frozen base in 4-bit during training. Article/JSON examples train the adapter through Qwen. Lower loss and changed weights prove optimization happened, not that quality is sufficient. Phase 3 selected extraction-only-v3 using the predeclared validation gate. This is a research candidate being applied in Phase 4, not a production claim.",
         '''for run in report["extraction_experiments"]:
    display(run)
    loss_path = ARTIFACTS / "runs" / run["run_id"] / "loss.jsonl"
    loss = read_jsonl(loss_path) if loss_path.exists() else []
    display(loss[:3] + loss[-3:] if loss else "No saved optimizer steps")
from src.phase3_configuration import selected_configuration
selection = selected_configuration()
display({"validated_phase3_selection": selection["run_id"] if selection else None,
         "selected_adapter": str(selection["adapter_path"]) if selection else None,
         "displayed_example_model": analysis["model"]})'''),
        ("extraction", "6. Qwen output: generated facts with source evidence", "These JSON fields were generated in the recorded run, not hand-entered article tags. Cached inference is legitimate when tied to identical text and model versions. Unsupported predictions are shown and excluded from filters. Run the optional live cell at the end to test genuinely new text.",
         '''display({"generated_at": analysis["generated_at"], "cache_key": analysis["cache_key"],
         "raw_generated_text": analysis["raw_model_outputs"], "parsed_json": analysis["extraction"],
         "grounded_passages": analysis["evidence"], "excluded_predictions": analysis["ungrounded_predictions"]})'''),
        ("linking", "7. Learned entity linking", "E5 retrieves candidate entity descriptions. A supervised logistic classifier scores context and name features and can abstain. Known aliases remain useful features, but are not a manually configured final lookup decision. The catalogue supplies IDs and is limited to entities in the training partition. An unresolved name is not a made-up entity.",
         '''from src.phase3_configuration import selected_configuration
selection = selected_configuration()
display(analysis["entity_links"])
display(read_json(selection["linker_path"] / "metrics.json", {"status": "metrics unavailable"}))'''),
        ("filters", "8. Qwen + linker: the structured recommendation contribution", "Put the user's requested filters next to article tags. A direct match means every requested category has at least one match. These are application comparisons of model-derived facts, not an E5 prediction. Two names co-occurring do not prove a contract or other relationship.",
         '''from src.news_pipeline import compare_filters
comparison, coverage, direct = compare_filters(required_filters, analysis["tags"])
display(comparison)
display({"required_categories_covered": coverage, "all_required_categories_match": direct})'''),
        ("e5", "9. A different pretrained model: multilingual E5", "E5 compares your written interest with the article's natural text, without adding the exact filters. It directly embeds Arabic and English in a shared space; there is no manual translation. It is not fine-tuned here. Long articles are chunked and their normalized vectors are averaged. The dot product is similarity, not a probability.",
         '''from src.learned_linker import encode_batch
query_text = "query: " + interest
passage_text = "passage: " + text
vectors = encode_batch([query_text, passage_text])
display({"query": query_text, "passage": passage_text, "vector_shape": vectors.shape,
         "first_8_user_values": vectors[0, :8].tolist(), "first_8_article_values": vectors[1, :8].tolist(),
         "cosine_similarity": float(vectors[0] @ vectors[1])})'''),
        ("phrases", "10. Show supporting phrase similarities", "These pairs are calculated by E5 for the current texts, not manually authored translations. They are independent phrase probes, not a definitive explanation of attention or causality. They inspect a bounded set of phrases and can be wrong.",
         '''from src.embeddings import find_e5_phrase_matches
display(find_e5_phrase_matches(interest, text))'''),
        ("ranking", "11. Rank the current processed collection", "The same NewsPipeline powers the live app. It compares a fresh user vector with stored article vectors. Hybrid ranking combines E5 with exact filter coverage; without filters it equals E5. Only records matching the current model/configuration enter the live index. If the index is stale, rebuild it rather than mixing old and new model results.",
         '''from src.news_pipeline import NewsPipeline
pipeline = NewsPipeline()
ranking = pipeline.recommend(interest, required_filters, limit=5)
display(ranking)
second_interest = "Elections and constitutional reform"
display(pipeline.recommend(second_interest, limit=5))'''),
        ("live", "12. Optional: analyze genuinely new article text", "Set RUN_LIVE to True and replace the text. This performs fresh Qwen generation and linking; on a 1080 Ti it can take minutes. Your text is not persisted or used for training by default. Training holds a GPU lock, so wait for that job to finish. Failures remain failures, never prewritten tags.",
         '''RUN_LIVE = False
new_article = {"title": "Replace with your article title", "body": "Replace this sentence with a real article you may use."}
if RUN_LIVE:
    live = pipeline.analyze_article(new_article, persist=False)
    display(live)
    display(pipeline.explain(interest, live, required_filters))
else:
    print("Live generation not requested. The earlier JSON was a labeled saved model run.")'''),
        ("evaluation", "13. What counts as evidence", "JSON parsing measures syntax; schema validity measures required fields and types; entity F1 measures extracted content against independently reviewed labels. The linker has its own accuracy/abstention metrics. Recommendations use Recall@5 and nDCG@5 on the three fixed judged pools. All 30 extraction and 30 recommendation references have human judgments; the pools are small and machine-assisted training labels remain separate.",
         'display(evidence_report())'),
    ]
    training = [
        ("setup", "1. Environment and budget", "This notebook exposes the actual training path. CUDA training belongs on the NVIDIA machine, not a visitor's browser or Mac. Start with the A-to-Z notebook. Every expensive action is explicit.", SETUP),
        ("training-data", "2. Real articles and label provenance", "Raw news comes from licensed historical Wikinews/Mewsli-9. Published hyperlinks are partial positive entity annotations, not exhaustive JSON gold. The base model proposes JSON, literal grounding and agreement checks reject unsuitable candidates. Accepted labels remain machine-assisted and imperfect.",
         '''display(read_json(ARTIFACTS / "corpus_manifest.json", {}))
training_path = ARTIFACTS / "extraction_training.jsonl"
examples = read_jsonl(training_path) if training_path.exists() else []
print("Accepted examples:", len(examples))
display(examples[:2])'''),
        ("training-split", "3. Verify separation from evaluation", "Article groups, translations and detected near-duplicates stay within a split. Thirty independently reviewed articles are frozen outside our training. Historical data might have been seen during original pretraining; we make no stronger claim.",
         '''reviews = read_jsonl(ARTIFACTS / "review" / "extraction.jsonl")
assert not ({r["article_id"] for r in reviews} & {r["article_id"] for r in examples})
assert all(e["split"] == "train" for e in examples)
print("Frozen review articles:", len(reviews))'''),
        ("training-code", "4. Inspect the full real trainer", "The shared implementation loads frozen 4-bit Qwen, attaches LoRA, masks prompt tokens for extraction training, computes next-token loss, backpropagates into adapter parameters, and saves changed weights. Domain adaptation first learns raw news token prediction; it does not resize the vocabulary. Inspect the function rather than duplicating a second trainer in this notebook.",
         '''import inspect
from src.news_training import train_adapter, prepare_training_examples
print(inspect.getsource(train_adapter))'''),
        ("training-run", "5. Run the bounded comparison deliberately", "This command prepares labels and runs extraction-only, raw-domain adaptation, and domain-plus-extraction experiments. Each gets an immutable run ID. A time-limited or failed stage is recorded honestly. Existing completed runs are not silently overwritten. The shared GPU lock prevents conflicting jobs.",
         '''RUN_TRAINING = False
import subprocess
command = [sys.executable, str(ROOT / "scripts" / "run_news_experiments.py"), "--examples", "100", "--steps", "24", "--label-hours", "2"]
print(" ".join(command))
if RUN_TRAINING:
    subprocess.run(command, cwd=ROOT, check=True)'''),
        ("training-proof", "6. Inspect optimization evidence", "Changed weights and loss establish that training happened. They do not establish better extraction. Compare base, extraction-only and domain-plus-extraction on the same frozen references before promoting any candidate.",
         '''for manifest in sorted((ARTIFACTS / "runs").glob("*/manifest.json")):
    display(read_json(manifest))
    display(read_jsonl(manifest.parent / "loss.jsonl") if (manifest.parent / "loss.jsonl").exists() else "No optimizer steps saved yet")
display(read_json(ARTIFACTS / "extraction_evaluation.json", {"status": "pending human review"}))'''),
        ("training-next", "7. Evaluate, promote, or roll back", "Generate predictions separately with evaluate_news.py. Review the frozen references in the local app; score them without changing the test set. Promotion checks validation evidence and saves the previous deployment for rollback. Ingesting an article never retrains automatically.",
         '''print("python scripts/evaluate_news.py predict --hours 2")
print("python scripts/evaluate_news.py score")
print("python scripts/manage_news_model.py promote --run-id extraction-only-v1")
print("python scripts/manage_news_model.py rollback")'''),
    ]
    recommendation = [
        ("setup", "1. Shared project environment", "This notebook isolates retrieval and evaluation. It calls the same code as the app and does not train E5.", SETUP),
        ("retrieval-code", "2. Inspect retrieval and explanations", "Article vectors are saved once per model/content version. Each new interest gets a new vector. Exact filters and semantic similarities remain separate visible contributions. Phrase matches are similarity probes, not hidden reasoning.",
         '''import inspect
from src.news_pipeline import NewsPipeline
print(inspect.getsource(NewsPipeline.recommend))
print(inspect.getsource(NewsPipeline.explain))'''),
        ("retrieval-run", "3. Compare three ranking methods", "TF-IDF is the keyword baseline, E5 is multilingual semantic retrieval, and hybrid adds exact required-category coverage. Without required filters E5 and hybrid should agree. Scores are not percentages of confidence.",
         '''pipeline = NewsPipeline()
interest = "Iran nuclear diplomacy and sanctions"
required_filters = {"countries": ["Iran"]}
for method in ("keyword", "e5", "hybrid"):
    display({"method": method, "ranking": pipeline.recommend(interest, required_filters, method=method, limit=5)})'''),
        ("retrieval-change", "4. Change the input, not the stored result", "Try another interest. The ranking is recalculated using the same processed corpus. If no version-matching index exists, the explicit empty status tells you to rebuild it.",
         'display(pipeline.recommend("Lebanon and regional diplomacy", limit=5))'),
        ("retrieval-review", "5. Frozen relevance judgments", "Review thirty query/article pairs, graded 0, 1 or 2. Evaluation reranks each ten-article judged pool; it is not a claim of full-corpus Recall. Unjudged articles must never become automatic negatives. Selection and query roles are frozen before grading.",
         '''path = ARTIFACTS / "review" / "ranking.jsonl"
judgments = read_jsonl(path) if path.exists() else []
display(judgments[:2])
print("Judgments:", len(judgments))
display(read_json(ARTIFACTS / "recommendation_evaluation.json", {"status": "pending"}))'''),
        ("retrieval-metrics", "6. Read how the metrics are calculated", "Recall@5 asks how many judged relevant articles reached the top five. nDCG@5 rewards putting highly relevant articles nearer the top. No reviewed judgments means no claimed recommendation quality score.",
         '''from src.news_evaluation import ranking_metrics, ndcg_at_k
print(inspect.getsource(ranking_metrics))
print(inspect.getsource(ndcg_at_k))
display(evidence_report())'''),
    ]
    files = {
        "tactical_report_a_to_z_walkthrough.ipynb": notebook("Multilingual news recommender: A to Z", "The project name is historical; this is a standalone portfolio, not an enterprise product. Follow the data, tokens, adapter, generated facts, learned links, embeddings and recommendations in order.", stages),
        "tactical_report_entity_extraction_qlora.ipynb": notebook("Real-news extraction training", "A bounded learning experiment, with provenance and honest evaluation gates.", training),
        "tactical_report_recommendation_demo.ipynb": notebook("Multilingual retrieval and evaluation", "Compare keyword search, E5 and hybrid ranking using the shared live pipeline.", recommendation),
    }
    for name, value in files.items():
        nbf.validate(value)
        nbf.write(value, ROOT / "notebooks" / name)
    mapping = {identifier: {"title": title, "source": code,
               "notebook": "notebooks/tactical_report_a_to_z_walkthrough.ipynb", "anchor": identifier}
               for identifier, title, _, code in stages}
    destination = ROOT / "data" / "release"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "notebook_stages.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Built three notebooks and stable stage mapping")


if __name__ == "__main__":
    main()
