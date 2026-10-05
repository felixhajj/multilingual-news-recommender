# Historical README: Superseded By START_HERE.md

This prototype demonstrates how a future Tactical Report article can be enriched and recommended without requiring that exact article to exist in the original dataset.

For a meeting-ready walkthrough, start with [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md)
and then run `notebooks/tactical_report_a_to_z_walkthrough.ipynb`.

It uses the visible filter categories:

```text
Countries | Companies | Organisations | Profiles | Systems | Topics
```

All included articles and profiles are mock demonstration data. No private Tactical Report content is included.

## What It Proves

```text
new article text
-> multilingual extraction or deterministic alias enrichment
-> canonical article filters
-> multilingual E5 embeddings
-> direct required-filter matches + related semantic matches
-> ranked recommendations with explanations
```

The prototype handles three important cases:

- Arabic, English, abbreviations, and transliterations can resolve to one canonical entity.
- Unknown names are retained as low-confidence review candidates instead of receiving a fabricated type.
- Explicit requirements such as `Lockheed Martin AND Saudi Arabia` rank as direct matches, while semantically similar reports remain related results.

## Models And Training

The project separates two model responsibilities:

- `Qwen/Qwen2.5-3B-Instruct` plus QLoRA learns article-to-filter extraction.
- `intfloat/multilingual-e5-base` embeds user interests and enriched articles for recommendation.

`data/extraction_examples.jsonl` contains reviewed mock extraction examples with fixed train, validation, and test splits. `notebooks/tactical_report_entity_extraction_qlora.ipynb` measures JSON validity and extraction F1 before and after training, then saves the adapter separately from Qwen.

Run the saved adapter over incoming articles with:

```powershell
$env:HF_HOME = "D:\hf-cache"
.\.venv\Scripts\python.exe scripts\extract_article_filters.py
```

The output is written to `output/extractions/incoming_article_filters.jsonl` with `review_status: pending`. Model labels are merged with deterministic catalogue matches, so a known Arabic alias can fill an entity the model missed while unknown names remain review candidates. This is the operational link between the trained QLoRA adapter and the recommendation pipeline; malformed JSON is rejected instead of silently indexed.

The browser prototype displays pending extraction metadata so it can be inspected. A production caller can use `apply_extraction_outputs(..., include_pending=False)` to expose only approved model-derived tags.

The recommender embeds the article title and summary as well as canonical filters. This means unseen future terminology remains available to semantic matching even before an alias is added to the catalogue.

New articles are embedded and indexed immediately. Model updates should be periodic and based on reviewed examples rather than automatic retraining on every article.

## Data Intake And Continual Learning

The project now keeps three kinds of data separate:

1. **Discovery metadata** finds potentially relevant articles. It is not model-training text.
2. **Approved domain corpus** contains article text with recorded provenance, reuse permission, and human approval.
3. **Reviewed extraction examples** teach the QLoRA adapter which filters and relationships to return.

This distinction prevents a crawler from silently turning arbitrary publisher content or incorrect model predictions into training data.

`data/corpus_sources.json` is the source-rights manifest. A document enters the generated training corpus only when all of these are true:

- Its source has `rights_status: approved` and allows `model_training`.
- The individual record has `allowed_for_training: true`.
- The record has `review_status: approved`.
- It is not a duplicate ID or duplicate article body.

Build and audit the approved corpus:

```powershell
.\.venv\Scripts\python.exe scripts\build_domain_corpus.py
```

The included `data/domain_corpus_sample.jsonl` has eight synthetic English, Arabic, and mixed-language records. It verifies the ingestion path; it is deliberately too small to justify a domain-adaptation training run.

Raw geopolitical articles and extraction labels are not interchangeable. A future domain-adaptation LoRA can improve familiarity with approved terminology, but it does not teach the fixed filter JSON schema. The current extraction LoRA therefore learns from reviewed article-to-filter examples, while the full approved article stream is embedded and indexed immediately for recommendation.

Discover current article URLs and titles through GDELT:

```powershell
.\.venv\Scripts\python.exe scripts\discover_gdelt_articles.py --max-records 100
```

GDELT is used for discovery only. The script sets `allowed_for_training` to `false` and does not copy publisher article bodies. After rights approval, authorised article text can be imported through the corpus builder. GDELT documents its global news graph and APIs on its [official data page](https://www.gdeltproject.org/data.html).

For continual improvement, an analyst reviews the extractor's output and corrects the labels. Approved records can then be promoted into the extraction training split:

```powershell
.\.venv\Scripts\python.exe scripts\promote_reviewed_extractions.py "path\to\reviewed_extraction_queue.jsonl"
```

`data/reviewed_extraction_queue.sample.jsonl` shows the required review-record format. The fixed validation and test examples are not changed by promotion. Retrain a versioned adapter periodically after enough approved examples accumulate; do not update its weights on every incoming article.

## Verified Results

The notebooks were executed end to end on the local GTX 1080 Ti:

```text
Extraction baseline:  micro-F1 0.1538 | valid JSON 40%
Extraction QLoRA:     micro-F1 0.5500 | valid JSON 100%
Recommendation case 1 Recall@5: 1.0
Recommendation case 2 Recall@5: 1.0
```

The saved extraction adapter is approximately 43 MB including tokenizer files. These numbers validate the pipeline, not production quality; the extraction dataset is intentionally small and must be expanded with approved reviewed examples.

## Hybrid Ranking

The score combines:

```text
75% multilingual embedding similarity
25% structured-filter coverage
```

If a profile has `required_filters`, every populated required category must match for the result to receive the `direct` tier. Direct results sort before broader `related` results.

The weights are prototype defaults. They should eventually be calibrated from evaluation cases, analyst feedback, and user behaviour.

## Entity Enrichment

`data/entities.json` is a small seed catalogue with canonical IDs and multilingual aliases. For example:

```text
Lockheed Martin | Lockheed | LMT | لوكهيد مارتن
Saudi Arabia | KSA | السعودية | المملكة العربية السعودية
```

`data/incoming_articles.json` contains mock future articles without prewritten tags. The enrichment pipeline:

1. Matches known aliases.
2. Adds canonical values to the correct filter categories.
3. Records matched entities and confidence.
4. Keeps unknown candidate names for review.

This catalogue is a seed, not a complete world model. A production catalogue can be expanded from approved internal metadata and reusable structured sources such as Wikidata.

Wikidata structured labels, aliases, and IDs are marked for `entity_catalogue` use in the source manifest, not article-text training. Its query service states that Wikidata data is available under [CC0](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/Copyright).

Generate a review queue of high-visibility defence companies, weapon systems, and MENA political profiles with English and Arabic labels:

```powershell
.\.venv\Scripts\python.exe scripts\discover_wikidata_entities.py --limit-per-category 100
```

The candidates stay at `review_status: pending`; the script does not merge questionable categories or aliases into the production catalogue automatically.

After an analyst changes accepted candidate records to `review_status: approved` and adds `reviewed_by` and `reviewed_at`, merge them with:

```powershell
.\.venv\Scripts\python.exe scripts\promote_reviewed_entities.py output\entities\wikidata_candidates.jsonl
```

## Evaluation

`data/evaluation_cases.json` stores manually reviewed relevance expectations. The recommendation notebook calculates Recall@5 so model changes can be compared against the same cases. The extraction notebook uses a separate untouched test split and measures JSON validity and entity micro-F1.

The evaluation data is not intended to contain all future terminology. Its purpose is to measure whether the system behaves correctly on representative English, Arabic, alias, exact-filter, and semantic cases.

## Run The Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Install The Project Packages

The existing course environment already has the working CUDA build of PyTorch for the GTX 1080 Ti. Install both project requirement files into that environment without reinstalling PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-training.txt
```

Verify the selected kernel before QLoRA training:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run The Notebook

Open:

```text
notebooks/tactical_report_a_to_z_walkthrough.ipynb
notebooks/tactical_report_recommendation_demo.ipynb
notebooks/tactical_report_entity_extraction_qlora.ipynb
```

Start with `tactical_report_a_to_z_walkthrough.ipynb`. It is the presentation notebook:
it exposes the actual token IDs, saved LoRA configuration and measured improvement, Qwen
JSON, canonical entities, E5 vector values, cosine similarity, exact matches, and final
hybrid score in chronological order.

Select the CUDA-enabled Python environment at:

```text
.\.venv\Scripts\python.exe
```

The first embedding cell downloads the model. Keep the Hugging Face cache on `D:`:

```powershell
$env:HF_HOME = "D:\hf-cache"
```

## Run The Browser Demo

```powershell
$env:HF_HOME = "D:\hf-cache"
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8501`.

The verified local server may already be running on that address. If the command reports that the port is in use, open the address before starting another process.

Useful endpoints:

```text
/api/recommendations?user_id=user_004
/api/pipeline-demo?user_id=user_004&article_id=incoming_001
/api/enrichment-preview
/api/embedding-map?user_id=user_004&article_id=incoming_001
```

## Project Structure

```text
app.py                                  HTTP server and API routes
data/entities.json                     Canonical entity and alias seed catalogue
data/incoming_articles.json            Untagged mock future articles
data/evaluation_cases.json             Reviewed relevance expectations
data/extraction_examples.jsonl         Reviewed extraction train/validation/test examples
data/corpus_sources.json               Source rights and allowed-use manifest
data/domain_corpus_sample.jsonl        Synthetic corpus-ingestion fixtures
data/reviewed_extraction_queue.sample.jsonl
notebooks/tactical_report_recommendation_demo.ipynb
notebooks/tactical_report_entity_extraction_qlora.ipynb
src/entity_enrichment.py                Alias linking and review candidates
src/entity_catalogue_pipeline.py        Reviewed Wikidata candidate promotion
src/extraction_data.py                  Extraction dataset validation and chat formatting
src/llm_extractor.py                    Saved QLoRA adapter loading and strict inference
src/corpus_ingestion.py                 Provenance, rights, review, and deduplication gate
src/review_pipeline.py                  Reviewed-example promotion for later retraining
src/embeddings.py                       Multilingual E5 embeddings and PCA
src/scoring.py                          Hybrid direct/related ranking
src/evaluation.py                       Recall@k evaluation
scripts/build_domain_corpus.py          Produce an approved auditable training corpus
scripts/extract_article_filters.py      Run the trained adapter over incoming articles
scripts/discover_gdelt_articles.py      Create a current-article review queue
scripts/discover_wikidata_entities.py   Create a multilingual entity review queue
scripts/promote_reviewed_entities.py    Merge approved entities and aliases
scripts/promote_reviewed_extractions.py Add analyst-approved corrections to training
tests/test_scoring.py                   Unit tests
```
