# Start Here

## The Project

Write an interest such as "Iran nuclear diplomacy and sanctions." Rank historical English and Arabic news, or paste a new article to inspect generated facts, entity links and relevance.

**Built with Qwen. Research/evaluation only.** No enterprise affiliation. This is not a current-news feed, fact checker, or production intelligence service.

## Two-Minute Visitor Route

1. Inspect the labeled, recorded model example without downloading models or waiting for a GPU. Verified public URLs are recorded in [deployment.json](data/release/deployment.json); an absent URL means public deployment is not verified.
2. In the app, change the interest and click **Find relevant articles**. Required filters are separate constraints. The verified local index contains 300 articles (156 Arabic, 144 English) under the selected model versions. Local app and notebook checks passed; public deployment is not verified.
3. Select a current-version result. Inspect **Qwen + learned linker** facts versus your filters, then **E5** similarity and phrase probes. Scores are not confidence percentages.
4. Paste an article you may use under **Try your own article**. This runs fresh inference, not a prewritten answer. Models stay on the server. Free GPU quotas/startup delays may interrupt this step; saved results and the guide stay readable.

## Five-Minute Technical Route

Open [the A-to-Z notebook](notebooks/tactical_report_a_to_z_walkthrough.ipynb). Run cells in order in the project kernel. Every block has an explanation; expensive generation is opt-in.

| Stable stage | What you see | Shared implementation |
| --- | --- | --- |
| `inputs`, `tokens` | Article, interest, token pieces and IDs | Qwen tokenizer |
| `adapter` | Loss, optimizer steps, changed weights, checkpoint identity | `src/news_training_v3.py` |
| `extraction` | Generated JSON, grounding, unsupported predictions | `src/news_pipeline.py` |
| `linking` | Candidates, learned scores, ID or abstention | `src/learned_linker.py` |
| `filters` | Requested and article filters side by side | `compare_filters` |
| `e5`, `phrases` | Actual vectors and computed similarities | `encode_batch`, `find_e5_phrase_matches` |
| `ranking`, `live` | New interest ranking and fresh generation | `NewsPipeline` |
| `evaluation` | Measured evidence, completed reviews and limitations | `src/news_evaluation.py` |

Explore [extraction training](notebooks/tactical_report_entity_extraction_qlora.ipynb) for the full trainer and [recommendation evaluation](notebooks/tactical_report_recommendation_demo.ipynb) for keyword/E5/hybrid comparisons. The app uses stage IDs, not fragile cell numbers.

## Why These Models

Qwen2.5-3B-Instruct generates extraction JSON. During QLoRA, its frozen base uses 4-bit storage while small added LoRA matrices learn from article/answer examples. It was already downloaded and fits the local budget; it is not claimed to be globally best.

Multilingual E5 independently turns article text and the written interest into comparable 768-number vectors. Arabic and English map directly into a shared space, without manual translation. E5 is **not fine-tuned** here. A separate small candidate-ranking classifier is trained using E5 and name features.

Two specialized models make extraction and retrieval separately testable. Unknown entity links can remain unresolved. Multiple tokens for an Arabic name are normal; fine-tuning does not automatically merge them.

## Proof And Limits

The [evidence snapshot](data/release/evidence.json) comes from actual artifacts, not manually entered scores. Phase 3 selected `extraction-only-v3` and its learned linker under the predeclared validation rules. Phase 4 has indexed all 300 target articles under those identities; this remains a research prototype, not a production-quality claim.

- Training proof: losses, optimizer steps, processed examples, changed adapter weights and hashes. These prove optimization, not sufficient quality. Failed/time-limited stages stay labeled.
- Extraction quality: JSON parsing, schema validity and entity precision/recall/F1 are separate. All 30 review items are complete: ten validation and twenty test articles, fixed before scoring. They are outside **our training**, not necessarily original model pretraining. Machine-assisted training labels are not human ground truth.
- Linker quality: the held-out pool uses published entity IDs and is article-disjoint from development. Read accepted precision together with coverage; unseen-alias performance is weak, so uncertain names remain unresolved.
- Retrieval quality: all 30 judgments are complete across three ten-article pools. Recall@5 and nDCG@5 are within those pools, not full-corpus scores; only one validation query and two test queries support the comparison.
- Integrity: changed text or models invalidate cached results. Failed inference never substitutes manual tags. Pasted visitor text is not persisted into the public corpus or training data.

The original 20-mock-example adapter remains an explicitly labeled historical baseline. The selected real-news adapter passed the project's validation gate, but is not a production model. Twenty-three current-version indexing attempts failed extraction/schema checks and remain preserved, alongside the 194 historical failures. The separate live smoke check passed 28 of 30 articles; both failures remain visible. Training alone does not prove quality. Entity co-occurrence does not prove a relationship. Phrase similarities are probes, not internal reasoning. Public Pages and live inference are not verified yet.

## Reproduce Locally

Clone the repository and enter `1-tactical-recommendation-prototype`. You need code, dependencies and prepared artifacts, not only a notebook. Preserve the existing working environment on this PC; use these steps for a new machine.

Windows/NVIDIA, Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-portfolio.txt bitsandbytes==0.49.2 ipykernel
python -m ipykernel install --user --name news-recommender --display-name "Python (news recommender)"
```

Apple Silicon Mac, inference/notebooks only:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0
python -m pip install -r requirements-portfolio.txt ipykernel
python -m ipykernel install --user --name news-recommender --display-name "Python (news recommender)"
```

Mac inference uses non-quantized CPU/MPS and needs adequate RAM. CUDA QLoRA training is not promised on macOS. ZeroGPU uses a separate deployment PyTorch profile.

Restore the pinned runtime artifact before opening the app. Do not repeat training or review to install the release:

```bash
python scripts/restore_news_bundle.py /path/to/runtime-VERSION.zip --destination .
python portfolio_app.py
```

Verify the archive SHA-256 against its published manifest first. Runtime bundles exclude base models and dependencies; fresh local inference downloads the pinned base models. The separate private evidence bundle retains training checkpoints and the corpus. Until remote artifacts are published and verified, installation is not claimed to be fully reproducible from public links.

The following are historical preparation commands, not release installation steps:

```bash
python scripts/prepare_news.py --target 5000
python scripts/train_linker.py
python scripts/evaluate_linker_holdout.py
python scripts/run_news_experiments.py --examples 100 --steps 24 --label-hours 2
python scripts/evaluate_news.py predict --hours 2
python scripts/build_news_release.py --limit 300 --hours 3
python scripts/prepare_ranking_review.py
python portfolio_app.py
```

Open <http://127.0.0.1:8502>. Slow preparation is local, not required of website visitors. Run stages serially. The GPU lock prevents conflicting jobs; `output/portfolio/processing_budget.json` tracks cumulative bounded processing. Generated artifacts remain outside ordinary Git history.

## Review And Promotion

Set `NEWS_REVIEW_MODE=1`, then run the local app to review thirty extraction examples and thirty relevance judgments. The tab is disabled on the public Space. A real person must supply attributed judgments; the agent cannot fabricate them.

```bash
python scripts/evaluate_news.py score
python scripts/evaluate_ranking.py
python scripts/manage_news_model.py promote --run-id extraction-only-v1
python scripts/build_news_release.py --limit 300 --hours 3
python scripts/package_news_release.py
```

Promotion uses validation only: schema validity >= max(0.8, baseline), strictly improved overall F1, and no language F1 regression. Failed candidates stay unpromoted. Restart after promotion and rebuild the matching index. `python scripts/manage_news_model.py rollback` restores the prior model configuration.

Use `scripts/news_cli.py ingest --article article.json` to process new news. `scripts/approve_news_corrections.py` validates corrections and creates a versioned training dataset, rejecting held-out content. Retraining requires an explicit new run prefix and `--corrections-version`; inference does not teach the model automatically.

## Deployment

GitHub Pages serves the lightweight guide and genuine saved results. Gradio targets free ZeroGPU, subject to account eligibility and quotas: [official documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

Authenticate locally with `hf auth login`, never by putting tokens in chat/code. Run `scripts/publish_news_space.py --repo YOUR_ACCOUNT/multilingual-news-recommender` after the release checks pass. It requests no paid fallback. Public URLs must be verified before claiming completion. See [WORK_IN_FUTURE.md](WORK_IN_FUTURE.md).
