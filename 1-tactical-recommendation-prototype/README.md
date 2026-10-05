# Multilingual News Recommender

**Built with Qwen.** A standalone English/Arabic research and evaluation project: extract article facts, link contextual mentions to entities, and recommend news from a reader's interests. No enterprise affiliation.

Start with **[START_HERE.md](START_HERE.md)** for the visitor route, code, reproduction and honest release status.

[Portfolio website](https://felixhajj.github.io/multilingual-news-recommender/) | [Live models](https://huggingface.co/spaces/felixhajj/multilingual-news-recommender)

```text
article -> Qwen + LoRA -> grounded facts -> learned linker -> exact filter coverage
article + user interest -> pretrained multilingual E5 -> semantic similarity
both contributions -> ranked, traceable recommendations
```

The main application is `python app.py` (or `python portfolio_app.py`) at <http://127.0.0.1:8502>. App, CLI and all three notebooks use `src/news_pipeline.py`. `legacy_app.py` is explicitly historical, not the main product. Existing notebook/folder names are retained to preserve links.

## Evidence

[data/release/evidence.json](data/release/evidence.json) contains the latest exported artifact snapshot; the app's Evidence tab reads current local status. The corpus preparation produced 5,000 historical articles: 3,539 English and 1,461 Arabic. Model-assisted extraction labels are not human gold. A trained checkpoint is not automatically promoted.

See [WORK_IN_FUTURE.md](WORK_IN_FUTURE.md) for separate training, review, indexing and publishing gates. Pending does not mean complete. Legacy mock data and the original adapter remain historical baselines; their scores are not real-news release scores.

## Setup

Use Python 3.12 and a fresh environment on each machine. Follow the visitor guide for PyTorch and artifacts; a notebook alone is insufficient.

```bash
python -m pip install -r requirements-portfolio.txt
python -m unittest discover -s tests -v
python app.py
```

Website visitors do not download models. Local developers download them once. Generated datasets, indexes and new checkpoints stay in ignored `output/portfolio/`, outside ordinary Git history. Do not copy a Windows `.venv` to a Mac.

Qwen2.5-3B-Instruct has a research-only license, not Apache 2.0. See [DATA_AND_MODEL_CARD.md](DATA_AND_MODEL_CARD.md) for rights, provenance and limitations.
