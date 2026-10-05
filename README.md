# Multilingual News Recommender

English/Arabic news recommendation with Qwen + QLoRA extraction, a learned entity linker, and multilingual E5 retrieval. **Built with Qwen.** A standalone research/evaluation portfolio, not an enterprise product.

**[Start here: visitor guide and reproduction](1-tactical-recommendation-prototype/START_HERE.md)**

[Portfolio website](https://felixhajj.github.io/multilingual-news-recommender/) | [Live models](https://huggingface.co/spaces/felixhajj/multilingual-news-recommender)

## Projects

- `1-tactical-recommendation-prototype/`: notebooks, Qwen + LoRA extraction, E5 semantic matching, ranking logic, tests, and the local API prototype.
- Historical showcase code is preserved separately in the original private project, not published as the release.

The active app, CLI and three explained notebooks share the real-news pipeline. The 300-article release has 156 Arabic and 144 English articles. Measured results and limitations are in the visitor guide and `data/release/evidence.json`: twenty reviewed extraction test articles and two tiny recommendation test pools do not establish production accuracy.

## Run the prototype

```powershell
cd 1-tactical-recommendation-prototype
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:8502`. First restore the checksummed runtime as explained in the guide. Visitors use the hosted app without installing models.

The virtual environment, Hugging Face caches, and large training checkpoints are excluded from Git. Install dependencies from the project requirements files and download model artifacts locally when needed.
