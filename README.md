# Multilingual News Recommender

English/Arabic news recommendation with Qwen + QLoRA extraction, a learned entity linker, and multilingual E5 retrieval. **Built with Qwen.** A standalone research/evaluation portfolio, not an enterprise product.

**[Start here: visitor guide and reproduction](1-tactical-recommendation-prototype/START_HERE.md)**

## Projects

- `1-tactical-recommendation-prototype/`: notebooks, Qwen + LoRA extraction, E5 semantic matching, ranking logic, tests, and the local API prototype.
- `2-tactical-recommendation-showcase/`: historical static showcase, no longer the main product.

The active app, CLI and three explained notebooks share the real-news pipeline. Measured results and pending release gates are recorded in the visitor guide and `data/release/evidence.json`. Pending human review and unpublished endpoints are not claimed as complete.

## Run the prototype

```powershell
cd 1-tactical-recommendation-prototype
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:8501`.

The virtual environment, Hugging Face caches, and large training checkpoints are excluded from Git. Install dependencies from the project requirements files and download model artifacts locally when needed.
