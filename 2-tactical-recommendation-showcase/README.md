# Tactical Report Recommendation Showcase

A lightweight presentation build of the local learning prototype.

The complete local pipeline was run beforehand, and the displayed tokenizer, QLoRA, extraction, multilingual E5, and recommendation outputs were saved to `data/showcase_data.json`. No model is downloaded or loaded by this deployment.

Instead of running:

User + Article
      ↓
Qwen / E5
      ↓
Generate outputs
      ↓
Display results

the showcase simply does:

User + Article
      ↓
Read `showcase_data.json`
      ↓
Display the same previously generated results

## Run locally

```bash
python app.py
```

Then open http://127.0.0.1:8501.