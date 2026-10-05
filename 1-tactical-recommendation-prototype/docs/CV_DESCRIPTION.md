# Portfolio Description

## Short CV Entry

Built an English-Arabic news recommendation research prototype using Qwen QLoRA extraction, a supervised entity linker, multilingual E5 embeddings and versioned SQLite indexing. Collected 5,000 attributed news articles and processed a 300-article release; implemented human-reviewed evaluation, cache invalidation and reproducible artifact bundles.

## Evidence For An Interview

- Extraction-only adapter: 48 optimizer steps on 30 machine-assisted examples, with saved losses and changed-weight evidence.
- Held-out extraction test: 20 human-reviewed articles, entity micro-F1 0.313 and schema validity 0.95. This is limited extraction quality, not production accuracy.
- Recommendation test: two fixed ten-article pools; hybrid nDCG@5 0.956 versus keyword 0.612. Recall@5 was 0.5 for both. These tiny pools do not establish full-corpus superiority.
- Local live-input smoke: 28 of 30 valid outputs; failures retained rather than replaced with static tags.
- Runtime restore reproduced saved metrics without access to original data files or model inference. Python dependencies were reused, so a fresh dependency installation is a separate check.

Public deployment is not verified yet. Do not claim a working public demo until its recorded remote checks pass. Qwen's research license and source attribution apply; this is a research/evaluation project without enterprise affiliation.
