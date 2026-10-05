# Portfolio Description

## Short CV Entry

Built and deployed an English-Arabic news recommendation research prototype using Qwen QLoRA extraction, a supervised entity linker, multilingual E5 embeddings and versioned SQLite indexing. Collected 5,000 attributed news articles and processed a 300-article release; implemented human-reviewed evaluation, cache invalidation and checksummed, reproducible artifact bundles.

[Portfolio](https://felixhajj.github.io/multilingual-news-recommender/) | [Live demo](https://huggingface.co/spaces/felixhajj/multilingual-news-recommender) | [Code](https://github.com/felixhajj/multilingual-news-recommender)

## Evidence For An Interview

- Extraction-only adapter: 48 optimizer steps on 30 machine-assisted examples, with saved losses and changed-weight evidence.
- Held-out extraction test: 20 human-reviewed articles, entity micro-F1 0.313 and schema validity 0.95. This is limited extraction quality, not production accuracy.
- Recommendation test: two fixed ten-article pools; hybrid nDCG@5 0.956 versus keyword 0.612. Recall@5 was 0.5 for both. These tiny pools do not establish full-corpus superiority.
- Local live-input smoke: 28 of 30 valid outputs; failures retained rather than replaced with static tags.
- Isolated runtime restoration with a fresh lightweight dependency environment reproduced saved metrics without original working files or model weights. This is evidence replay, not a new training or inference result.
- Hosted ZeroGPU checks generated new English/Arabic synthetic inputs and changed rankings for new interests with the local server stopped. Synthetic inputs test functionality, not accuracy; extraction mistakes remain visible.

Qwen's research license and source attribution apply; this is a research/evaluation project without enterprise affiliation. Hosted generation uses float16 base weights with the same selected adapter; frozen quality metrics use the local 4-bit profile. Do not claim quality parity or production reliability. Free GPU quotas and cold starts can temporarily prevent inference; the guide and measured results remain readable.
