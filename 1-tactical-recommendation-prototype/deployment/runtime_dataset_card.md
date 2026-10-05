---
pretty_name: Multilingual News Recommender Runtime
language:
- en
- ar
license: other
license_name: mixed-research-and-attribution
license_link: https://github.com/felixhajj/multilingual-news-recommender/blob/main/1-tactical-recommendation-prototype/DATA_AND_MODEL_CARD.md
---

# Multilingual News Recommender Runtime

**Built with Qwen. Research/evaluation only.** This repository contains versioned,
checksummed runtime archives, not a current-news feed or production model.

[Visitor guide](https://felixhajj.github.io/multilingual-news-recommender/guide.html) |
[Live demo](https://huggingface.co/spaces/felixhajj/multilingual-news-recommender) |
[Source code](https://github.com/felixhajj/multilingual-news-recommender)

`release.json` identifies the current archive by immutable Hub commit, complete
version hash and archive SHA-256. The standard-library downloader verifies it;
restoration verifies every member. Historical versions are retained, not overwritten.

The approximately 123 MiB runtime includes 300 processed historical articles
(156 Arabic, 144 English), normalized vectors, article attribution, the selected
adapter/linker and small evidence required by the frozen selection checks. It
excludes downloadable base weights, Python environments, the full 5,000-article
working corpus and intermediate training checkpoints. Those research artifacts
are preserved in the owner's separately verified private archive.

News text is cleaned/modified historical Wikinews material under CC-BY-2.5;
source/history URLs provide contributor attribution. No images are redistributed.
The Qwen adapter is governed by the Qwen Research License, **not Apache 2.0 or a
blanket commercial license**. Included license notices and the model/data card
describe the mixed rights; source code and article/model licenses are not interchangeable.

Frozen extraction test: 20 reviewed articles, entity micro-F1 0.313, schema validity
0.95. Retrieval test: two ten-article pools, hybrid and E5 nDCG@5 0.956 versus keyword
0.612, Recall@5 0.5 for all. These small pools do not establish general superiority.
The linker has limited catalogue reach and weak unseen-alias performance.

Saved metrics can be replayed without downloading models. Fresh inference needs
model weights locally or the hosted demo, which uses the same adapter over a
distinct float16 execution profile. Hosted functionality is not a quality-parity claim.
