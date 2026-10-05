# Data And Model Card

**Built with Qwen.** Standalone research/evaluation, not operational intelligence or an enterprise endorsement.

## Data

A custom reconstruction of English/Arabic Wikinews revisions referenced by [Mewsli-9](https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli-9.md), from historical 2019 snapshots. This is not the official Mewsli benchmark or a current-news feed.

Each article retains source URL/revision, language, contributor attribution, license and content hash. Unknown publication dates are explicitly marked as revision dates. Historical text uses CC-BY-2.5; contributor lists are available through source/history pages. Cleaned text is modified from original wikitext; templates, markup and boilerplate are removed. Images are not redistributed.

Exact duplicates are removed and detected related versions grouped before splitting. Cross-language grouping is heuristic and may miss translations; the corpus is not certified leakage-free. Selection prioritizes geopolitical keywords and Arabic coverage but includes wider international news and is not country-balanced or comprehensive specialist military data.

Published hyperlinks are partial positive mention/ID labels. Unannotated passages are not negative labels. Qwen-assisted extraction labels pass literal-grounding and positive-overlap checks, but are still imperfect estimates, not human gold. Synthetic legacy fixtures remain separate from real-news evaluations.

## Models And Rights

[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) is retained for storage/hardware practicality. Its [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) is not Apache 2.0: research/evaluation use only, with separate permission needed for commercial use. Redistributed adapters retain the license, notice and modification attribution. Base weights are not recommitted.

[Multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base) is pretrained and unchanged here (MIT per its model card). Cosine similarity is not probability. The linker is a project-trained logistic candidate ranker over E5/name features, using a training-derived Wikidata-ID subset, not a complete knowledge base. Aliases are features, not guaranteed truth.

QLoRA changes added adapter matrices, not Qwen's frozen base or tokenizer. Bounded raw-domain adaptation is an experiment, not a claim of broad geopolitical expertise.

The frozen local extraction/indexing profile uses 4-bit base weights. The ZeroGPU demo loads the same selected adapter over float16 base weights, keeping the execution identity distinct. Hosted English/Arabic synthetic smoke tests verify functionality, not quality parity with the frozen local evaluation. Generic-official person labels and a city incorrectly labeled as a country occurred in those tests; generated output is not reliable ground truth.

## Evaluation And Failures

Extraction review: five English/five Arabic validation examples and ten English/ten Arabic final test examples, roles fixed before review. Test scores do not select deployment. Surface entity F1 does not measure relationship accuracy, factual truth or every alias. Relationships are unreviewed and not used to establish recommendation constraints.

The linker's initial development test was inspected while improving features. A fresh 200-mention article-disjoint holdout was frozen afterward with fixed weights. Report accepted precision alongside coverage, catalogue reach and weak unseen-alias performance.

Recommendation evaluation covers three ten-article judged pools, not full-corpus recall or general personalization. There is no click tracking, online reinforcement learning or implicit user profiling.

Exploratory visitor queries can rank off-topic articles: a European energy/gas query ranked a US pipeline story first and a Google-slander story second. Some cleaned excerpts still contain residual wikitext. These failures are separate from the frozen evaluation and were not used to retune the selected models or test pools.

Input text is untrusted data. Grounding/schema checks reduce unsupported outputs but cannot eliminate hallucination, bias, ambiguity or prompt injection. Failures and unknowns remain visible. Use public text or text you may process. Visitor input is not added to public indexes or training datasets.

## Resources

Existing model caches are reused. Additional corpus, adapters and indexes stay outside ordinary Git history. The original 24-hour cumulative processing cap was explicitly extended to 32 hours; previous charges remain intact. Actual optimizer steps and measured runtimes are separate evidence; allocated time is not completed learning. Startup delays and free hosted quotas remain limitations.
