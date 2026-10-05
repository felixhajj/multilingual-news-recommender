# Results and Limitations

This project combines Qwen extraction, a supervised entity linker and multilingual E5 retrieval in a working English-Arabic news recommender. It is an **experimental portfolio prototype, not a production-ready or broadly validated recommender**. The limitations below reflect measured quality and small training/evaluation samples, not training duration alone.

[Try the demo](https://huggingface.co/spaces/felixhajj/multilingual-news-recommender) | [Read the six-page walkthrough](walkthrough.pdf) | [Explore the code](https://github.com/felixhajj/multilingual-news-recommender)

## What Was Built

The corpus contains **5,000 historical Wikinews articles**: 3,539 English and 1,461 Arabic. The indexed release contains **300 successfully analyzed articles**: 144 English and 156 Arabic. Collection size and indexed release size describe different stages of the work.

The selected QLoRA adapter trained for **48 optimizer steps on 30 machine-assisted examples**. Human review supplied 30 extraction references and 30 recommendation judgments for evaluation. Training labels and human-reviewed references have separate provenance.

The app, scripts and three notebooks share the extraction, linking, embedding and ranking implementation. Recorded checks include 143 passing regression tests, execution of all three notebooks, fresh hosted English/Arabic article analysis and changed rankings for changed interests.

<h2 id="training-scope-and-time">Training Scope and Time</h2>

The selected Qwen adapter used **30 machine-assisted examples and 48 optimizer updates**. The 5,000 collected articles are not 5,000 supervised training examples; 300 articles were analyzed for the release index.

**The model-development work spanned hours, not just the minutes in the final adapter's timer.** Seven archived Phase 3 worker sessions alone span about **2.4 hours of process runtime**, including unsuccessful loading attempts and recovery. Other stages recorded training and evaluation separately. These worker durations are not a measure of successful GPU optimization time or a complete total for all experiments.

The selected adapter's saved function-session timers total **580.063 seconds (about 9.7 minutes)**, including model loading, training, checkpoint handling and saving. This is **a partial timing record, not an estimate of all training**: forced stops can prevent durations from being saved, and imports occur before this timer starts. The exact complete training-only duration is therefore unavailable. Qwen's original pretraining is not included; E5 was used pretrained and was not retrained. [Inspect the timing evidence and its scope](https://felixhajj.github.io/multilingual-news-recommender/training-summary.json).

More collected articles can improve coverage, but do not automatically fix extraction or ranking quality. Better supervised labels and broader held-out evaluation are needed before making stronger reliability claims.

## Extraction

Qwen generates structured article facts. JSON parsing checks whether the output is readable JSON; schema validity checks its expected structure; entity F1 measures how well extracted names match the reference names.

| Final test: 20 human-reviewed articles | Result |
| --- | --- |
| JSON parsing validity | 100% |
| Schema validity | 95% |
| Entity precision | 49.0% |
| Entity recall | 23.0% |
| Entity micro-F1 | 0.313 |

The sample contains ten English and ten Arabic articles, held out from our training. It may overlap with pretrained model data. F1 scores exact normalized entity surfaces; topics and relationships are checked against the schema but are not human-scored here.

Validation selected extraction-only training over base Qwen and domain-adaptation-plus-extraction. On ten validation articles, entity F1 rose from 0.289 for base Qwen to 0.372 for the selected adapter, while schema validity rose from 40% to 90%. These validation results are distinct from the final test above.

## Entity Linking

A trained classifier chooses among E5-retrieved candidate entities. It can leave a name unresolved when evidence is weak.

| Article-disjoint holdout: 200 mentions | Result |
| --- | --- |
| Correct accepted links | 95.9% precision |
| Mentions receiving an accepted link | 36.5% coverage |
| Accepted links | 73 of 200 |

High precision applies only to the accepted subset. This previously reported article-disjoint holdout was reused for the release, rather than collected afresh. The catalogue is a training-derived Wikidata subset, and many names remain unresolved. Performance on unseen or ambiguous aliases is weak. This is a limited contextual linker.

## Recommendation

Keyword search, E5 and hybrid ranking were compared on the same judged pools. Keyword and E5 rank all ten candidates per pool; hybrid ranks only eight and nine, because extraction failures exclude some candidates. nDCG@5 rewards placing more relevant items near the top; Recall@5 measures how many relevant items appear in the first five results.

| Final test: two queries, ten judged articles per query | nDCG@5 | Recall@5 |
| --- | --- | --- |
| Keyword search | 0.612 | 0.500 |
| Multilingual E5 | 0.956 | 0.500 |
| Hybrid: E5 plus optional filter signal | 0.956 | 0.500 |

These are small, fixed-pool results. Hybrid and E5 have equal reported averages; hybrid has not demonstrated an improvement over E5 on this test. The model choice was fixed using validation before final-test reporting.

## Where It Still Fails

The local new-article smoke check produced valid outputs for 28 of 30 articles. Both malformed outputs were retained. Hosted examples included generic officials classified as people and Beirut classified as a country. An exploratory European energy/gas interest ranked a US pipeline article first and an unrelated Google-slander story second. Some excerpts retain wikitext.

E5 scores the written interest against natural article text, not by checking individual matching words. Phrase similarities are illustrative probes, rather than a definitive explanation of the model's internal reasoning. Scores indicate relative relevance. Historical source articles and their claims are not independently fact-checked by this application.

Frozen extraction metrics use local 4-bit Qwen. Hosted inference uses the same adapter with float16 base weights; hosted quality parity has not been measured. Free GPU quotas and startup delays can prevent fresh analysis, while the website and recorded examples remain readable.

## Inspect or Reproduce

[The technical guide](guide.html) links the three notebooks and commands for replaying saved metrics in a clean environment. The [machine-readable evidence](evidence.json) records model identities and source hashes. [Data and model details](model-card.html) cover provenance, execution profiles and licensing.

Wikinews text retains contributor attribution under CC-BY-2.5. Qwen2.5-3B-Instruct uses the Qwen Research License; this release is for research and evaluation.
