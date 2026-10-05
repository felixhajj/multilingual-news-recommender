# Portfolio Release Checklist

Execution baseline: [phase plan](PHASE_PLAN.md). Phases 1-4 are complete;
Phase 5 packaging/publication is in progress. Concrete repairs and acceptance checks are in
[the Phase 1 audit](docs/PHASE_1_AUDIT.md). Implemented does not mean release-verified.

This replaces the old open-ended enterprise backlog. Implemented code, evaluated experiments and completed release gates are different statuses. See the live Evidence tab for current artifact counts.

| Item | Status / remaining gate |
| --- | --- |
| Standalone product, free-text interest, explicit model roles | Implemented in the new Gradio app |
| 5,000 real articles and provenance | Collected: 3,539 EN / 1,461 AR |
| Cleaning, duplicate handling and grouped splits | Implemented; translation grouping remains heuristic |
| Up to 500 grounded article/JSON examples | Experimentally evaluated: 30 accepted weak examples from 160 attempts; not 500 human-gold examples |
| Raw-domain QLoRA plus extraction ablations | Evaluated and selection locked; extraction-only-v3 selected after 48 optimizer steps; failed experiments preserved |
| Learned linker replaces primary manual lookup | Trained and evaluated on fresh 200-mention holdout; limited unseen-alias coverage |
| Contextual canonical IDs and unresolved mentions | Implemented with a limited training-derived catalogue |
| Dynamic prototype linking and extraction for arbitrary articles | Implemented; no prewritten tag fallback on failure |
| Shared analyze/recommend/explain operations | Implemented in `src/news_pipeline.py` |
| SQLite, compact vectors, content/model invalidation | Implemented and locally verified, including edited-text and model-identity checks |
| 300 processed release articles | Verified: 300 selected-v3 records, 156 Arabic / 144 English; 23 current and 194 historical failed attempts preserved |
| Dynamic E5 supporting phrase similarities | Implemented; explicitly not causal explanations |
| Tokenizer demonstration with unchanged vocabulary | Implemented in A-to-Z notebook |
| Approved corrections, explicit retraining, rollback | Implemented with frozen-reference integrity, validation selection and matching-index activation gates |
| Three explained notebooks and stable stage links | Verified default runs; live inference and training remain explicit opt-ins |
| Visitor guide and lightweight landing page | Published on GitHub Pages; desktop/mobile route verified |
| Thirty extraction and thirty recommendation human reviews | Complete; original identities, attribution and frozen splits preserved |
| Human-reviewed F1, Recall@5 and nDCG@5 | Evaluated; isolated runtime reproduces saved metrics. Only 20 extraction test articles and two recommendation test queries |
| Candidate model promotion | Validation-selected extraction-only-v3; research release, not production-quality claim |
| Free ZeroGPU deployment | Running; fresh English/Arabic generation, changed interests and edited-input cache identity verified remotely |
| Desktop/mobile and failure-state verification | Hosted browser checks passed with local server stopped; failure clears stale output and unlocks controls; physical second-device confirmation pending |
| Portable artifact restoration | Lightweight clean dependency installation reproduces saved metrics; final pinned publication receipt is in the Phase 5 handoff |
| CV-ready final completion | Technical verification recorded in the Phase 5 handoff; do not claim physical second-device testing or bulk cleanup before their gates pass |

## Outside This Release

Large-scale domain pretraining; additional Global Voices coverage after reaching the source target; tokenizer/vocabulary expansion; separate military/person-name adapters; E5 fine-tuning without sufficient relevance data; production accuracy guarantees; continuous current-news crawling; automatic learning from predictions; commercial licensing assumptions; enterprise integration.

Historical mock metrics and static showcase files are preserved for comparison, not used to conceal unsuccessful real-news experiments. Original ideas remain in `docs/history/ORIGINAL_BACKLOG.md`.
