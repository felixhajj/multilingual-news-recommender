# Phase 4 Index Completion Check

Historical checkpoint from October 1, 2026. The app, notebook and recommendation
checks listed as remaining at that point were subsequently completed; see the
[final Phase 4 handoff](PHASE_4_HANDOFF.md) and
[`completion.json`](../output/portfolio/phase4/completion.json).

Checked October 1, 2026 after the supervised worker exited with code 0 at
16:12 Beirut. No worker remained active.

- Current pipeline: `7bb217afa48f24dc5134c8d30d990c008fe25fd53e710d1cf9e3df594149d14c`.
- Extractor: `extraction-only-v3`; model bytes and selected configuration match.
- Prepared collection: 300 records, 156 Arabic and 144 English.
- Continuation: 209 additional successes, 9,506.547 seconds including final exports.
- Current failures: 23 attempts, including nine new failures; legacy 194 attempts remain.
- Budget: 95,791.155/115,200 seconds charged, 19,408.845 seconds remaining.
- D: free space at check: 29.83 GiB. No data or model caches were deleted.

Read-only SQLite verification checked every current record's ready status, full
model/pipeline identity, content-addressed cache key, source grounding offsets,
and normalized finite 768-dimensional vector. All 393 stored raw model-output
chunks passed their declared extraction schema. No integrity error was found.
Three articles contain no grounded mentions; this is a real model outcome and
must not be represented as human-written tags or complete extraction coverage.

Tokenizer-only verification of the complete collection found 405 E5 chunks,
maximum 453 encoded tokens, with zero chunks over the model's 512-token limit.
This check did not load model weights, recompute vectors or change encoder behavior.

This completes the indexing requirement. Phase 4 still needs isolated selected-
pipeline recommendation verification on the fixed judged pools, executed notebooks,
live unseen-input/cache checks, and desktop/mobile presentation checks. Phase 5
publishing and Phase 6 verified cleanup remain pending.
