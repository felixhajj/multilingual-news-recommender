# Phase 4 Handoff: Application and Walkthrough

Completed October 2, 2026. Phase 4 is complete for the local application and
walkthrough. The saved audit reports `phase4_complete_with_model_failures_preserved`;
it does not claim that every model response succeeded or that the public service
is deployed.

## What Passed

- The selected `extraction-only-v3` pipeline is active in the shared app and index.
  The release index contains 300 articles (156 Arabic, 144 English), with pipeline
  identity `7bb217afa48f24dc5134c8d30d990c008fe25fd53e710d1cf9e3df594149d14c`.
- Local browser checks passed for English and Arabic recommendations; the visible
  cards matched the pipeline output. Distinct interests produced different
  top-five rankings. The 390-pixel mobile viewport had no horizontal overflow.
- Fresh English and Arabic article inference completed in the app. An identical
  input hit the cache; an edited title missed it. Unit coverage verifies edited
  article text and a changed model revision also invalidate cached analysis.
- The separate 30-article integration smoke batch accounted for every input:
  28 generated successfully; two malformed model responses were retained with
  their raw outputs. These are unlabelled smoke inputs, not extraction metrics.
- All six app completion checks passed, including unchanged release-database
  checksum, visible error handling, interest-driven reranking and notebook status.
- All three notebooks passed default-cell execution. Saved outputs and source
  fingerprints were reverified. Their optional live-generation and training cells
  were not run; fresh inference was instead exercised through the application.
- Selected-pipeline ranking metrics were recalculated from saved model analyses,
  vectors and the unchanged 30 human-reviewed judgments (27 unique articles).
  On the two test queries, keyword scored Recall@5 0.500 / nDCG@5 0.612; E5 and
  hybrid each scored Recall@5 0.500 / nDCG@5 0.956. This is a small judged pool,
  not corpus-wide evidence or a claim that hybrid beats E5. Four failed extractions
  in these judged pools remain explicit and are excluded from hybrid candidates.
- The final regression suite passed 128 tests. It covers unknown/unresolved names,
  malformed output, unavailable models, empty indexes, changed interests, changed
  text/model identities, and stale-index behavior.

## Evidence And Reproduction

- `output/portfolio/phase4/completion.json` is the final cross-artifact audit.
- `output/portfolio/phase4/live_app_check/report.json` and `progress.json` contain
  the local UI, multilingual inference, cache and smoke results.
- `output/portfolio/phase4/live_app_check/failures.jsonl` preserves the two failed
  smoke responses and raw model output.
- `output/portfolio/phase4/selected_ranking/report_v2.json` contains the measured
  validation/test metrics; `manifest.json`, `analyses.jsonl` and `vectors.npz`
  bind them to the frozen inputs and selected model.
- `output/portfolio/notebook_validation/report.json` and its executed notebooks
  preserve the notebook results and hashes.
- `output/portfolio/phase4/live_app_check/unit-tests-20261002-final.log` records
  the 128-test run.

From the project root, verify the saved evidence with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_phase4_completion.py
```

The verifier recalculates ranking metrics from saved vectors/references and checks
the release DB hash, notebook fingerprints, smoke accounting, test log and budget;
it does not load Qwen or rerun article generation.

## Remaining Work

Phase 5 is next: build runtime/evidence bundles, verify clean restoration and
off-PC backup, publish the guide and live demo, and test them from a fresh remote
visitor session. Public deployment is not verified yet. Phase 6 storage cleanup
must wait until Phase 5 restoration and backups pass. The 32-hour cumulative budget
has 17,144.987 seconds remaining; historical charges remain intact.
