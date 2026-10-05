# Phase 1 Audit and Repair Specification

September 16, 2026. Phase 1 is complete; the release is not. Authoritative sequence:
[PHASE_PLAN.md](../PHASE_PLAN.md). Paths below are relative to the project root.

## Evidence and Limits

Run `python scripts/audit_news_baseline.py --compare data/release/phase_1_audit_verified.json`
to verify the protected baseline. Run without `--compare` after intentional,
versioned changes; preserve the old snapshot and explain each changed hash.
The audit imports no model code, uses read-only SQLite, counts/hashes prediction
files without parsing them, and verifies no protected file changed during its run.

`python -m unittest discover -s tests -v`: **55 passed**, September 16. These include
test-double pipeline behavior, not proof of current real-Qwen end-to-end quality.
No Python worker was present in the process check. No new GPU inference/training
was launched. Default notebook execution was previously recorded as successful;
it was not repeated or treated as live-model verification in this audit.

Protected SHA-256 inventories are in `data/release/phase_1_audit.json` and
`data/release/phase_1_audit_verified.json`. The second snapshot adds current-index
aggregates and confirms the same protected files. Source files and docs are not
covered by that model/data inventory; pre-existing Git changes were preserved.

| Artifact | Verified state | Meaning |
| --- | --- | --- |
| `output/portfolio/corpus.jsonl` | 5,000; EN 3,539 / AR 1,461 | Historical Wikinews reconstruction, not an official Mewsli benchmark run |
| Corpus splits | Train 3,983 / validation 498 / test 519 | No duplicate stored body hashes, duplicate IDs or cross-split stored groups |
| `extraction_training.jsonl` | 26; EN 17 / AR 9 | Machine-assisted, not human gold; no literal grounding/hash/train-review overlap failures detected |
| Nonempty training fields | People 16; countries 8; organizations 7; companies 2; systems 0; topics 26 | Small and poorly balanced for the intended task |
| `runs/*/manifest.json` | Three completed runs, 24 updates each, adapter hashes match | Weight changes prove optimization, not quality improvement |
| SFT runs | 26 examples, 13,512 supervised tokens each | Same small dataset for both ablations |
| Domain run | 500 input articles; 90 actually seen; 67,225 tokens | Do not claim training on all 500 |
| `evaluation/*/predictions.jsonl` | Three files, 30 opaque lines each | Contents deliberately not inspected for Phase 1 decisions |
| `review/` | 30 extraction + 30 ranking pending | No human-reviewed extraction/retrieval quality claim yet |
| `release.sqlite` | 300 current + 7 old-version rows; quick_check OK | Only current identity counts toward release |
| Current 300 | EN 223 / AR 77; 52 without grounded mentions | Inference completion does not mean extraction success or good recommendations |
| `release_failures.jsonl` | 194 logged attempts | Failure coverage must accompany success-only release counts |
| `data/release/deployment.json` | Public verification pending | Not a finished public portfolio |

Stored duplicate groups do not prove that every translated/related article was
detected. Frozen extraction-review IDs are excluded from the release index and
their groups do not overlap the 26 training examples. All frozen recommendation
article IDs are present in the current index; preserve that property across models.

## D1: Durable Label Preparation (Phase 2)

Location: `src/news_training.py::prepare_training_examples`.

The current log contains 207 rejected attempts: 120 nonliteral labels, 40 missing
or extra schema keys, 24 invalid JSON, 15 missing agreement with published positive
mentions, and 8 malformed relationships. This is **not necessarily cumulative**:
the preparer starts a new rejection list on resume, overwrites the log, remembers
only accepted IDs, and drops rejection totals from its final summary. The language
interleaving uses `zip`, omitting the longer language list's tail.

Repair: append/version attempt records with article/content/group IDs, language,
teacher/prompt/schema identities, raw output where available, duration and explicit
reason. Resume on complete attempt identity, with bounded explicit retry policy;
changed prompt/model gets a new version, not an overwrite. Preserve the 26 examples
and v1 run inputs. Alternate languages without discarding remaining candidates.

Acceptance: interruption/resume cannot duplicate accepted examples or erase
rejections; unchanged failures are not retried forever; all candidate tails are
reachable; totals and elapsed time reconcile. Add synthetic unit fixtures for
these behaviors before a budgeted training-only labeling job.

## D2: Grounding, Completeness and Coverage (Phase 2)

Locations: `src/news_training.py`, `src/news_corpus.py`,
`scripts/approve_news_corrections.py`, `scripts/run_news_experiments.py`.

The 26 examples are literally grounded but not verified exhaustive. Published
hyperlinks only check positive agreement. They cannot prove absent companies,
systems or people; empty arrays still become supervised negative labels in SFT.
Zero equipment examples and only two company examples justify targeted training
candidate selection, not an arbitrary demand to reach 500 at any quality.

Repair: select additional candidates from existing **train groups only**, stratify
language and underrepresented entity types, and retain source passages, provenance
and field-level checking status. Audit training examples against published positive
mentions and source text. Reject or quarantine unresolved incomplete annotations
for exhaustive SFT; never fill absent annotations with empty gold arrays. A
separate explicitly weak-label experiment is permitted only with its own identity
and honest label-source reporting. Do not fabricate a human completeness check.

Keep strict seven-field output validation and literal entity grounding. Do not
silently discard malformed relationships or infer missing JSON keys to improve
acceptance. Sol can implement versioning/checks; changes to labeling semantics,
partial-loss supervision or extraction schema require Astra's decision.

Acceptance: versioned dataset <=500 examples, unique IDs, zero frozen-group
overlap, matching source hashes, preserved license/source metadata, raw teacher
identity, meaningful coverage report and no claim of human gold for machine labels.
Report missing coverage honestly if the corpus/budget cannot supply it. Do not
redownload 5,000 articles or alter corpus splits to make sampling easier.

## D3: Review Integrity and Usability (Phase 2)

Locations: `src/news_evaluation.py::validate_review`, `review_roles`,
`validate_ranking_review`; `output/portfolio/review/`.

Extraction validation hashes body text but not the title, even though inference
consumes both. Set equality also permits duplicate records; existing role files
are checked for IDs, not a frozen role mapping. Fix this before review import.

Preserve existing 30 identities and roles. Add a supplement manifest binding title,
body, language, group, source, role and input/prompt identity without reselecting
examples. Reject duplicate/unknown/missing IDs, title/body edits and role changes.
Present source and clear category instructions without model answers or rankings
that could bias review. Import the person's labels/grades with attribution and
timestamps. Validate exact source spans and duplicate judgments; reject boolean
grades masquerading as integer relevance. Never set human_reviewed automatically.

Acceptance: 30 extraction references and 30 graded query/article judgments actually
reviewed; validation 10 / test 20 preserved for each batch. Ranking has three
queries, not 30 independent users: reports must state that small evaluation scope.
Any extra Arabic-interest or edge-case demos stay separate from the frozen metrics.

## M1: Validation Before Final Test (Phase 3, Blocking)

Locations: `scripts/evaluate_news.py::score`, `scripts/evaluate_ranking.py`,
`scripts/manage_news_model.py::promote`.

The extraction scorer currently computes and prints validation **and test** in one
operation before promotion. Do not run it as-is to select a model. Implement a
validation-only command/report and an explicit selection-lock artifact. Final-test
scoring must require the lock and exact references, roles, model, prompt and
prediction identities. Promotion should read validation evidence, not require a
report that already exposed test scores. Support explicit immutable run IDs rather
than only hard-coded v1 prediction runs.

Acceptance: a test-only prediction perturbation cannot affect validation selection;
test reporting refuses a missing/stale lock; validation output contains no test
metrics or examples. Reuse the 90 saved predictions if their identities still
match. If generation configuration changes, create separate evaluation artifacts.
Keep JSON parsing validity, schema validity and entity precision/recall/F1 separate;
do not change existing strict scoring to make malformed generations look correct.

## M2: Identity-Bound Promotion and Release (Phase 3, Blocking)

Locations: `src/news_pipeline.py::QwenBackend`, `src/embeddings.py`,
`src/learned_linker.py`, `scripts/manage_news_model.py`,
`scripts/package_news_release.py::release_gates`.

Promotion checks adapter weights and candidate prediction hashes, but not all
configuration/base/prompt/role/baseline identities. Packaging largely checks
reported statuses and run IDs instead of validating their full evidence chain.
Runtime resolves the current Qwen base config rather than explicitly restoring
the training revision. Encoder environment overrides can disagree with the fixed
embedding identity reported by the pipeline.

Repair: bind base and tokenizer revisions, adapter weights/config, prompt, schema,
chunking/generation settings, actual encoder/revision, linker model/catalogue/vector
hashes and ranking configuration. Validate matching reference/prediction hashes for
baseline and candidate, and current database/index identity/counts at promotion,
package and restore. Reject unsupported overrides instead of mislabeling them.

Keep the intentional local NF4 versus hosted FP16 distinction: the existing index
may be served when all core identities match and its original inference execution
mode remains visible. Fresh inference gets its actual execution identity. Do not
erase this provenance or unnecessarily recompute all articles on every host start.

Acceptance: tests tampering with each identity/evidence component fail closed;
unchanged artifacts reuse valid cache; unrelated old index versions are excluded.

## M3: Bounded Experiments and Failure Diagnosis (Phase 3)

Locations: `src/news_training.py::train_adapter`, `scripts/run_news_experiments.py`,
`scripts/build_news_release.py`, `output/portfolio/processing_budget.json`.

Of 194 indexing failures, 122 fail relationship structure, 40 JSON parsing, 31
schema keys and one systems-array type. The current all-or-nothing schema validator
aborts valid entity fields if a relationship is malformed; relationships themselves
are not used for ranking. The release failure log lacks raw output, so these counts
identify the failure boundary, not the full linguistic root cause.

First inspect training/validation failures with raw generation and truncation/chunk
metadata. Keep strict validity metrics and failed generations visible. If Astra
chooses partial operational output, it must have an explicit partial/failed status,
not count as schema-valid or silently enter the fully processed release set.
Do not tune prompts against final-test predictions. Empty grounded output must
remain visibly empty, never filled with catalogue tags.

Training currently saves final weights only; an interrupted optimizer run cannot
resume its optimizer/RNG/data cursor. Logs show a microbatch loss, and weight-change
evidence tracks one LoRA tensor, not every parameter. Preserve these historical
facts. Before any new justified run, add durable optimizer/scheduler-if-used/RNG/
step/data-cursor checkpoints, dataset identity validation, dropped-length IDs,
per-run peak memory reset and clearly defined loss aggregation.

Acceptance: controlled interruption resumes without duplicate updates or budget
reset; changed dataset rejects resume; record actual articles/tokens/steps/time.
Reconcile unmetered early processing before spending the ledger's nominal 17.26
remaining hours. Do not repeat completed optimization simply to fill a phase.

## M4: Learned Linking and Fair Retrieval (Phase 3)

Locations: `src/learned_linker.py`, `scripts/train_linker.py`,
`scripts/evaluate_linker_holdout.py`, `scripts/evaluate_ranking.py`.

Previously reported frozen linker results are historical evidence, not tuning
inputs: 70 correct of 73 accepted out of 200 mentions (95.9% accepted precision,
36.5% coverage, 35% overall correct versus 33.5% alias baseline). The reported
unseen/ambiguous subset was weak; all 114 out-of-KB mentions abstained. Do not call
accepted precision overall accuracy. This audit did not inspect held-out errors.

On training/validation only, measure candidate recall@k, KB coverage, accepted
precision/coverage, unseen aliases, EN/AR and out-of-KB abstention. Diagnose candidate
retrieval separately from ranking/thresholding before changing either. Define and
lock validation acceptance criteria before another final report. Preserve the
already evaluated linker version and its results; document any subsequent final
set reuse and do not pretend previously inspected data became a fresh holdout.

Compare keyword, E5 and hybrid on identical judged pools. A changed index must
include the same frozen IDs rather than selecting whichever articles now succeed.
Failed inference is reported, not replaced by another judged article. Keep model
selection and weighting validation-only, final-test judgments sealed until lock.
Report Recall@5/nDCG@5 as small pooled experiments, not broad production accuracy.

## A1: Failed Updates and Rollback (Phase 4, Blocking)

Locations: `src/news_store.py`, `src/news_pipeline.py::analyze_article`.

Confirmed with existing test doubles in a temporary database: persist article A,
edit its body under the same ID, make generation raise Invalid JSON, then recommend.
The old A is still returned. Cache keys change on successful inference, but failed
updates never invalidate the old indexed row. No real corpus record was edited.

Implement a current-content/version state with pending/failed status or equivalent
tombstone. On an explicit persisted update, hide the superseded row until the new
analysis succeeds. Preserve historical analyses and versioned vectors or a
checksummed restorable index for rollback. Nonpersistent visitor input must not
invalidate the stored collection. Do not delete historical evidence.

Acceptance: failed persisted update excludes stale content; successful retry
restores it with the new hash; nonpersistent input leaves collection unchanged;
rollback restores the matching model and index without regenerating judged pools.

## A2-A3: Integration and Demonstration (Phase 4)

Locations: `portfolio_app.py`, `src/news_pipeline.py`, `scripts/news_cli.py`,
`scripts/build_portfolio_notebooks.py`, the three `notebooks/`, `START_HERE.md`.

Synchronize the selected identity across all entry points and index/examples.
Reuse shared analyze/recommend/explain code. Make the 52 zero-grounding records
visible as empty extraction, distinct from extraction exceptions and unresolved
entity IDs. Validate article inputs as strings instead of coercing arbitrary objects
to strings. Show source URL, author, license and date_kind, not a revision date
misrepresented as publication date. Required filters mean OR within a category,
AND across categories; co-mentions do not prove an actual company-country deal.

Show Qwen facts/filter comparisons first, then E5's natural-language interest versus
article input and dynamic phrase similarities. No manual bilingual explanation
pairs, no probability claims for cosine scores, no neural-reasoning claims for
phrase probes. Keep model-generated/cached/failed states explicit and UI concise.

Acceptance: real-model new article plus changed-interest journey, model-unavailable
and quota errors, empty results, unknown entity and malformed output tests; inspect
desktop/mobile; all three explained notebooks execute their documented route and
stable stage links still point to current shared code. Record actual live proof,
not only test-double passes or previously saved screenshots.

## P1-P2: Portable Release and Public Verification (Phase 5)

Locations: `scripts/package_news_release.py`, `scripts/restore_news_bundle.py`,
`scripts/publish_news_space.py`, `deployment/`, parent repository `docs/` and workflows.

Strengthen gates using M2, verify an externally recorded bundle checksum and clean
restore without D:/G: path dependence. Confirm all needed runtime artifacts/configs
are included, while caches/environments/large artifacts stay out of ordinary Git.
Keep failed-experiment evidence and training lineage, with model/data license terms
and actual runtime/storage limits. Qwen's research restrictions must not be
misrepresented as unrestricted commercial licensing.

Public URLs are currently unverified. Authentication, free hosting eligibility,
ZeroGPU startup/quota behavior, final FP16 inference and a fresh visitor session
need direct verification at deployment time. A publisher returning a URL is not
enough; the landing page must distinguish pending versus verified live availability.
Static genuine results/guide remain accessible when models are unavailable.

Acceptance: portable restoration, identity gates, real public links, unseen hosted
inference without visitor downloads, license notices, precise CV claims and a final
independent audit. No future task is silently dropped to achieve a green checklist.

## Handoff

Changes: added phase plan, this repair specification, read-only audit script and
two checksum/evidence snapshots; updated the resume/backlog pointers.
Evidence: 55 tests pass; adapter hashes match; protected artifacts unchanged;
failed-update stale-index bug reproduced in a temporary test store.
Unresolved: D1-D3, M1-M4, A1-A3 and P1-P2; all 60 human reviews; validated model
promotion; live/public verification. No inference quality improvement claimed.

Next: **Phase 2, 5.6 Sol Medium**, begin with the safe comparison command above,
then implement D1-D3. Do not run the old preparation queue or combined score
command as a shortcut. Pause for genuine human review before completing Phase 2.
