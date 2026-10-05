# Finish the Existing Baseline in Six Phases

Status: **Phases 1-4 complete. Phase 5 packaging/public verification and Phase 6 storage cleanup remain incomplete.**
Updated October 2, 2026. Continue this project, not a replacement implementation.

## Fixed Scope

A standalone English/Arabic geopolitical news recommender for a CV visitor:
change interests, inspect recommendations, paste an unseen article, and follow
extracted facts -> learned entity links -> semantic ranking. Qwen plus its adapter
extracts facts; E5 embeds natural text; a trained candidate ranker links entities.
Filters use structured comparisons, not an additional language model.

Retain the agreed 5,000 real-article collection, up to 500 extraction training
examples, bounded domain-adaptation comparison, 300 processed release articles,
30 extraction reviews, 30 recommendation judgments, three explained notebooks,
START_HERE visitor guide, GitHub Pages landing page and authenticated free live
inference deployment. Keep provenance, licenses, failures and measured limitations.
No vocabulary expansion is required merely because Arabic names split into tokens.

Use existing Qwen/E5 caches. Free services, less than 5 GB additional artifacts,
and **32 cumulative hours of local ML processing** remain constraints. The user
approved the eight-hour extension on October 1, 2026; earlier charges remain intact. Visitors
must not download models. Human review and verified public inference are required;
an adapter checkpoint or a static example does not satisfy release completion.

Development-assistant assignments below follow the user's requested routing.
They do not change application models or automatically switch this session's model.

## Protected Starting Point

See [Phase 1 audit](docs/PHASE_1_AUDIT.md) and its
[machine-readable snapshot](data/release/phase_1_audit_verified.json).

- 5,000 articles: 3,539 English, 1,461 Arabic; no duplicate content hashes or cross-split stored groups found.
- Three completed 24-step experiments; extraction uses 26 machine-assisted examples (17 EN, 9 AR). Domain training saw 90 of its 500 input articles.
- All three adapter checksums match their run manifests. Checkpoints prove updates, not improved extraction quality.
- 90 saved evaluation predictions, counted and hashed without inspecting their answers.
- All 60 review items pending. Preserve extraction roles (10 validation, 20 test) and ranking pools (three queries, ten judgments each).
- 300 current-version indexed articles use the legacy adapter. Seven older-version rows are separate, not additional current release articles.
- 194 recorded failed extraction attempts; 52 current indexed articles have no grounded extracted mentions.
- 55 existing tests pass. Public deployment and a fresh end-to-end real-model visitor session remain unverified.

## Phase 1: Audit and Lock

**Assistant: Astra High. Complexity: high. Status: complete.**

- [x] Verify counts, splits, run manifests, adapter hashes, review roles and index versions.
- [x] Investigate rejection/failure categories and training coverage without consulting final-test answers.
- [x] Review shared pipeline, caching, promotion and release gates; reproduce failed-update stale-index behavior using test doubles only.
- [x] Protect baseline files by SHA-256, rerun existing tests and assign repairs below.
- [x] Replace obsolete resume instructions; record what is verified versus still untested.

Exit evidence: `data/release/phase_1_audit*.json`, `docs/PHASE_1_AUDIT.md`.
No training, inference, promotion or frozen-reference edits were performed.

## Phase 2: Data and Human References

**Assistant: 5.6 Sol Medium. Complexity: moderate/repeatable. Status: complete September 30, 2026.**

- [x] Implement D1: versioned, resumable labeling attempts with complete rejection provenance and balanced language scheduling.
- [x] Implement D2 preparation: train-only coverage report and a 150 EN / 150 AR targeted candidate queue without overwriting the 26-example baseline.
- [x] Implement D3 integrity/UI: freeze complete article inputs/roles and present the existing 30 + 30 review batch in a local review interface.
- [x] Complete the bounded v2 weak-label attempt: 300 attempts produced 3 accepted additions, for 23 usable v2 examples; preserve all 297 rejections.
- [x] Introduce extraction schema v3 with a separate `locations` category without changing frozen article IDs, roles, or ranking judgments.
- [x] Complete location-aware review for all 30 extraction items, preserving the original IDs and roles.
- [x] Validate all 30 recommendation judgments and reviewer attribution; seal both review batches and their candidate pools.
- [x] Report accepted/rejected counts by language and field, remaining gaps, actual processing time, and dataset version.

Exit evidence: [Phase 2 handoff](docs/PHASE_2_HANDOFF.md),
`output/portfolio/phase2/completion.json` and its checksummed snapshot.
Verify with `python scripts/finish_phase2.py --check`.
The 23 usable v2 examples remain machine-assisted weak labels; there are zero
v3 training examples. Preparing and validating a v3 candidate is the explicit
Phase 3 requirement. Human evaluation references must never enter training.

Resolved infrastructure blocker: Windows had reduced `C:\pagefile.sys` to 1.45 GB.
The project now configures a 1-2 GB C: page file plus an 8-16 GB D: page file; the
D: allocation is active and a real Qwen smoke test loaded both checkpoint shards.
The bounded labeling worker was restarted with 512-token chunks and 256 generated
tokens. It has written durable attempts without memory errors. Restart Windows only
after this worker finishes, so the new page-file configuration becomes permanent.

Cleaning result: 6 of the historical 26 machine-assisted examples have nonliteral
relationship endpoints. Their immutable source file and historical runs remain as
evidence; Phase 2 quarantines them and starts its versioned dataset from 20 usable
examples. They are not silently repaired or reused as clean labels.

Review simplification: the extraction UI presents independent candidates from published
article hyperlinks with Wikidata type hints. The reviewer verifies five named-entity
categories plus named non-country locations and adds omissions; no JSON, topic discovery or relationship annotation
is required. Primary F1 is pre-registered on named entities only. Topics and
relationships remain strict generated-JSON schema checks, not human-scored fields.
An authenticated temporary Gradio tunnel permits a second reviewer on another
device. Review saves remain local and cross-reviewer overwrites are rejected.

Schema v3 rule: `countries` contains sovereign states; `locations` contains named
non-country geography such as Gaza/Gaza Strip, cities, territories, regions, seas,
straits, borders and named bases. Nationality adjectives and unnamed places are
excluded. The legacy adapter and its 23 usable examples remain explicit v2
historical evidence. They must not be padded with empty locations or described as
v3 training. Phase 3 must create a v3 candidate from genuinely v3-labeled training
data and compare it against the historical v2 baseline without hiding incompatibility.

Acceptance: D1-D3 checks in the audit pass; training inputs are usable and all 60
items genuinely reviewed. If review is pending, stop with a handoff rather than
advance or mark the phase complete. Sol escalates ambiguous labeling policies to
Astra; it does not relax grounding or change the extraction task to boost yield.

First safe command, from the project directory using its virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts/audit_news_baseline.py --compare data/release/phase_1_audit_verified.json
```

Then implement D1-D3. This audit command does not prepare data or perform review.

## Phase 3: ML Validation and Selection

**Assistant: Astra High. Complexity: highest. Status: evaluated October 1, 2026; selected candidate and component gates passed.**

- [x] Implement M1 command separation: validation-only reports and locked final-test reporting; combined scoring is retired.
- [x] Train and evaluate schema-v3 extraction candidates that predict `locations`; keep the v2 adapter explicitly historical.
- [x] Compare the 10 reviewed validation items for existing runs without rerunning their 90 saved predictions: base F1 0.244, extraction-only 0.220, domain-plus 0.224 (historical v2 scope excludes locations).
- [x] Diagnose validation coverage, JSON/schema errors and entity omissions/extras. With 30 weak examples, further training was not warranted in this bounded phase.
- [x] Implement M2 promotion/index identity gates. New v3 reports bind full inputs, pinned base/tokenizer, adapter/config, training manifest and dataset. Promotion requires the locked candidate, identity-complete selected linker and matching prepared index; actual rebuilding/promotion belongs to Phase 4.
- [x] Implement M3 checkpointed v3 experiments with optimizer/RNG/cursor restoration, all-tensor change evidence, mean update loss, length exclusions and a conservative prepaid budget. Eight focused tests passed, including a CPU interruption/resume fixture; real GPU execution is underway.
- [x] Evaluate the identity-bound learned linker and unknown-name abstention using the declared validation gate; preserve the previously reported final-linker holdout limitation.
- [x] Evaluate keyword/E5/hybrid on the same reviewed pools; select using validation only.
- [x] Lock the extraction and ranking decisions before producing separate held-out test reports.

Existing extraction gate stays fixed: validation schema validity >= max(0.80, base),
overall entity F1 strictly better than base, and neither language F1 below base.
There is no existing numerical linker/retrieval promotion threshold: Astra must
declare any additional threshold on validation before final-test access, never
invent one after observing outcomes. Do not imply a comparative improvement that
was not measured. If no candidate passes, preserve a clearly labeled research
preview and an unmet release requirement, not a silently weakened gate.

Acceptance: reproducible validation report, identity-bound selection decision,
separate final-test report after lock, justified deployment choice and limitations.
Any index needed for comparison is an isolated evaluation index, not premature
replacement of the application deployment. All judged IDs must be represented.

Measured outcome: extraction-only v3 was selected. On the ten held-out-from-our-
training validation articles it reached entity micro-F1 0.372 and schema validity
0.90, compared with base Qwen 0.289 and 0.40. Its 20-article test report reached
F1 0.313 and schema validity 0.95. Linker validation passed the predeclared gate;
test accepted precision was 0.959 at 0.365 coverage, but unseen-alias accuracy
was only 0.133. Hybrid won the fixed ranking tie-break; the test set has only two
queries. See [measured results](docs/PHASE_3_RESULTS.md) and [handoff](docs/PHASE_3_HANDOFF.md).

Phase 3 is the completed evaluation/selection phase, not an application release.
No model has been promoted in the application. Phase 4 must apply the selected
configuration, rebuild affected indexes, test the visitor workflow, and keep the
linker's weak unseen-alias behavior and small evaluation pools visible.

## Phase 4: Application and Walkthrough

**Assistant: 5.6 Sol Medium. Complexity: moderate. Status: complete October 2, 2026; two smoke inferences failed and remain visible.**

- [x] Implement A1: failed-update invalidation and version-safe restoration/index rebuild; preserve the legacy index under its old pipeline identity.
- [x] Apply the validation-locked Phase 3 extractor, linker and ranker identities to the shared app/pipeline; keep static historical examples labeled until current examples exist.
- [x] Add location filters and location evidence to the app, exact matching and schema-v3 analyses.
- [x] Finish basic visitor controls and stale-index/unavailable states; refresh stable notebook stage explanations.
- [x] Complete the selected-model rebuild: 300/300 successes (156 Arabic, 144 English), 23 failed attempts logged, historical 194 failures retained, no manual fallback. Worker exited 0 on October 1 at 16:12 Beirut.
- [x] Present structured facts and exact filters before E5 similarity; phrase probes are labeled illustrative, not causal explanations.
- [x] Reach 300 current-version records and verify stored identities, content hashes, grounding offsets, all 393 generated JSON chunks and normalized 768-dimensional vectors. Three articles have no grounded mentions; retain that real outcome.
- [x] Verify live English/Arabic recommendations and unseen article inference; 28/30 additional smoke articles generated, while two malformed outputs and their raw generations remain recorded as failures, not replacements.
- [x] Record the 32-hour cap without resetting historical charges, with 4.5 hours for indexing, 1.5 for application/notebooks, one for recommendation verification and one for contingencies.
- [x] Add index-allocation, storage and generation-deadline guards, language counts, crash reconciliation and final-export accounting.
- [x] Verify the pinned E5 tokenizer: all 300 articles produce 405 chunks, maximum 453 tokens; no actual chunk exceeds the 512-token limit. Preserve encoder implementation and existing artifact identities.
- [x] Recalculate and independently verify recommendation results from the selected-model analyses/vectors on the unchanged, frozen 30 human judgments; retain validation/test separation and historical comparison.
- [x] Execute all three notebooks' default cells, verify saved output hashes and stage-linked source fingerprints, and check desktop/mobile layout. Notebook live-generation/training switches remain deliberately opt-in; fresh English/Arabic article inference was verified in the app.
- [x] Verify changed interests reorder recommendations, edited content misses cache, model identity is in cache keys, unknown entities remain unresolved, malformed output and unavailable models are visible, and an empty index returns an explicit state.
- [x] Run the final regression suite: 128 tests passed, including cache/model revision, stale-index activation, failure visibility, empty results, linker abstention, and tokenizer chunk limits.

Acceptance met for the local application and walkthrough: one shared, traceable
pipeline works through app and notebooks; no manual tags substitute for failures;
cached inference is content/model bound. The smoke batch is integration evidence,
not a quality evaluation. See [Phase 4 handoff](docs/PHASE_4_HANDOFF.md) and
machine-readable [completion evidence](output/portfolio/phase4/completion.json).
Public hosting and fresh remote-visitor verification belong to Phase 5.

## Phase 5: Package and Public Proof

**Assistant: 5.6 Sol Medium. Complexity: routine execution. Status: in progress.**
**Independent final release audit: Astra High, limited to acceptance/claim checks.**

- [x] Implement P1: runtime/evidence bundle profiles, portable restore, identity-bound bundle gates, model/data license notices and artifact checksums.
- [x] Keep the runtime bundle below 500 MiB (excluding base models/dependencies), with 300 processed articles and every file needed for selection-lock verification; exclude intermediate checkpoints.
- [ ] Preserve full corpus, attempts, reviews, predictions, loss histories, manifests, final adapters, resumable checkpoints and recovery history in a private off-PC evidence archive.
- [ ] Verify clean restoration without the original working directory, pinned published artifacts, remote checksums, and metrics reproduced from saved predictions/references.
- [ ] Check ordinary Git history excludes environments, caches and large model/data artifacts; retain artifact retrieval instructions.
- [ ] Publish Pages and live ZeroGPU only after account authentication, current free-hosting eligibility and gates are verified (P2).
- [ ] Test real public links as a fresh visitor, including an unseen article without local model downloads; record date and outcome.
- [ ] Keep static guide/results usable if hosted inference is unavailable; do not advertise an unverified live URL as working.
- [ ] Finish accurate CV bullets and classify every backlog item as implemented, experimentally evaluated or explicitly outside this release.

Acceptance: public links, restored artifacts, trained/evaluated pipeline, completed
human reviews, walkthroughs and independent audit all pass. Authentication or quota
failures remain visible blockers, not evidence that publishing succeeded.

## Phase 6: Cleanup and Independence

**Assistant: 5.6 Sol Medium. Status: blocked until Phase 5 passes.**

- [ ] Create a default-dry-run cleanup tool with an exact allowlist, sizes, retained dependencies, checksums and verified off-PC backups.
- [ ] Test a restored release without original corpus/checkpoints; download and verify the evidence archive and restore representative training/evaluation artifacts.
- [ ] Remove only verified project-owned caches, environment, temporary downloads and archived bulk work; retain the compact local final release and evidence.
- [ ] Refuse cleanup with active workers, missing backups, changed checksums, protected artifacts, or escaped paths. Shared caches and Windows paging files are outside automatic cleanup.
- [ ] Recheck public inference after local cleanup, reproduce saved metrics, and report actual reclaimed space.

Acceptance: public recommendations and unseen-article inference continue with the
local worker/server stopped; an off-PC copy preserves the 300 articles and complete
research evidence. Local inference/training afterward requires restoring dependencies
and pinned model downloads. Target about 14 GiB reclaimed within this project;
the final verified inventory determines the amount, not this estimate.

## Budget and Handoffs

At Phase 3 closeout, the ledger recorded 81,113.639 seconds (22.53 hours). Phase 4 was charged 5,170.969 seconds: 2,323.281 seconds durably recorded before interruption, a 300-second in-flight reserve, and 2,547.688 seconds in the resumed process. The approved October 1 extension changes the limit from 86,400 to 115,200 seconds while retaining 86,284.608 seconds consumed, leaving 28,915.392 seconds before new work. Allocation ID: `release-completion-20261001-32h`. New index sessions and crash reserves share one 4.5-hour section; retries do not reset it.
Measured October 1 inventory: corpus 23.01 MiB; database 14.48 MiB; project model cache 5.75 GiB; environment 5.41 GiB; pip cache 2.33 GiB; output 1.65 GiB, including about 1 GiB of resumable checkpoints. Shared model caches and paging files are separate and protected. The final runtime index must remain hosted and backed up before local bulk cleanup.

Completed continuation: the indexing worker used 9,506.547 seconds (2.64 hours),
including final exports. Phase 4 then used 1,719.359 seconds in its 5,400-second
application/notebook allocation and 544.499 seconds in its 3,600-second ranking
allocation. The cumulative ledger now records 98,055.013/115,200 seconds
(27.24/32 hours), leaving 17,144.987 seconds (4.76 hours). D: had 29.82 GiB free
at the final check. No project worker is active. See `output/portfolio/processing_budget.json`;
do not reset prior charges or treat unallocated remainder as a new approval.

Each phase ends with: changes, evidence, unresolved requirements, next command,
assistant assignment and remaining budget. Long work runs as a single resumable
background job with logs and checkpoint state; report estimated wait and pause
the assistant. After a power outage, check actual processes before restarting.

## Outside This Release

Large-scale continued pretraining, new base-model downloads without justification,
tokenizer/vocabulary replacement, automatic unsupervised retraining from visitor
articles, enterprise-specific features, commercial use beyond model/data licenses,
and paid hosting are outside the bounded release. Learned linking, live arbitrary
input, review, evidence and public verification are **not** deferred out of scope.
Approved corrections still feed versioned training data via explicit evaluated
retraining with rollback, never silent online learning.
