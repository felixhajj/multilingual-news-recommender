# Resume Status

## Current: October 1, 2026

Latest completion check: the Phase 4 index worker exited successfully (code 0)
at 16:12 Beirut. All 300 selected-version articles are prepared: 156 Arabic and
144 English. The worker added 209 successful articles in 9,506.547 seconds.
There are 23 current-version failed attempts (nine new), plus the preserved 194
historical failures. No worker is active and no further indexing is needed.

Read-only integrity verification passed for all 300 records: selected model and
pipeline identities, content-addressed cache keys, grounding offsets, 393 raw
generated JSON chunks and finite normalized 768-number vectors. Three articles
have no grounded mentions; this is retained as a model limitation.
The completed collection also passed tokenizer-only verification: all 405 E5
chunks fit the 512-token limit, with a largest encoded chunk of 453 tokens.

The ledger now records 95,791.155/115,200 seconds, leaving 5.39 hours. D: has about
29.83 GiB free. Continue Phase 4 with the isolated selected-pipeline judged-pool
verification, notebook execution, fresh-input and desktop/mobile checks. Do not
rerun completed indexing. Public deployment and cleanup have not passed yet.

The launch notes below are historical context for this completed worker.

Approved continuation: the baseline now has six phases, with a 32-hour cumulative
ML-processing cap. All 86,284.608 seconds previously charged remain preserved.
The completion allocation reserves 4.5 hours for indexing, 1.5 for app/notebook
checks, one for selected-pipeline recommendation verification and one for contingencies.
Phase 6 cleanup is blocked until Phase 5 deployment/backup/restoration passes.
All 120 regression tests passed before launch; the live pipeline identity and
all 91 existing records were verified unchanged by the new worker guards.

The guarded worker was launched October 1 at 13:34 Beirut. Its actual supervisor
and worker processes were verified alive, and its first durable manifest reports
91/300 records, 49 Arabic / 42 English, processing `en-32562`. Canonical live
status: `output/portfolio/phase4/process.json`; log stem `20261001-103402`.
Startup verification: both Qwen checkpoint shards loaded successfully in 88 seconds;
the worker remained alive, with about 25.9 GiB free on D: after loading.
Allow roughly 3-4.5 hours for the remaining 209 articles. The assistant pauses
while this detached process works. A power interruption preserves completed
records, but requires checking process state and restarting the supervisor;
restart time does not reset the cumulative/index-section allowance.

Resume Phase 4 with `python scripts/supervise_release.py --hours 4.5 --limit 300`.
First inspect `output/portfolio/phase4/process.json` and the actual OS process:
do not launch a second worker. The supervisor records durable logs and exit status;
the worker skips the existing 91 articles, preserves failures and old versions,
shares the indexing allocation across restarts, and charges final exports.

Tokenizer-only check: all 131 chunks from the 91 saved articles fit E5's limit
(largest 452 tokens, limit 512). The whole-text tokenization warning was not
evidence of actual over-limit encoded chunks in this saved collection.

The notes immediately below describe the pre-extension October 1 checkpoint and
are historical budget instructions, superseded by the approved continuation above.

Latest check: **Phase 3 is evaluated; no Phase 4 worker is active.** The selected
Phase 3 configuration is applied in the shared app. The bounded re-index was
resumed from its durable state and exited successfully with **91/300 selected-
version articles**. Its version-specific failure log records 14 malformed or
schema-invalid extraction outputs; none were substituted with manual tags. The
historical 194 failed attempts remain preserved. The old 300+7 records remain
under their legacy pipeline identities, and the old SQLite table is retained.

The ledger charged 5,170.969 seconds for Phase 4: 2,323.281 seconds last durably
recorded before interruption, a 300-second in-flight reserve, and 2,547.688
seconds for the resumed process. The cumulative ledger is now 86,284.608/86,400
seconds, leaving **115.392 seconds**. Do not launch additional model inference
within this remaining sliver. Run evidence: `output/portfolio/
release_manifest.json`, `output/portfolio/phase4/release_failures-7bb217afa48f.jsonl`,
and `output/portfolio/processing_budget.json`. A tokenizer warning reported an
input sequence length of 531 against a 512 maximum; identify its source before
claiming there was no truncation risk.

All 112 unit tests pass, including index-version migration and batched-linker
equivalence. The app reports the current index as partial. Notebook execution,
fresh visitor article generation, and desktop/mobile visual checks remain
unverified. Public deployment is still unverified. Continue Phase 4 with
non-inference checks only unless the processing plan is explicitly revised.

The restart notes below describe earlier checkpoints and are historical.

**Phases 1 and 2 are complete. Phase 3 is in progress, assigned Astra High in the baseline.**
Read [PHASE_PLAN.md](PHASE_PLAN.md) and [docs/PHASE_3_HANDOFF.md](docs/PHASE_3_HANDOFF.md).

All 30 schema-v3 extraction reviews and all 30 recommendation judgments passed
integrity, grounding and attribution checks. The two original 10-validation /
20-test splits and all article/query identities are preserved. A versioned
checksummed snapshot seals the references and preparation evidence.

The v2 weak-label queue exhausted 300 attempts (3 accepted, 297 rejected), leaving
23 usable machine-assisted examples: 17 English / 6 Arabic, zero equipment
coverage. A versioned export restores source/license metadata without changing
labels. Phase 3 now has 30 frozen machine-assisted v3 examples (20 EN / 10 AR)
from all 160 completed attempts. Human review references remain evaluation-only.

Checkpoint verification at 19:08 and restart at 19:12 Beirut, September 30: the verified checkpoint is
`output/portfolio/runs/extraction-only-v3/checkpoints/step-0016`, not under
`phase3/runs`. Adapter/config/state checksums, 288 optimizer states, RNG states,
data cursor and the unchanged training identity were verified. The 49 backup
files under `phase3/history/recovery-20260930-1528` passed their SHA inventory.
The intentional stop left update 8 durable; resumption successfully reached
update 16 and saved optimizer/RNG/cursor state again. Another load failed with
1455; after authorization, only the background Gradle daemon was gracefully
stopped. The BathroomRater Firebase emulator was left running. Windows settings,
libraries, trainer source, original adapters and human reviews were not changed.

Historical validation has been scored separately from test: v2 base F1 0.244,
extraction-only 0.220, domain-plus 0.224. None is a location-aware v3 candidate.
The worker loaded both shards and resumed actual optimizer updates, but stopped
when the disk guard observed less than 2 GiB free on D:. Windows' existing dynamic
page file grew during processing; the portfolio grew only about 85 MiB.
The user then freed space, increasing D: to 30.5 GiB free. The supervisor restarted
at 19:12 from update 16 using the existing allocation. Both Qwen shards loaded
and the GPU became active; D: still had approximately 27 GiB free. The budget
file hash was unchanged by the retry. Training subsequently saved checkpoint
32/48, and all three pointer hashes were verified. Check current logs before assuming it is
still running. Do not weaken the guard or delete model/project artifacts.
The remaining job reuses the frozen dataset, finishes both candidates, saves
90 new predictions and scores validation only. No test score is exposed.

The ledger conservatively charges a four-hour allowance for unmetered historical
overhead, the original three-hour job, the two-hour resume allowance and the
user-approved 90-minute continuation: 21.03 hours charged, 2.97 hours remain.
The latest allocation ID is `phase3-resume-20260930-b`; neither allocation
can be charged twice or have its deadline reset by a retry.
These allowances are not measured training runtime. Actual sessions are saved
separately. The active release still uses the legacy adapter; no promotion yet.

Next command:

```powershell
.\.venv\Scripts\python.exe scripts\phase3_status.py
```

Inspect the live worker/logs before rerunning anything. The prepaid deadline
is September 30, 20:37 Beirut time. A guard stops processing below 2 GiB of free
artifact-drive space or near the 5 GiB artifact cap; it never deletes evidence.
All 100 tests passed, including 25 focused Phase 3 checks. The old combined evaluation can no longer authorize
promotion, and linker cache identity now includes actual vector/model bytes.
Saved linker validation was audited separately; its missing historical inference
bindings remain an explicit limitation, not retroactively repaired provenance.
Then finish M2/M4, lock all validation choices,
and only afterward report final tests. Do not rerun the completed Phase 2 queue.

Reproducible recovery evidence is saved in
`output/portfolio/phase3/reports/resume_verified_step-0016.json`; regenerate it
with `python scripts/verify_phase3_resume.py` only while the worker is stopped.
Once sufficient disk space is available and the deadline has not expired, use:

```powershell
.\.venv\Scripts\python.exe scripts\supervise_phase3.py --extend-hours 1.5 --allocation-id phase3-resume-20260930-b
```

This reuses the already approved allocation without charging it again. If its
deadline has expired, inspect the remaining ledger; do not reset the deadline.

## September 17 Notes (Historical)

The following entries describe the earlier Phase 2 state and are superseded above.
Follow [PHASE_PLAN.md](PHASE_PLAN.md) and the concrete
[repair specification](docs/PHASE_1_AUDIT.md), not the historical queue instructions below.

The bounded Phase 2 v2 labeling job finished all 300 attempts: 3 additions were
accepted and 297 rejected, leaving 23 usable v2 examples including the 20-example
clean seed. Do not rerun it. The old three training runs, 90 predictions and 300
indexed articles remain historical v2 artifacts using the legacy adapter.

Extraction schema v3 now adds named non-country `locations`. The same 30 frozen
article IDs and roles are preserved. The one previously completed v2 review is
stored as a draft and reopened for its location check; extraction progress is
therefore 0/30 under v3. Recommendation progress remains 19/30 and its file hash
was unchanged by migration. Candidate preparation produced 35 conservative location suggestions after adjective and generic-surface filtering.
Public deployment and model promotion remain unverified.

The audit found 26 machine-assisted training examples (17 EN / 9 AR), only two
company examples and no systems examples; 194 failed indexing attempts; and 52
current indexed articles with no grounded extracted mentions. A failed persisted
article update can leave stale content in recommendations. The evaluation command
also needs validation/test separation before selecting an adapter.

The v2/v3 boundary is explicit: old examples cannot pass v3 validation by silently
receiving empty locations. No retraining or test-answer inspection was performed
for this migration.
The ledger records 6.74 hours used; its nominal 17.26-hour remainder requires
reconciling unmetered early work before another processing allocation.

Next safe status command from this project:

```powershell
.\.venv\Scripts\python.exe scripts/phase2_status.py
```

D1-D3 and the location migration are prepared. Complete the real review using the
newest URL printed by `scripts/review_news.py`; see
[docs/PHASE_2_REVIEW.md](docs/PHASE_2_REVIEW.md). The superseded five-category
review processes on ports 7863 and 7864 were shut down. Do not fabricate judgments or
advance to Phase 3 while Phase 2 remains incomplete.

September 16 later update: the virtual-memory failure was traced to a 1.45 GB C:
page file. A D: page file (8-16 GB) is configured and active; restart after the
current job, not during it. The lower-memory Qwen configuration passed checkpoint
loading and the three-hour weak-label worker is running with durable attempts.
The reviewer also supports an authenticated temporary Gradio share link and
prevents different reviewers from overwriting one another's completed items.

## Historical Recovery Notes (Superseded)

The remaining notes describe earlier interrupted jobs. Their resume commands and
counts are historical, not instructions to run now.

## September 9 Power-Outage Recovery

All three real-news training experiments completed 24 optimizer steps and saved their adapters. Extraction training used 26 accepted machine-assisted examples; domain training actually saw 90 distinct articles from its 500-article input pool. These are completed optimization runs, not yet validated quality improvements.

All 90 evaluation predictions were saved (30 each for base, extraction-only and domain-plus-extraction). Human review remains pending. The outage interrupted release indexing after nine current-version article records were saved. The old twelve-example export belongs to an earlier configuration and is not added to that current-version count.

Resume with `python scripts/finish_news_preparation.py --resume`. It preserves completed stages, accounts for interrupted indexing time and continues from cached article records. Training is not restarted. Allow approximately **three hours** for the remaining bounded indexing and review-preparation work. Check logs and actual process state before launching another worker.

## Background Processing

The earlier tool-attached process was interrupted. Nine grounded machine-assisted examples were saved; no new real-news adapter had been completed. The interrupted labeling time is accounted for from the last durable progress record, not described as successful training.

Processing is restarted with a detached Windows background process, so the assistant can stop without consuming credits while local Python works. Keep the PC powered on, awake and connected to the internet. Do not run another Qwen training/inference job at the same time.

The queue runs serially:

1. Resume label preparation, targeting 100 examples, capped at 2.5 additional hours. Accepted labels remain machine-assisted, not human gold.
2. Run extraction-only, domain adaptation and domain-plus-extraction experiments, capped at one hour each and 24 optimizer steps. At least 20 accepted examples are required; otherwise training fails explicitly.
3. Generate saved evaluation predictions, capped at two hours.
4. Process/index up to 300 articles, capped at three hours. This uses the currently active, explicitly identified model, not automatic promotion of unreviewed weights.
5. Freeze recommendation judgments and refresh the visitor site.

Allow roughly **8-12 hours** before continuing. These are budgets, not promises that every stage will succeed or improve accuracy. The cumulative processing limit remains 24 hours.

Check `output/portfolio/preparation_queue.json`, `training_preparation.json`, `processing_budget.json`, `runs/*/manifest.json`, and the timestamped files in `output/portfolio/logs/`. A file saying "running" is not sufficient proof that its OS process is still alive.

## Verified So Far

- Corpus: 5,000 real historical articles, 3,539 English and 1,461 Arabic.
- Learned linker: fresh 200-mention holdout; 70/73 accepted links correct, 36.5% coverage, all 114 out-of-catalogue mentions unresolved. Unseen-alias performance remains weak.
- 55 regression tests pass.
- All three notebooks passed default execution. Expensive opt-in training/live-generation cells were not exercised by that notebook check.
- Initial 12 real inference records exist, explicitly using the historical adapter. A changed prompt/embedding configuration requires re-indexing; stale records are not silently treated as current inference.
- Landing page was inspected in the browser; app input validation worked. A dark-mode contrast bug was fixed, but the final post-fix desktop/mobile checks remain to be done.

## Next Assistant Session

1. Check actual OS processes and queue logs before starting another job. Inspect failures, dataset counts, optimizer steps, changed weights and runtime budgets.
2. Audit the real-news extraction outputs and ablations. Do not describe a time limit or a checkpoint alone as evidence of better quality.
3. Finish app visual verification, current-index interaction tests and new-article generation checks.
4. Prepare the local human-review route: 30 extraction articles and 30 recommendation judgments. No agent-fabricated human labels. Evaluation/promotion remains blocked until review is done.
5. Verify release packaging, data/model notices, and deployment. No public URLs have been verified or published in this implementation session. Hugging Face authentication and eligible free ZeroGPU hosting remain necessary.
6. Recheck Git changes and publish only a verified, accurately labeled release. Source and documentation are currently local changes, not a completed public release.

Start with `START_HERE.md`. The project is not "done done" yet; this pause lets the slow local work finish without paying for an assistant to watch it.
