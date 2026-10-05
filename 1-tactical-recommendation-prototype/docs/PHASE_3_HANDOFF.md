# Phase 3 Processing Handoff

October 1, 2026. **Phase 3 evaluation complete.** The project and five-phase
baseline remain in place. All 60 human reviews and frozen splits are preserved.

## Completion and Measured Outcome: October 1, 10:42 Beirut

The worker saved all 90 Qwen predictions, all validation reports, the selection
lock, ranking comparisons, and 200 linker predictions. It exited with code 1 when
the prepaid deadline expired just before its final report/check stage. No model
inference needed repeating: I ran the saved-evidence integrity check, reproduced
the metrics, ran all 110 tests successfully, and wrote
`output/portfolio/phase3/completion.json`. The job status records this recovery;
the supervisor's original deadline exit remains in its process record.

**Selected:** `extraction-only-v3`. Validation F1/schema validity: **0.372/0.90**
versus base Qwen **0.289/0.40**. Held-out-from-our-training test F1/schema:
**0.313/0.95**. The domain-plus candidate's test F1/schema was 0.295/0.85.
Results are small-sample, exact-surface scores on 10 validation and 20 test
articles; training used 30 machine-assisted weak examples.

The learned linker passed its declared validation gate. On its previously reported
200-mention holdout, accepted precision was **95.9%** at **36.5% coverage**; unknown
mentions were all rejected. Overall correct-link accuracy was only **35.0%**, versus
**33.5%** for unique-alias matching, and unseen-known-alias accuracy was **13.3%**.
Treat this as a promising but weak prototype, especially for unseen names; the
holdout was not newly created for this experiment.

Hybrid was selected because all ranking methods tied on the single validation
query (Recall@5 0.50, nDCG@5 1.00). On two test queries, hybrid/E5 averaged Recall@5
0.50 and nDCG@5 0.956, versus keyword nDCG@5 0.612. This is too small a judged pool
to claim general ranking superiority. The pool uses legacy extraction metadata.

The declared Phase 3 selection gates passed, but this does not mean the portfolio
release is complete. No application model was promoted. Phase 4 still must apply
the selected extractor/linker consistently, rebuild affected indexes, and verify
the live visitor workflow. The original 90-minute allocation and budget ledger
were not reset; the ledger reports 22.53/24 hours charged and 1.47 unallocated.

## Previous Check: 21:12 Beirut, September 30

Both v3 training runs are complete at **48 optimizer updates**, using the same
30 frozen weak examples. Both record 30 unique articles, 29,354 supervised tokens,
and 288/288 changed adapter tensors. The original extraction-only final checkpoint
was reused, not retrained. The hash-bound trainer is unchanged. The domain run
successfully loaded in a fresh isolated process after the earlier 1455 failure;
the historical failure is preserved in its manifest, not treated as a new failure.

One supervisor is active: PID 21236, controller launcher 29732. The current
isolated base-prediction stage is launcher 25964; **19/30** base predictions were
saved at 21:09. The other two sets remain pending. Do not launch a duplicate.
Current logs begin `20260930-174308-supervised`; subsequent children save separate
per-stage output/error logs. Completion of training is not completion of validation.

The user approved allocation `phase3-resume-20260930-c`: 90 minutes, preserving
all previous charges. The unchanged deadline is **22:13 Beirut, September 30**.
Cumulative charged allowance is 22.53/24 hours; **1.47 hours remain unallocated**.
D: has about 25.2 GiB free and portfolio artifacts occupy about 1.37 GiB. The
2 GiB free-space guard and 5 GiB artifact cap remain enforced. No libraries,
Windows settings, frozen reviews, labels, attempts or prior adapters were reset.

The supervisor now runs the remaining sequence automatically within that deadline:
all 90 frozen predictions -> fresh identity-bound linker training/validation ->
keyword/E5/hybrid on identical judged pools -> validation selection lock -> final
reports -> reproducibility checks and unit tests -> `phase3/completion.json` and
`docs/PHASE_3_RESULTS.md`. No model is automatically deployed. Failed gates remain
unmet requirements, not relaxed success criteria. New linker encoding batches and
completed fitting are resumable by input/model/code identity. Validation error
reports distinguish JSON, schema, missed and extra entities without scoring test
references before lock. Ranking uses the explicitly disclosed fixed legacy index;
Phase 4 must apply validated models and rebuild affected records.

All 110 tests passed, including the final validation-only diagnostic fixture;
the final stage reruns the complete suite. A completion record cannot be written without passing tests
and independently reproduced saved extraction, linker and ranking metrics.
An older job-policy string may remain until the next child begins; actual staged
code and the immutable validation policy govern this continuation.

Check using `python scripts/phase3_status.py`, then inspect the actual PIDs/logs.
If interrupted before the deadline, and no worker is alive, resume with:

```powershell
.\.venv\Scripts\python.exe scripts\supervise_phase3.py --extend-hours 1.5 --allocation-id phase3-resume-20260930-c
```

This reuses the same charge and deadline; it does not grant more time. After the
deadline, do not consume the remaining allowance without approval. If completion
exists, reproduce reports with `python scripts/phase3_analysis.py finish` rather
than rerunning model inference. Previous chronological notes below are historical.

## Latest Check: 20:22 Beirut

No active worker. `extraction-only-v3` is actually complete at 48 optimizer updates:
the completed adapter, final checkpoint's three checksums, optimizer/RNG/cursor,
and unchanged trainer/dataset identity were verified. All 288 weight-change
records are nonzero. Actual exposure is 30 unique articles and 29,354 supervised
tokens; loss went from 0.2113 to 0.00382. These are training results, not validation
quality. Evidence: `phase3/reports/resume_verified_step-0048.json`.

The worker stopped at 19:19 Beirut with error 1455 while opening the first Qwen
shard for `domain-extraction-v3`. That run has zero updates. All three new v3
prediction sets are empty; validation and selection remain pending. D: now has
about 29.6 GiB free, so the earlier disk-space blocker is resolved. Labels,
reviews, backups and budget still verify; no automatic restart was performed.

Next repair: isolate each model-loading/training/prediction stage in a fresh
process so memory from a preceding stage cannot persist into the next load.
This is a proposed mitigation, not yet an established root cause or tested fix.
Do not change the hash-bound trainer, rerun the completed extraction experiment,
regenerate labels or reset allocations. The existing deadline is 20:37 Beirut;
only about 15 minutes remained at this check. The 2.97-hour unallocated cumulative
remainder is not permission to silently extend that deadline.

## Latest Restart: 19:12 Beirut

The user freed additional space; D: had 30.5 GiB free. No worker was active before
the restart. The supervisor reused allocation `phase3-resume-20260930-b` with the
same 1.5-hour value; the budget-file SHA-256 remained
`10282cb2681285466fcc50d08db597b88242831875c50be63cd2cf0a7e7666ba`.
The worker resumed checkpoint 16, loaded both Qwen shards and became GPU-active.
It then saved checkpoint **32/48**, with all three checkpoint-pointer hashes
verified. Subsequent update logs show continued optimization. This establishes
actual resumed training, not merely a successful model load or a stale status.
Logs: `phase3/logs/20260930-161241-supervised.*.log`. At the loading check D: still
had about 27 GiB free. Deadline remains 20:37 Beirut. Do not launch a duplicate
worker; inspect live processes and durable checkpoint pointers first.

## Previous Verification: 19:08 Beirut (Historical)

**No worker is active. The disk guard stopped the resumed training, not a model
loading error.** New durable progress is update **16/48**, at
`output/portfolio/runs/extraction-only-v3/checkpoints/step-0016`. All three pointer
checksums, 288 optimizer states, RNG and identity were verified again. Actual
exposure is 30 unique articles and 9,656 supervised tokens; latest durable loss
is 0.06153. `phase3/reports/resume_verified_step-0016.json` records this evidence
and all protected review/dataset/trainer hashes. The original step-0008 checkpoint
and recovery inventory remain intact. Replay only uncheckpointed later updates.

D: fell below the 2 GiB safety reserve during training as the existing Windows
dynamic page file expanded. Portfolio artifacts increased only about 85 MiB.
After worker exit D: had about 4.7 GiB free, but the page file can expand further
from approximately 12.7 GiB to its existing 16 GiB maximum. Ask the user to free
at least 4 GiB additional space before retrying. Do not change Windows settings,
reduce the guard, delete project artifacts or charge the same allocation again.
When space is ready, check no active worker and reuse:

```powershell
.\.venv\Scripts\python.exe scripts\supervise_phase3.py --extend-hours 1.5 --allocation-id phase3-resume-20260930-b
```

The 20:37 Beirut deadline is unchanged. The worker must stop if its allocation is
exhausted; another allocation needs approval within the cumulative remainder.

## Resume Details: 18:59 Beirut

Continue the existing plan; no relabeling or budget reset. The actual durable
checkpoint is `output/portfolio/runs/extraction-only-v3/checkpoints/step-0008`.
Its state/adapter/config hashes match the pointer. CPU inspection verified 288
optimizer states, CPU/CUDA RNG state, eight loss records, 30 seen article IDs,
4,929 supervised tokens and an identity matching the unchanged trainer/dataset.
The eight-update loss moved from 0.2113 to 0.0989; this is training evidence,
not a validation-quality claim. Uncheckpointed updates 9-10 are replayed.
All 49 recovery backup files passed `preserved_sha256.json` verification.

The first chat-controlled restart again failed with 1455 before any new update.
After explicit permission, Gradle 8.14's background daemon was stopped gracefully;
the separate BathroomRater Firebase emulator was left running. No Windows setting
or dependency changed. The next supervisor loaded both Qwen shards. Logs:
`phase3/logs/20260930-155955-supervised.*.log`. Actual durable progress must still
be checked rather than inferred from the manifest's `running` field.

The user approved 90 additional minutes from the existing cumulative remainder.
Allocation `phase3-resume-20260930-b` preserves every prior charge and deadline;
new deadline is **20:37 Beirut, September 30**. Total conservatively charged
allowance is 21.03 hours; 2.97 hours remain. No budget was reset.
At launch D: had 5.68 GiB free and portfolio artifacts occupied 0.35 GiB.
`src/phase3_resources.py` enforces 2 GiB free-drive headroom and a 256 MiB
checkpoint reserve within the 5 GiB artifact cap, without deleting anything.

The fixed dataset contains 30 weak examples (20 EN / 10 AR), from 160 preserved
attempts: 22 raw accepted and eight case-only derived. Its hash is
`e51fe22d97bc1d5871659037a53119fffb7a1732aca57b79299d7f7450cf3d9e`.
No further labeling is scheduled. The trainer source hash remains
`4b1ade24863f7bc7e2b250c1279693bb91970373480b767acab290fd87b980b8`.

### Safeguards and Validation Work

All 100 tests passed, including 25 focused Phase 3 checks.
`manage_news_model.py` now requires the hash-verified
validation selection lock, the selected run, complete linker provenance and a
matching prepared index; the old combined report cannot promote an adapter.
Linker runtime identity binds actual model/catalogue/vector bytes and feature/
encoding source. Future training records these identities; existing artifacts
are not silently upgraded. This invalidates the old index for the current linker,
which must be rebuilt consistently in Phase 4, not substituted invisibly.

`scripts/validate_phase3_linker.py` only reads saved validation predictions. On
300 published-hyperlink mentions, accepted precision is 94/103 (91.3%), coverage
34.3%, overall correct-link accuracy 31.3% versus the unique-alias baseline 29.3%,
known-KB candidate recall@10 90.4%, unknown abstention 95.7%, and unseen known-alias
accuracy 5/25 (20%). These are explicitly historical weak-reference diagnostics.
`phase3/reports/linker_historical_validation.json` binds the audit inputs, but
does not claim that current vector/implementation hashes generated old predictions.
Fresh identity-bound linker validation remains required; no final-test file was
read by this audit. Do not declare M4 or Phase 3 complete from these results.

Next: verify new optimizer progress/checkpoints, let the worker finish its two
48-update runs and 90 predictions, inspect validation only, finish identity-bound
linker/retrieval checks on unchanged pools, then lock decisions before final-test
reporting. No candidate has been selected or promoted. If the worker is active,
do not launch another one. After interruption use the status command first;
reuse allocation ID `phase3-resume-20260930-b` only with the same 1.5-hour value.

## Earlier Resume: 16:39 Beirut (Historical)

The first labeling job stopped at 88 attempts: 13 raw-schema-valid, grounded
examples (10 EN / 3 AR), 75 rejections, and no new optimizer updates. That complete
state is archived under `phase3/history/label_job_v1/`; its attempt-file SHA-256 is
`7fbfc01a7978ae33b81897e0d7b0ac4889cb6665cf8ddc58bc1fe6a6b029c4a4`.

A separate offline weak-label derivation can recover **four** rejected generations
by changing only top-level field-name casing. It requires all eight fields,
rejects missing/colliding keys, nonliteral values and incomplete article chunks,
and records that the original output failed schema validation. This yields
17 usable examples (13 EN / 4 AR), still below the unchanged 20-example minimum.
This is not a repair to live inference, not a valid raw generation claim and not
a human correction. Strict evaluation and the extraction promotion gate remain
unchanged. A native-tokenizer audit reproduced the prior tokenizer's dataset hash.

The resumed worker continues the **72 unattempted** queue items, targeting 32 usable
weak examples within its allowance. It saves raw attempts in the original log
and creates a distinct `weak_labels_v3_case_recovery_v1` dataset only after the
same bilingual minimum is met. Existing rejected attempts are not re-generated.
The selected dataset is hash-bound through `phase3/training_dataset.json` and
the actual training manifest; old reviewed files and old adapters stay unchanged.

The explicit two-hour extension uses allocation ID `phase3-resume-20260930-a`.
Repeated use of that ID neither charges twice nor resets the deadline. The current
deadline is **19:07 Beirut, September 30**. Charged cumulative allowance is now
19.53 hours, leaving 4.47 hours for remaining local processing; this is still
conservative accounting, not a claim that 19.53 hours were measured.

Six recovery tests and three extension tests passed; source compilation passed.
The supervised resume loaded both Qwen shards successfully. Training and selection
are still unverified. Check `scripts/phase3_status.py`, actual processes and logs.
The earlier notes below are historical and are superseded by this resume.

## Evidence So Far

The existing 90 predictions were reused. Only the ten extraction validation
references (five English, five Arabic) were scored for selection:

| Historical v2 run | Schema validity | Entity micro-F1 |
| --- | --- | --- |
| Base Qwen | 0.40 | 0.244 |
| Extraction-only v1 | 0.80 | 0.220 |
| Domain-plus-extraction v1 | 0.90 | 0.224 |

These scores exclude locations because the old runs use v2. Their separate
strict-v3 reports fail schema validity; no v2 adapter is eligible for a
location-aware release. Better formatting is not improved extraction recall.
No final-test metric has been computed by the new Phase 3 commands.

Ten focused tests passed, including CPU optimizer/RNG/cursor restoration after
interruption, dataset-change refusal, strict grounding and the language-regression
gate. This is not a claim that a completed GPU training run has been verified.
Python source compilation and the sealed Phase 2 integrity check also passed.

## Running Job

`scripts/run_phase3_job.py --hours 3`, launched 14:07 Beirut time, September 30.
The prepaid wall-clock deadline is **17:07 Beirut time**. Expected processing is
roughly two to three hours, but the job can fail or stop earlier; inspect status.

The job selects at most 160 existing train-only candidate articles, targets 80
v3 weak-label examples, checks literal spans, and preserves all rejected output.
No frozen extraction/ranking article group is eligible for training. The teacher
checks fields, but label exhaustiveness remains unverified: this is an explicitly
weak-label research experiment, not a human-gold dataset. Published hyperlink
annotations are incomplete and are not converted into exhaustive negatives.

It trains `extraction-only-v3` and `domain-extraction-v3` for up to 48 optimizer
updates each. The second reuses the completed `domain-v1` initialization; raw
domain training is not repeated and its actual 90-article exposure remains visible.
Checkpoints every eight updates bind adapter, optimizer, RNG and deterministic
data cursor together. A restart replays only work after the last durable update.
All trainable tensor changes, supervised tokens, dropped long examples and actual
session runtimes are recorded. Vocabulary stays unchanged.

It saves predictions for the same 30 frozen articles for each new v3 run, including
errors, raw output and generation-limit metadata. It scores validation only and
does not select, promote or score final tests automatically.

Artifact locations:

- `output/portfolio/phase3/job.json`: progress, deadline, observed runtime and errors.
- `output/portfolio/phase3/logs/`: worker stdout/stderr.
- `output/portfolio/phase3/weak_labels_v3/`: queue, attempts, dataset and provenance.
- `output/portfolio/runs/*-v3/`: loss, checkpoints, tensor changes and manifests.
- `output/portfolio/evaluation/*-v3/`: immutable prediction inputs and saved outputs.
- `output/portfolio/phase3/reports/`: validation-only reports.

Status command:

```powershell
.\.venv\Scripts\python.exe scripts\phase3_status.py
```

Do not launch a second worker while one holds the GPU lock. If interrupted before
the deadline, `python scripts/supervise_phase3.py` resumes durable stages. After the deadline,
it refuses to reset its budget; inspect saved progress and the remaining ledger
before allocating any further processing.

Two early launches failed inside `safetensors`/`torch.storage` while loading the
first checkpoint shard. Windows recorded a low-virtual-memory warning and a native
access violation, not a Python training error. No labels or optimizer updates
were produced by those launches. Their logs are preserved; the original unsafe
"running" status is not treated as proof of processing. The supervised retry
uses `NEWS_SAFETENSORS_BACKEND=pread` and one CPU thread. The scoped loader adapter
restores Transformers' original function even after an exception. See the
[official storage-backend API](https://huggingface.co/docs/safetensors/api/torch).
The process record and logs, not merely a nonempty job file, must establish that
the retry is alive. Do not increase system paging or close unrelated applications
without checking the machine's current state and the user's ongoing work.

Verified at approximately 14:23 Beirut: the buffered retry loaded both Qwen
checkpoint shards, remained alive, and was actively generating at 98% GPU
utilization. This is live processing evidence, not a completed training claim.

## Budget and Remaining Work

The cumulative ledger now conservatively charges 17.53 hours: 10.53 historical
recorded hours, a four-hour uncertainty allowance for unmetered early work, and
the new three-hour job. **6.47 hours remain**, including later indexing/deployment
processing. The allowances are conservative accounting, not measured runtimes.
No budget was reset. New checkpoints are bounded to about 1-2 GB for both runs;
actual total artifact size must still be checked after processing.

To finish Phase 3: inspect actual v3 validation failures, finish identity-bound
promotion/index checks, evaluate linker candidate recall/KB coverage/abstention
on validation, compare keyword/E5/hybrid on identical frozen judged pools, and
declare linker/ranking criteria before final reporting. Then lock all decisions
and report final-test results with unchanged references and artifact identities.
If no candidate passes the original gate, record no promotion and an unmet
portfolio release requirement. Do not weaken criteria or claim completion.
