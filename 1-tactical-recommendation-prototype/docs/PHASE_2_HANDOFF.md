# Phase 2 Handoff

Completed September 30, 2026. Continue with Phase 3 in [PHASE_PLAN.md](../PHASE_PLAN.md).

## Saved References

All 30 extraction articles and 30 recommendation judgments pass the frozen-input,
schema, literal-grounding and reviewer-attribution checks. Each batch retains
10 validation and 20 test items. Recommendation evaluation covers three fixed
interest queries and their original ten-article pools.

Extraction uses schema v3, including non-country locations. Joyce reviewed 9
articles; Felix reviewed 21. Recommendation attribution preserves the entered
names `Felix` (19) and `felix` (11). Answers were not corrected or rescored by the
assistant. These are human-reviewed references aided by independent published
hyperlink suggestions, held out from our training. They are not training examples.

The versioned snapshot contains the references, roles, integrity manifests,
protocol, review audit events and training preparation evidence. Its manifest
records SHA-256 for each file. Any subsequent edit invalidates the closeout check;
reference changes require an explicit new version before evaluation.

## Prepared Training Data

The bounded queue exhausted 300 candidates: 150 English and 150 Arabic.
Three English additions passed; 147 English and all 150 Arabic attempts failed.
All 297 rejections and their raw output/provenance remain available.

| Rejection category | Attempts |
| --- | ---: |
| Missing/extra schema fields | 112 |
| Entity grounding | 107 |
| Invalid JSON | 33 |
| Published positive mentions omitted | 28 |
| Relationship grounding/structure | 17 |

Six historical examples were quarantined for nonliteral relationship endpoints.
The remaining 20 seed examples plus 3 additions form 23 usable v2 weak examples:
17 English and 6 Arabic. Nonempty field coverage is countries 7, companies 3,
organizations 6, people 15, equipment 0 and topics 23. Machine checks establish
literal grounding and agreement with published positive annotations, not exhaustive
human gold. Zero v3 training examples exist yet.

The new versioned export restores source revision/license metadata from matching
corpus records and explicitly records schema v2; its labels are unchanged.
The source files, earlier experiments and legacy adapter remain historical evidence.
Phase 3 must diagnose label quality and prepare location-aware training on train
articles. It must not fill legacy examples with unreviewed empty locations.

## Processing and Next Work

Completed Phase 2 attempts total 13,528.235 seconds (3.76 hours); recorded Phase 2
invocations total 13,555.343 seconds (3.77 hours). The distinction includes load and
failed-invocation overhead. The cumulative ledger is 37,913.639 seconds (10.53 hours),
with 48,486.361 seconds (13.47 hours) nominally remaining. Historical unmetered
preparation/outage time still requires reconciliation before new compute allocation.
No budget reset, new GPU run or model metric calculation occurred at closeout.

Phase 3, assigned Astra High in the baseline, starts by implementing validation-only
scoring and a model-selection lock. It then compares existing validation evidence,
diagnoses the low acceptance rate, prepares/evaluates a v3 candidate, checks learned
linking and compares keyword/E5/hybrid ranking. Final-test results stay closed until
selection decisions are fixed. Model promotion and public deployment remain pending.

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts\finish_phase2.py --check
.\.venv\Scripts\python.exe scripts\phase2_status.py
```

The full machine-readable handoff is `output/portfolio/phase2/completion.json`.
It includes dataset/export identities, the snapshot path, review counts, training
coverage, rejection counts, runtimes and explicit Phase 3 requirements.

Verification at closeout: the seal check passes, status reports `complete: true`,
and 20 focused data/review tests pass, including five completion and tamper checks.
The combined test command stalled importing Gradio; the focused data/markup tests
were rerun with that UI dependency stubbed. No new browser/UI execution is claimed
by this check. Review saves now reject updates after sealing, and the temporary
review server was stopped after all submissions were confirmed saved.
