# Phase 2 Human Review

Completed September 30, 2026: all 30 extraction and 30 recommendation reviews are
saved, validated and sealed. See [PHASE_2_HANDOFF.md](PHASE_2_HANDOFF.md).
The instructions below document the review procedure; further changes require
versioning the references and updating the completion seal before evaluation.

The 30 extraction references and 30 recommendation judgments were completed by
the human reviewers. The review
screen does not load Qwen, E5 or any saved prediction. It cannot suggest the
answers and therefore does not leak model output into the references.

Open the newest review URL printed by the launch command. If it is not running, use:

```powershell
cd "D:\1-Projects\1-TacticalReport\2-Final_Mock_Project\1-tactical-recommendation-prototype"
.\.venv\Scripts\python.exe scripts\review_news.py --port 7865
```

For a reviewer on another network, launch an authenticated temporary tunnel:

```powershell
.\.venv\Scripts\python.exe scripts\review_news.py --host 0.0.0.0 --port 7865 --share --username reviewer --password "a-private-password"
```

Share only the generated `gradio.live` URL, username and password with the intended
reviewer. The tunnel is temporary: this PC and the review process must stay on.
Submissions are still written to the local frozen files. Different reviewers cannot
overwrite one another's completed items; divide item ranges to avoid duplicate work.

Enter your name or a stable reviewer ID. It is saved with each decision so the
project can distinguish genuine human review from machine-assisted labels.

## Extraction tab

The screen is a verification aid, not a blank extraction exercise. It highlights
candidate names in the article and pre-fills six plain-language entity lists.
To add or reclassify a name, click one of the six colored category buttons and
select the exact phrase directly in the article. The highlight and corresponding
list update together. In Remove mode, select a phrase or click an existing
highlight. Undo reverses the latest change; Reset restores the original suggestions
for that article. The lists remain editable as a fallback. There is no spreadsheet
or JSON editing.

Candidates come from published Wikinews hyperlinks in the selected article and
matching hyperlink surfaces elsewhere in the 5,000-article corpus. Wikidata
provides conservative draft categories. These aids are independent of the models
being evaluated, but they remain heuristic and require human confirmation.

The measured categories are countries, named non-country locations, companies,
organizations, people and named systems/equipment. Countries are sovereign states.
Locations include Gaza/Gaza Strip, cities, territories, regions, seas, straits,
borders and named bases. Do not add nationality adjectives or unnamed places.
Topics are subjective and relationships are not used by
the recommender, so neither is part of the human F1 task. Both remain part of the
model-output schema-validity test.

Copy names exactly as written in the title or article, including Arabic spelling.
Do not translate or canonicalize aliases. Do not add dates, job titles, generic
equipment, topics or relationships. The save action validates grounding
before advancing. Five English and five Arabic items are frozen
validation references; the other twenty are final test references. The screen
hides those roles to reduce review bias.

The review protocol is now schema v3. One item completed under v2 was preserved as
a prefilled draft rather than discarded; it must be saved once more after checking
locations. The 19 existing recommendation judgments were not changed. The old v2
protocol, labels and adapters remain historical evidence and are not presented as
location-aware training.

## Recommendation tab

Judge whether the displayed article satisfies the displayed user's interest and
required filters:

- `0`: irrelevant
- `1`: partly relevant or useful background, but not a direct answer
- `2`: directly relevant

Judge the article itself, not whether you think a model would rank it. There are
three fixed user queries with ten articles each. This is a small judged-pool
experiment, not a claim about every possible geopolitical query.

## What Is Frozen

The new integrity manifests bind article IDs, titles, bodies, sources, languages,
groups, instructions, required filters and validation/test roles. Review labels,
grades, reviewer attribution and timestamps are the only intended mutable fields.
Every save also appends a hash-only audit event. The interface refuses invented
entity strings, duplicate frozen items, boolean ranking grades and changed inputs.

Do not run evaluation scoring after review. Phase 3 first separates validation
from final-test reporting and locks model selection before test metrics are opened.

## Background Label Preparation

The first launches were recorded but stopped before useful inference. The GPU had
about 10 GB free, but Windows had shrunk its system-managed page file to 1.45 GB;
Qwen then failed while loading/allocating the checkpoint. `configure_ml_pagefile.ps1`
adds an 8-16 GB page file on D: while keeping a small C: file. It requires an
administrator prompt and a Windows restart before retrying:

```powershell
Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$PWD\scripts\configure_ml_pagefile.ps1"
```

After the restart, resume the same bounded operation:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2_labeling.py --target 60 --hours 3
```

The stricter Phase 2 grounding check quarantined 6 of the historical 26 examples
because at least one relationship endpoint was not literal article text. The
original file and historical runs remain unchanged; the new seed has 20 examples.

The job can run for at most three hours, keeps ten nominal ledger hours reserved for
later validation/training/indexing, and resumes by article/content/model/prompt
identity. It produces machine-assisted weak labels, not human references, and does
not train an adapter. Check both workstreams without loading a model:

```powershell
.\.venv\Scripts\python.exe scripts\phase2_status.py
```
