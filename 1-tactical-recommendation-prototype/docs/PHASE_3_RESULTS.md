# Phase 3 Measured Results

Generated from hash-bound saved predictions and references.

Evaluation status: **evaluated**. Release eligible: **True**.

Selected extractor: `extraction-only-v3`.
Ranking method: `hybrid`. No application model was automatically promoted.

## Extraction

The same frozen 10 validation and 20 test articles are used for all runs. Decisions use validation only.

| Run | Role | JSON Parse | Schema | Precision | Recall | Entity F1 |
| --- | --- | --- | --- | --- | --- | --- |
| base-v3 | validation | 1.000 | 0.400 | 0.727 | 0.180 | 0.289 |
| base-v3 | test | 0.900 | 0.400 | 0.511 | 0.106 | 0.176 |
| extraction-only-v3 | validation | 1.000 | 0.900 | 0.680 | 0.256 | 0.372 |
| extraction-only-v3 | test | 1.000 | 0.950 | 0.490 | 0.230 | 0.313 |
| domain-extraction-v3 | validation | 1.000 | 0.900 | 0.667 | 0.241 | 0.354 |
| domain-extraction-v3 | test | 1.000 | 0.850 | 0.484 | 0.212 | 0.295 |

Exact surface matching is used. Topics and relationships are schema-checked but not entity-F1 targets.

## Actual Training

30 machine-assisted examples, not human-gold labels; all 160 labeling attempts remain preserved.

- `extraction-only-v3`: 48 optimizer updates, 30 unique articles, 29,354 supervised tokens, 288/288 changed adapter tensors.
- `domain-extraction-v3`: 48 optimizer updates, 30 unique articles, 29,354 supervised tokens, 288/288 changed adapter tensors.

Domain initialization reuses the completed bounded `domain-v1` experiment; raw-domain training was not repeated.

## Learned Linker

E5 candidate retrieval plus a trained logistic ranker; catalogue entries are not manual decisions.

| Role | Accepted Precision | Coverage | Correct-Link Accuracy | Alias Baseline | Unknown Abstention |
| --- | --- | --- | --- | --- | --- |
| validation | 0.913 | 0.343 | 0.313 | 0.293 | 0.957 |
| test | 0.959 | 0.365 | 0.350 | 0.335 | 1.000 |

References are published hyperlink positives, not exhaustive human entity labels. The final linker pool was previously reported, not a fresh independent holdout.

## Recommendation Comparison

Identical human-judged pools for keyword, E5 and hybrid; fixed legacy extraction metadata.

| Role | Method | Recall@5 | nDCG@5 |
| --- | --- | --- | --- |
| validation | keyword | 0.500 | 1.000 |
| validation | e5 | 0.500 | 1.000 |
| validation | hybrid | 0.500 | 1.000 |
| test | keyword | 0.500 | 0.612 |
| test | e5 | 0.500 | 0.956 |
| test | hybrid | 0.500 | 0.956 |

## Limits and Next Phase

- 30 weak training examples; only 10 human-reviewed extraction validation articles.
- One validation recommendation query and two test queries, ten judged articles each.
- Retrieval comparison uses fixed legacy extraction metadata, not the newly selected adapter.
- Linker final pool was previously reported and is not a fresh independent test.
- Phase 4 must apply the validated configuration and rebuild affected indexes consistently.

Charged cumulative processing allowance: 22.53/24 hours; 1.47 hours remain. Prepaid allowances are not measured runtimes.

Reproduce metrics with `python scripts/phase3_analysis.py finish`; no model inference is needed when saved evidence is intact.
Inspect validation failures in `output/portfolio/phase3/reports/*_validation_errors.json`.
Phase 4 applies the validated configuration only if gates pass, rebuilds affected indexes, and finishes the shared application/notebook journey.
