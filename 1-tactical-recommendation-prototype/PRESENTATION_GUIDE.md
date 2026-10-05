# Tactical Report Prototype: What to Present

## The one-sentence goal

> The system reads a new tactical article, extracts structured facts with an adapted
> language model, compares the article with a user's interests using multilingual
> embeddings, and produces an explainable recommendation score.

## Use these two screens

1. Open `notebooks/tactical_report_a_to_z_walkthrough.ipynb`.
2. Open the prototype at `http://127.0.0.1:8501`.

The notebook proves what happens technically. The prototype shows the same pipeline as
one understandable product flow.

## Five-minute walkthrough

### 1. Start with the inputs

Show the selected article and user profile.

Say:

> A recommendation requires two inputs: what this article contains and what this user
> wants to follow.

### 2. Show the tokenizer

Show the token table with text pieces and token IDs.

Say:

> Qwen does not receive words directly. Its tokenizer converts the article into IDs.
> For this example, the Arabic article becomes 42 tokens.

Do not spend time explaining every token ID. The point is only to make the model input
visible.

### 3. Show Qwen and the LoRA adapter

Show the adapter configuration and measured progress.

Say:

> Qwen2.5-3B is the pretrained language model. We did not retrain all three billion
> parameters. QLoRA kept Qwen in a memory-efficient 4-bit form and trained a small
> 7.37-million-parameter adapter for our extraction task.

Then show:

- Entity micro-F1: `0.1538 -> 0.5`
- Valid JSON: `40% -> 100%`

Say:

> The dataset is too small for production, but this before-and-after test proves that
> the adapter learned the required extraction format.

### 4. Show Qwen's JSON

Show the generated JSON containing countries, companies, organisations, systems,
topics, and relationships.

Say:

> This is Qwen's job in our application: turn unstructured article text into structured
> tactical filters.

Be precise: the notebook normally loads the previously generated output. The optional
live-Qwen cell can regenerate it, but loading the full model is intentionally disabled
during a quick presentation.

### 5. Show entity enrichment

Show how the Arabic article mentions map to canonical English entities.

Say:

> The current prototype connects known aliases to stable canonical IDs so an Arabic
> article and an English user profile can share the same exact filter.

Also state the limitation:

> This entity-linking step is currently catalogue-based. A learned entity linker is
> planned future work.

Do not claim that Qwen learned every Arabic alias in the catalogue.

### 6. Show E5 vectors

Show the exact `query:` and `passage:` strings, the `(2, 768)` tensor shape, and the
first vector values.

Say:

> Multilingual E5 performs a different job. It converts the user interest and article
> into 768-dimensional normalized vectors. Their dot product measures semantic
> similarity, including related meaning with different wording or languages.

Do not try to assign a human meaning to individual dimensions. Meaning is distributed
across the complete vector.

### 7. Show the final score

Show the formula:

```text
(E5 semantic score x 75%) + (exact metadata coverage x 25%) = final score
```

Say:

> Exact matching gives reliable direct evidence. E5 gives broader semantic evidence.
> The prototype combines them instead of asking one model to do both jobs.

The `75/25` weighting is a prototype rule, not something the model learned.

### 8. Finish with the ranking

Show the top five articles and point out the direct and related result tiers.

Say:

> The same comparison is repeated for every article. Direct required-filter matches
> are shown first, followed by broader semantically related developments.

## Why these exact models

- `Qwen/Qwen2.5-3B-Instruct` was selected because it supports English and Arabic,
  follows extraction instructions, produces structured text, and can fit on the
  11 GB GTX 1080 Ti through QLoRA.
- `intfloat/multilingual-e5-base` was selected because it is designed for fast
  multilingual retrieval embeddings. It can compare many articles more efficiently
  than asking Qwen to generate a decision for every article.
- LoRA and QLoRA are training methods, not additional pretrained models.

## Honest ending

> This is a working architectural proof, not a production-quality intelligence model.
> The next work is a larger reviewed extraction dataset, legal geopolitical domain
> adaptation, learned entity linking, and evaluation on realistic unseen reports.

That limitation makes the demonstration credible. It does not erase the progress already
shown: the adapter learned, Qwen produced structured output, E5 produced meaningful
vectors, and the complete ranking pipeline runs.
