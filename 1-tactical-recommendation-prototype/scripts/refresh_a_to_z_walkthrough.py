from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "tactical_report_a_to_z_walkthrough.ipynb"


def markdown_index(notebook, prefix):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.startswith(prefix):
            return index
    raise ValueError(f"Could not find notebook section: {prefix}")


def markdown_index_any(notebook, *prefixes):
    for prefix in prefixes:
        try:
            return markdown_index(notebook, prefix)
        except ValueError:
            continue
    raise ValueError(f"Could not find any notebook section: {prefixes}")


notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

notebook.cells[2].source = """import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if Path('D:/hf-cache').exists():
    os.environ.setdefault('HF_HOME', 'D:/hf-cache')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / 'data'
ADAPTER_DIR = ROOT / 'output' / 'entity_extraction_qwen25_3b' / 'adapter'
EXTRACTION_PATH = ROOT / 'output' / 'extractions' / 'incoming_article_filters.jsonl'

from src.data_loader import load_articles, load_entity_catalogue, load_incoming_articles, load_users
from src.embeddings import (
    article_content_text,
    encode_texts,
    find_e5_phrase_matches,
    user_interest_text,
)
from src.entity_enrichment import enrich_articles
from src.extraction_outputs import apply_extraction_outputs, load_extraction_outputs
from src.llm_extractor import article_extraction_text
from src.scoring import (
    exact_match_score,
    find_embedding_signals,
    find_exact_matches,
    is_direct_match,
    rank_articles_for_user,
    score_article_for_user,
)

print('Project root:', ROOT)"""

notebook.cells[4].source = """users = load_users(DATA_DIR)
catalogue = load_entity_catalogue(DATA_DIR)
extraction_records = load_extraction_outputs(EXTRACTION_PATH)

incoming = enrich_articles(load_incoming_articles(DATA_DIR), catalogue)
incoming = apply_extraction_outputs(incoming, extraction_records)
all_articles = load_articles(DATA_DIR) + incoming

user = next(item for item in users if item['user_id'] == 'user_004')
article = next(item for item in all_articles if item['article_id'] == 'incoming_001')

display({
    'USER PROFILE': user['name'],
    'EXPLICIT USER INTEREST': user['query'],
    'STRUCTURED INTERESTS': user['interests'],
    'REQUIRED FILTERS': user['required_filters'],
    'ARTICLE': article['title'],
    'SUMMARY': article.get('summary'),
})"""

extraction_heading = markdown_index(notebook, "## 6. See Qwen + LoRA's extraction JSON")
notebook.cells[extraction_heading].source = """## 6. See Qwen + LoRA's extraction JSON

Loading Qwen and generating output live takes much longer than the rest of this walkthrough, so this cell loads a real stored adapter result.

Important limitation: this is a demonstration artifact, not evidence that every selectable incoming article runs through Qwen + LoRA. The future trained prototype must generate and store a versioned extraction for every incoming article before that result reaches the recommender."""

entity_heading = markdown_index(notebook, "## 7. See entity enrichment")
notebook.cells[entity_heading].source = """## 7. See entity enrichment

This current step is **manually configured**, not learned. It only works for aliases already entered in the entity catalogue. Known aliases are matched to canonical IDs and filters.

After a learned entity linker is trained and evaluated, the Tactical Report model prototype must replace this catalogue-matching step with model-scored links for every incoming article, while sending uncertain matches for review."""

pre_e5_heading = "## 8. See what works before E5"
try:
    pre_e5_index = markdown_index_any(
        notebook,
        pre_e5_heading,
        "## 8. See the structured result before E5",
        "## 8. Model 1 result: Qwen + LoRA creates article filters",
    )
except ValueError:
    entity_code_index = entity_heading + 1
    notebook.cells[entity_code_index + 1:entity_code_index + 1] = [
        nbformat.v4.new_markdown_cell(),
        nbformat.v4.new_code_cell(),
    ]
    pre_e5_index = entity_code_index + 1

notebook.cells[pre_e5_index].source = """## 8. Model 1 result: Qwen + LoRA creates article filters

Qwen + LoRA reads the article and produces structured facts. The current catalogue step normalizes known aliases, then we compare only the user's required filters with the article filters.

The useful result is simple: how many required filter categories are present in the article?"""
notebook.cells[pre_e5_index + 1].source = """required_filters = user['required_filters']
article_filters = article['tags']
filter_rows = []

for category, user_values in required_filters.items():
    if not user_values:
        continue
    article_values = article_filters.get(category, [])
    shared = {
        value.casefold() for value in user_values
    }.intersection(value.casefold() for value in article_values)
    filter_rows.append({
        'filter': category,
        'user_requires': user_values,
        'article_has': article_values,
        'match': bool(shared),
    })

matched_filters = sum(row['match'] for row in filter_rows)
display(filter_rows)
print(f'MODEL 1 RESULT: {matched_filters} of {len(filter_rows)} required filters matched.')"""

old_e5_index = markdown_index_any(
    notebook,
    "## 8. Build the exact text sent to E5",
    "## 9. A different model starts here: E5 analyzes the user's interest",
    "## 9. A different model starts: build the exact text sent to E5",
    "## 9. Model 2 starts: E5 compares meaning across languages",
)
notebook.cells[old_e5_index].source = """## 9. Model 2 starts: E5 compares meaning across languages

Multilingual E5 receives the user's written interest and the article's natural text. It maps Arabic and English directly into the same meaning space.

There is **no translation step** and no entity-catalogue lookup inside E5. The `query:` and `passage:` prefixes are simply its retrieval format."""
notebook.cells[old_e5_index + 1].source = """user_e5_text = user_interest_text(user)
article_e5_text = article_content_text(article)

print('USER INTEREST SENT TO E5 (no filters):\\n')
print(user_e5_text)
print('\\nNATURAL ARTICLE TEXT SENT TO E5 (no Qwen JSON):\\n')
print(article_e5_text)"""

embedding_index = markdown_index_any(
    notebook,
    "## 9. See the E5 embedding vectors",
    "## 10. See E5's embedding vectors",
    "## 10. Encode both languages into the same meaning space",
    "## 10. E5 creates one meaning vector for each text",
)
notebook.cells[embedding_index].source = """## 10. E5 creates one meaning vector for each text

The English interest and Arabic article each become one normalized 768-number vector. The individual numbers are not useful to present; their comparison is what matters."""
notebook.cells[embedding_index + 1].source = """vectors = np.asarray(encode_texts([user_e5_text, article_e5_text]))
user_vector, article_vector = vectors

print('Translation step used: NO')
print('Combined tensor shape:', vectors.shape)
print('One text vector shape:', user_vector.shape)
print('User vector norm:', round(float(np.linalg.norm(user_vector)), 5))
print('Article vector norm:', round(float(np.linalg.norm(article_vector)), 5))"""

similarity_index = markdown_index_any(
    notebook,
    "## 10. Calculate semantic similarity",
    "## 11. See E5's result by itself",
    "## 11. Show the recommendation benefit of E5",
    "## 11. Model 2 result: the two texts have related meaning",
    "## 11. Model 2 result: E5 finds related phrases",
)
notebook.cells[similarity_index].source = """## 11. Model 2 result: E5 finds related phrases

For explanation, the code builds short phrases from both inputs, embeds every phrase with E5, and keeps the strongest non-overlapping cross-language matches.

These links are now genuinely generated by E5 at runtime. The final recommendation similarity still compares the two complete texts."""
notebook.cells[similarity_index + 1].source = """semantic_similarity = float(np.dot(user_vector, article_vector))
article_phrase_text = ' '.join(
    str(article.get(field, '')) for field in ('title', 'summary', 'body')
)
phrase_matches = find_e5_phrase_matches(user['query'], article_phrase_text)

display(phrase_matches)
print(f'MODEL 2 RESULT: {semantic_similarity:.0%} meaning similarity.')"""

exact_index = markdown_index_any(
    notebook,
    "## 11. Separate exact evidence from semantic evidence",
    "## 12. Compare the two separate results",
    "## 12. Keep exact evidence and semantic evidence separate",
    "## 12. Put the two model results side by side",
)
notebook.cells[exact_index].source = """## 12. Put the two model results side by side

- **Model 1, Qwen + LoRA:** required article filters matched.
- **Model 2, multilingual E5:** the user's English interest and Arabic article are close in meaning."""
notebook.cells[exact_index + 1].source = """display({
    'MODEL 1 - QWEN + LORA': f'{matched_filters}/{len(filter_rows)} required filters matched',
    'MODEL 2 - MULTILINGUAL E5': f'{semantic_similarity:.0%} meaning similarity',
})"""

final_index = markdown_index_any(
    notebook,
    "## 12. Calculate the final hybrid score",
    "## 13. Calculate the final hybrid score",
)
notebook.cells[final_index].source = """## 13. Calculate the final hybrid score

Only now are the two model results combined: 75% E5 meaning similarity and 25% Qwen/filter matching.

This weighting is a prototype decision, not something learned automatically."""
notebook.cells[final_index + 1].source = """result = score_article_for_user(user, article)

print(
    f"({result['metadata_score']} × 0.25) + "
    f"({result['embedding_score']} × 0.75) = {result['score']}%"
)
display({
    'MODEL 1 - QWEN + LORA FILTER MATCH': f'{matched_filters}/{len(filter_rows)}',
    'MODEL 2 - E5 MEANING SIMILARITY': f"{result['embedding_score']}%",
    'FINAL RECOMMENDATION SCORE': f"{result['score']}%",
})"""

ranking_index = markdown_index_any(
    notebook,
    "## 13. Rank the complete article collection",
    "## 14. Rank the complete article collection",
)
notebook.cells[ranking_index].source = """## 14. Rank the complete article collection

The final application repeats both comparisons for every article and sorts the results. Direct required-filter matches are placed before broader related results."""

proof_index = markdown_index(notebook, "## What the complete prototype proves")
notebook.cells[proof_index].source = """## What the complete prototype proves

```text
Article text
    -> Qwen tokenizer produces token IDs
    -> Qwen + trained LoRA adapter produces structured JSON
    -> manually configured catalogue enrichment connects known aliases
    -> MODEL 1 RESULT: user-required filters are compared with article filters
    -> MODEL 2, multilingual E5, compares English interest with Arabic article meaning
       directly in one 768-number meaning space, without translation
    -> the two named model results are combined
    -> articles are ranked with visible explanations
```

The current system is a working proof of the architecture. Its next quality improvements require a larger reviewed extraction dataset, domain adaptation on approved geopolitical text, and a learned entity-linking component. Once those trained components pass evaluation, the prototype must replace saved mock extraction outputs and manual alias matching with live, versioned model results for every incoming article."""

nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Updated {NOTEBOOK_PATH}")
