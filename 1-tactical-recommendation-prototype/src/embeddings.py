import os
import re
from functools import lru_cache
from threading import RLock

import numpy as np


MODEL_NAME = os.getenv("TACTICAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TAG_CATEGORIES = ("countries", "companies", "organizations", "profiles", "systems", "topics")
CATEGORY_LABELS = {
    "countries": "Countries",
    "companies": "Companies",
    "organizations": "Organisations",
    "profiles": "Profiles",
    "systems": "Systems",
    "topics": "Topics",
}
VECTOR_CACHE_LIMIT = 4096
_VECTOR_CACHE = {}
_VECTOR_CACHE_LOCK = RLock()
PHRASE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
    "follow", "monitor", "track",
    "عن", "في", "على", "إلى", "الى", "من", "مع", "و", "أو", "او", "هذا",
    "هذه", "التي", "الذي",
}


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _metadata_text(values_by_category):
    sections = []
    for category in TAG_CATEGORIES:
        values = values_by_category.get(category, [])
        sections.append(f"{CATEGORY_LABELS[category]}: {', '.join(values) if values else 'None'}")
    return " | ".join(sections)


def user_interest_text(user):
    interest = str(user.get("query", "")).strip()
    if not interest:
        raise ValueError(f"User {user.get('user_id', '<unknown>')} has no explicit interest statement.")
    return f"query: {interest}"


def article_content_text(article):
    parts = []
    seen = set()
    for label, field in (("Title", "title"), ("Summary", "summary"), ("Body", "body")):
        value = str(article.get(field, "")).strip()
        normalized = normalize(value)
        if value and normalized not in seen:
            parts.append(f"{label}: {value}")
            seen.add(normalized)
    return "passage: " + " | ".join(parts)


# Backward-compatible names for the older presentation notebook.
def user_metadata_text(user):
    return user_interest_text(user)


def article_metadata_text(article):
    return article_content_text(article)


@lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements.txt before loading embeddings."
        ) from exc

    try:
        from src.portfolio_config import EMBEDDING_MODEL, EMBEDDING_REVISION
        if MODEL_NAME != EMBEDDING_MODEL:
            raise ValueError("Unsupported encoder override: create a separately versioned pipeline/index first")
        return SentenceTransformer(MODEL_NAME, revision=EMBEDDING_REVISION, device=os.getenv("NEWS_EMBEDDING_DEVICE") or None)
    except Exception as exc:
        raise RuntimeError(f"Could not load embedding model {MODEL_NAME!r}: {exc}") from exc


def encode_texts(texts):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    requested = list(texts)
    with _VECTOR_CACHE_LOCK:
        requested_vectors = {text: _VECTOR_CACHE[text] for text in requested if text in _VECTOR_CACHE}
        missing = list(dict.fromkeys(text for text in requested if text not in _VECTOR_CACHE))
        if missing:
            model = _get_model()
            encoded = model.encode(
                missing,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, encoded):
                _VECTOR_CACHE[text] = np.asarray(vector, dtype=np.float32)
                requested_vectors[text] = _VECTOR_CACHE[text]

            while len(_VECTOR_CACHE) > VECTOR_CACHE_LIMIT:
                _VECTOR_CACHE.pop(next(iter(_VECTOR_CACHE)))

        # A single request can exceed the LRU capacity. Do not evict its own results.
        return np.stack([requested_vectors[text] for text in requested])


def embedding_similarity(user, article):
    vectors = np.asarray(encode_texts([user_interest_text(user), article_content_text(article)]))
    return float(np.dot(vectors[0], vectors[1]))


def _phrase_candidates(text, max_words=48, max_ngram=2):
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", str(text), flags=re.UNICODE)[:max_words]
    candidates = []
    seen = set()

    for size in range(1, max_ngram + 1):
        for start in range(len(words) - size + 1):
            phrase_words = words[start : start + size]
            if size > 1 and (
                phrase_words[0].casefold() in PHRASE_STOPWORDS
                or phrase_words[-1].casefold() in PHRASE_STOPWORDS
            ):
                continue
            content_words = [
                word
                for word in phrase_words
                if word.casefold() not in PHRASE_STOPWORDS and len(word) > 1
            ]
            if not content_words:
                continue
            phrase = " ".join(phrase_words)
            key = normalize(phrase)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "phrase": phrase,
                    "start": start,
                    "end": start + size,
                    "word_count": size,
                }
            )
    return candidates


def find_e5_phrase_matches(user_text, article_text, max_matches=3, min_similarity=0.55):
    """Return E5-derived phrase pairs that explain a full-text semantic match."""
    user_candidates = _phrase_candidates(user_text)
    article_candidates = _phrase_candidates(article_text)
    if not user_candidates or not article_candidates:
        return []

    texts = [
        *[f"query: {item['phrase']}" for item in user_candidates],
        *[f"passage: {item['phrase']}" for item in article_candidates],
    ]
    vectors = np.asarray(encode_texts(texts), dtype=np.float32)
    user_vectors = vectors[: len(user_candidates)]
    article_vectors = vectors[len(user_candidates) :]
    similarities = user_vectors @ article_vectors.T

    ranked = []
    for user_index, user_item in enumerate(user_candidates):
        for article_index, article_item in enumerate(article_candidates):
            similarity = float(similarities[user_index, article_index])
            if similarity < min_similarity:
                continue
            length_bonus = 0.002 * (
                min(user_item["word_count"], 3) + min(article_item["word_count"], 3) - 2
            )
            ranked.append(
                {
                    "user_index": user_index,
                    "article_index": article_index,
                    "rank_score": similarity + length_bonus,
                    "similarity": similarity,
                }
            )

    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    selected = []
    used_user_positions = set()
    used_article_positions = set()

    for item in ranked:
        user_candidate = user_candidates[item["user_index"]]
        article_candidate = article_candidates[item["article_index"]]
        user_positions = set(range(user_candidate["start"], user_candidate["end"]))
        article_positions = set(range(article_candidate["start"], article_candidate["end"]))
        if user_positions.intersection(used_user_positions):
            continue
        if article_positions.intersection(used_article_positions):
            continue

        selected.append(
            {
                "user_phrase": user_candidate["phrase"],
                "article_phrase": article_candidate["phrase"],
                "similarity": round(item["similarity"], 3),
            }
        )
        used_user_positions.update(user_positions)
        used_article_positions.update(article_positions)
        if len(selected) == max_matches:
            break

    return selected


def _tag_items(values_by_category, source):
    items = []
    prefix = "query" if source == "user" else "passage"
    for category in TAG_CATEGORIES:
        for value in values_by_category.get(category, []):
            items.append(
                {
                    "label": value,
                    "source": source,
                    "category": category,
                    "text": f"{prefix}: {CATEGORY_LABELS[category]}: {value}",
                }
            )
    return items


def find_semantic_matches(user, article, exact_values, max_matches=3, min_similarity=0.45):
    user_items = _tag_items(user.get("interests", {}), "user")
    article_items = _tag_items(article.get("tags", {}), "article")
    if not user_items or not article_items:
        return []

    vectors = np.asarray(encode_texts([item["text"] for item in user_items + article_items]))
    user_vectors = vectors[: len(user_items)]
    article_vectors = vectors[len(user_items) :]
    candidates = []

    for user_index, user_item in enumerate(user_items):
        if normalize(user_item["label"]) in exact_values:
            continue
        for article_index, article_item in enumerate(article_items):
            if normalize(article_item["label"]) in exact_values:
                continue
            similarity = float(np.dot(user_vectors[user_index], article_vectors[article_index]))
            if similarity < min_similarity:
                continue
            candidates.append(
                {
                    "interest": user_item["label"],
                    "interest_category": user_item["category"],
                    "article_tag": article_item["label"],
                    "article_category": article_item["category"],
                    "similarity": round(similarity, 3),
                    "reason": (
                        f"Semantic connection: {user_item['label']} is related to "
                        f"{article_item['label']} ({round(similarity * 100)}%)."
                    ),
                }
            )

    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    selected = []
    used_pairs = set()
    for candidate in candidates:
        pair = (normalize(candidate["interest"]), normalize(candidate["article_tag"]))
        if pair in used_pairs:
            continue
        selected.append(candidate)
        used_pairs.add(pair)
        if len(selected) == max_matches:
            break
    return selected


def reduce_vectors_to_3d(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(vectors) == 1:
        return np.zeros((1, 3), dtype=np.float32)

    from sklearn.decomposition import PCA

    dimensions = min(3, len(vectors), vectors.shape[1])
    reduced = PCA(n_components=dimensions).fit_transform(vectors)
    if dimensions < 3:
        reduced = np.pad(reduced, ((0, 0), (0, 3 - dimensions)))
    scale = np.max(np.abs(reduced))
    return reduced / scale * 3 if scale else reduced


def embedding_map_for_user_article(user, article):
    user_items = _tag_items(user.get("interests", {}), "user")
    article_items = _tag_items(article.get("tags", {}), "article")
    items = user_items + article_items
    if not items:
        return []

    user_values = {normalize(item["label"]) for item in user_items}
    article_values = {normalize(item["label"]) for item in article_items}
    exact_values = user_values.intersection(article_values)
    vectors = encode_texts([item["text"] for item in items])
    coordinates = reduce_vectors_to_3d(vectors)

    points = []
    for item, coordinate in zip(items, coordinates):
        points.append(
            {
                "label": item["label"],
                "source": item["source"],
                "category": item["category"],
                "x": round(float(coordinate[0]), 4),
                "y": round(float(coordinate[1]), 4),
                "z": round(float(coordinate[2]), 4),
                "is_exact_match": normalize(item["label"]) in exact_values,
            }
        )
    return points
