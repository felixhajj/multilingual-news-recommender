from src.embeddings import embedding_similarity, find_semantic_matches
from src.explanations import embedding_reason, exact_reason


TAG_CATEGORIES = ("countries", "companies", "organizations", "profiles", "systems", "topics")


def normalize(value):
    return str(value).strip().lower()


def normalized_set(values):
    return {normalize(value): value for value in values}


def find_exact_matches(user, article):
    matches = []
    interests = user.get("interests", {})
    tags = article.get("tags", {})

    for category in TAG_CATEGORIES:
        user_values = normalized_set(interests.get(category, []))
        article_values = normalized_set(tags.get(category, []))
        common = set(user_values).intersection(article_values)
        for value_norm in sorted(common):
            value = article_values[value_norm]
            matches.append(
                {
                    "category": category,
                    "value": value,
                    "reason": exact_reason(category, value),
                }
            )
    return matches


def find_embedding_signals(user, article, articles=None):
    exact_values = {normalize(match["value"]) for match in find_exact_matches(user, article)}
    return find_semantic_matches(user, article, exact_values)


def scale_similarity_to_score(similarity):
    return max(0, min(100, round(similarity * 100)))


def _category_match_count(values_by_category, article_tags):
    active_categories = 0
    matched_categories = 0
    for category in TAG_CATEGORIES:
        requested = set(normalized_set(values_by_category.get(category, [])))
        if not requested:
            continue
        active_categories += 1
        available = set(normalized_set(article_tags.get(category, [])))
        if requested.intersection(available):
            matched_categories += 1
    return matched_categories, active_categories


def exact_match_score(user, article):
    required = user.get("required_filters", {})
    source = required if any(required.values()) else user.get("interests", {})
    matched, active = _category_match_count(source, article.get("tags", {}))
    return round(matched / active * 100) if active else 0


def is_direct_match(user, article):
    required = user.get("required_filters", {})
    if not any(required.values()):
        return False
    matched, active = _category_match_count(required, article.get("tags", {}))
    return active > 0 and matched == active


def score_article_for_user(user, article, articles=None):
    exact_matches = find_exact_matches(user, article)
    semantic_matches = find_embedding_signals(user, article, articles)
    similarity = embedding_similarity(user, article)
    embedding_score = scale_similarity_to_score(similarity)
    metadata_score = exact_match_score(user, article)
    score = round(embedding_score * 0.75 + metadata_score * 0.25)
    direct_match = is_direct_match(user, article)

    breakdown = [
        {
            "type": "embedding",
            "score": embedding_score,
            "reason": embedding_reason(similarity),
            "weight": 0.75,
            "formula": "embedding_score * 0.75",
        },
        {
            "type": "metadata",
            "score": metadata_score,
            "reason": "Structured-filter coverage for the user's required filters or broader interests.",
            "weight": 0.25,
            "formula": "metadata_coverage * 0.25",
        },
    ]

    exact_reasons = [match["reason"] for match in exact_matches]
    semantic_reasons = [match["reason"] for match in semantic_matches]
    reasons = exact_reasons[:3] + [embedding_reason(similarity)] + semantic_reasons[:2]

    return {
        "article": article,
        "score": score,
        "match_tier": "direct" if direct_match else "related",
        "direct_match": direct_match,
        "embedding_similarity": round(similarity, 3),
        "embedding_score": embedding_score,
        "metadata_score": metadata_score,
        "exact_matches": exact_matches,
        "semantic_matches": semantic_matches,
        "breakdown": breakdown,
        "reasons": reasons[:5],
        "embedding_map_available": True,
    }


def rank_articles_for_user(user, articles):
    ranked = [score_article_for_user(user, article, articles) for article in articles]
    return sorted(
        ranked,
        key=lambda item: (item["direct_match"], item["score"], item["article"]["date"]),
        reverse=True,
    )
