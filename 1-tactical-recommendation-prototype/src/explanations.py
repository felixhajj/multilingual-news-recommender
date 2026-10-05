CATEGORY_LABELS = {
    "countries": "country",
    "companies": "company",
    "organizations": "organisation",
    "profiles": "profile/person",
    "systems": "system",
    "topics": "topic",
}


def exact_reason(category, value):
    label = CATEGORY_LABELS.get(category, category.rstrip("s"))
    return f"Exact metadata match: tracked {label} {value}."


def embedding_reason(similarity):
    percentage = round(similarity * 100)
    return (
        "E5 semantic similarity: the user's written interest and the natural article "
        f"text are {percentage}% semantically close."
    )


def build_short_summary(article, score, top_reasons):
    if not top_reasons:
        return "Limited match with this user's current intelligence profile."
    return f"{article['title']} scores {score}% because " + " ".join(top_reasons[:2])
