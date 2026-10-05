def recall_at_k(ranked_article_ids, relevant_article_ids, k):
    relevant = set(relevant_article_ids)
    if not relevant:
        return 0.0
    returned = set(ranked_article_ids[:k])
    return len(returned.intersection(relevant)) / len(relevant)


def evaluate_ranking_cases(users, articles, cases, ranker, k=5):
    users_by_id = {user["user_id"]: user for user in users}
    results = []
    for case in cases:
        user = users_by_id[case["user_id"]]
        ranked = ranker(user, articles)
        ranked_ids = [item["article"]["article_id"] for item in ranked]
        results.append(
            {
                "case_id": case["case_id"],
                "user_id": case["user_id"],
                "recall_at_k": round(recall_at_k(ranked_ids, case["relevant_article_ids"], k), 3),
                "top_article_ids": ranked_ids[:k],
                "relevant_article_ids": case["relevant_article_ids"],
            }
        )
    return results
