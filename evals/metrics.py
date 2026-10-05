"""Small, deterministic metrics for evaluating retrieval results."""


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Return the fraction of expected sources found in the first k results."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not expected_ids:
        return 0.0

    retrieved_relevant = set(retrieved_ids[:k]) & expected_ids
    return len(retrieved_relevant) / len(expected_ids)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Return the reciprocal rank of the first relevant result, or 0 if absent."""
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in expected_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_ids_by_query: list[list[str]],
    expected_ids_by_query: list[set[str]],
) -> float:
    """Return MRR across queries."""
    if len(retrieved_ids_by_query) != len(expected_ids_by_query):
        raise ValueError("Each query must have one expected source ID set")
    if not retrieved_ids_by_query:
        return 0.0

    scores = [
        reciprocal_rank(retrieved_ids, expected_ids)
        for retrieved_ids, expected_ids in zip(
            retrieved_ids_by_query, expected_ids_by_query
        )
    ]
    return sum(scores) / len(scores)
