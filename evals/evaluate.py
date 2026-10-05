"""Compare vector, hybrid, and hybrid + reranker retrieval on a gold set.

Expected dataset format: a JSON list with `question`, `bid_id`, and either
`expected_source_ids` or `expected_passages`. Passage entries should include
`file` and `page`, matching the metadata written on indexed chunks.
"""

import json
from pathlib import Path

from qdrant_client import models

from app.config import Settings
from app.search.embedding import embed_query
from app.search.hybrid import _build_filter, search_documents
from app.search.index import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_client,
)
from app.search.keyword import embed_keyword_query
from evals.metrics import mean_reciprocal_rank, recall_at_k


EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "golden_dataset.json"
TOP_K = 5
RETRIEVAL_LIMIT = 20


def source_id(result: dict) -> str:
    """Build a stable source ID from indexed file and page metadata."""
    file_name = result.get("file_name")
    page_number = result.get("page_number")
    if not file_name or page_number is None:
        return ""
    return f"{file_name}::page::{page_number}"


def expected_source_ids(item: dict) -> set[str]:
    """Accept explicit IDs or the file/page passage format used in the gold set."""
    if item.get("expected_source_ids"):
        return set(item["expected_source_ids"])

    passages = item.get("expected_passages", [])
    return {
        f"{passage['file']}::page::{passage['page']}"
        for passage in passages
        if passage.get("file") and passage.get("page") is not None
    }


def vector_search(question: str, bid_id: str, top_k: int) -> list[dict]:
    """Run dense-vector retrieval without fusion or reranking."""
    response = get_client().query_points(
        collection_name=Settings.QDRANT_COLLECTION,
        query=embed_query(question),
        using=DENSE_VECTOR_NAME,
        query_filter=_build_filter(bid_id=bid_id),
        limit=top_k,
        with_payload=True,
    )
    return [point.payload or {} for point in response.points]


def hybrid_search(question: str, bid_id: str, top_k: int) -> list[dict]:
    """Run dense + sparse fusion without reranking."""
    query_filter = _build_filter(bid_id=bid_id)
    response = get_client().query_points(
        collection_name=Settings.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=embed_query(question),
                using=DENSE_VECTOR_NAME,
                limit=max(top_k * 4, RETRIEVAL_LIMIT),
                filter=query_filter,
            ),
            models.Prefetch(
                query=embed_keyword_query(question),
                using=SPARSE_VECTOR_NAME,
                limit=max(top_k * 4, RETRIEVAL_LIMIT),
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return [point.payload or {} for point in response.points]


def hybrid_rerank_search(question: str, bid_id: str, top_k: int) -> list[dict]:
    """Use the project's existing hybrid retrieval and reranking pipeline."""
    return search_documents(query=question, bid_id=bid_id, top_k=top_k)


CONFIGURATIONS = {
    "vector_only": vector_search,
    "hybrid": hybrid_search,
    "hybrid_rerank": hybrid_rerank_search,
}


def evaluate_configuration(
    search_fn,
    dataset: list[dict],
    k: int = TOP_K,
) -> dict:
    """Evaluate retrieval and retain per-question details for error analysis."""
    retrieved_by_query = []
    expected_by_query = []
    recall_scores = []
    per_question = []

    for item in dataset:
        expected = expected_source_ids(item)
        if not expected:
            raise ValueError(
                f"No expected source passages for {item.get('id', item)}"
            )

        results = search_fn(
            question=item["question"],
            bid_id=item["bid_id"],
            top_k=k,
        )

        # Score unique file/page sources, not duplicate chunks from one page.
        retrieved = list(dict.fromkeys(
            sid
            for result in results
            if (sid := source_id(result))
        ))

        retrieved_by_query.append(retrieved)
        expected_by_query.append(expected)

        recall = recall_at_k(retrieved, expected, k)
        recall_scores.append(recall)

        missing = expected - set(retrieved[:k])
        per_question.append({
            "id": item.get("id"),
            "bid_id": item["bid_id"],
            "question": item["question"],
            f"Recall@{k}": recall,
            "expected_sources": sorted(expected),
            "retrieved_sources": retrieved[:k],
            "missing_sources": sorted(missing),
        })

    return {
        f"Recall@{k}": (
            sum(recall_scores) / len(recall_scores)
            if recall_scores else 0.0
        ),
        "MRR": mean_reciprocal_rank(
            retrieved_by_query,
            expected_by_query,
        ),
        "questions": len(dataset),
        "per_question": per_question,
    }


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Gold dataset not found: {DATASET_PATH}. Create golden_dataset.json "
            "with question, bid_id, and expected_passages or expected_source_ids."
        )

    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    display_names = {
        "vector_only": "Vector only",
        "hybrid": "Hybrid",
        "hybrid_rerank": "Hybrid + reranker",
    }
    results = []
    for name, search_fn in CONFIGURATIONS.items():
        scores = evaluate_configuration(search_fn, dataset, k=TOP_K)
        results.append((
            display_names.get(name, name),
            scores[f"Recall@{TOP_K}"],
            scores["MRR"],
        ))

    headers = ["Configuration", f"Recall@{TOP_K}", "MRR"]
    rows = [
        [name, f"{recall:.3f}", f"{mrr:.3f}"]
        for name, recall, mrr in results
    ]
    widths = [
        max([len(headers[index]), *(len(row[index]) for row in rows)])
        for index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(
            (
                value.ljust(widths[index])
                if index == 0
                else value.rjust(widths[index])
            )
            for index, value in enumerate(row)
        ) + " |"

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(border)
    print(format_row(headers))
    print(border)
    for row in rows:
        print(format_row(row))
    print(border)


if __name__ == "__main__":
    main()
