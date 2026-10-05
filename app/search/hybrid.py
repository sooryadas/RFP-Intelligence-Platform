from langsmith import traceable
from qdrant_client import models

from app.config import Settings
from app.search.citations import build_citation
from app.search.embedding import embed_query
from app.search.index import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_client,
)
from app.search.keyword import embed_keyword_query
from app.search.rerank import rerank_documents


def _build_filter(
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum_number: int | None = None,
) -> models.Filter | None:
    conditions = []

    if bid_id:
        conditions.append(
            models.FieldCondition(
                key="bid_id",
                match=models.MatchValue(value=bid_id),
            )
        )

    if doc_type:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type),
            )
        )

    if addendum_number is not None:
        conditions.append(
            models.FieldCondition(
                key="addendum_number",
                match=models.MatchValue(value=addendum_number),
            )
        )

    return models.Filter(must=conditions) if conditions else None


@traceable(name="Hybrid Search")
def search_documents(
    query: str,
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Run dense + BM25 retrieval, fuse results, then rerank them."""
    client = get_client()
    collection_name = Settings.QDRANT_COLLECTION

    query_filter = _build_filter(
        bid_id=bid_id,
        doc_type=doc_type,
        addendum_number=addendum_number,
    )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=embed_query(query),
                using=DENSE_VECTOR_NAME,
                limit=max(top_k * 4, 20),
                filter=query_filter,
            ),
            models.Prefetch(
                query=embed_keyword_query(query),
                using=SPARSE_VECTOR_NAME,
                limit=max(top_k * 4, 20),
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=max(top_k * 4, 20),
        with_payload=True,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}
        result = {
            "text": payload.get("text", ""),
            "bid_id": payload.get("bid_id"),
            "file_name": payload.get("file_name"),
            "doc_type": payload.get("doc_type"),
            "page_number": payload.get("page_number"),
            "addendum_number": payload.get("addendum_number"),
            "score": point.score,
        }
        result["citation"] = build_citation(result)
        results.append(result)

    return rerank_documents(query, results, top_k=top_k)