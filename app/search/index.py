from functools import lru_cache
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langsmith import traceable
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.ingestion.models import DocumentChunk
from app.search.embedding import DENSE_VECTOR_SIZE, embed_documents
from app.search.keyword import embed_keyword_documents


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if Settings.QDRANT_URL:
        return QdrantClient(
        url=Settings.QDRANT_URL,
        api_key=Settings.QDRANT_API_KEY,
        timeout=120,
        )

    return QdrantClient(path=".qdrant")


def ensure_collection() -> None:
    client = get_client()
    collection_name = Settings.QDRANT_COLLECTION

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    for field_name in ("bid_id", "doc_type"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="addendum_number",
        field_schema=models.PayloadSchemaType.INTEGER,
    )


@traceable(name="Index Chunks")
def index_chunks(chunks: list[DocumentChunk]) -> int:
    """Add or update chunks without reindexing other bid folders."""
    if not chunks:
        return 0

    ensure_collection()

    texts = [chunk.text for chunk in chunks]
    dense_vectors = embed_documents(texts)
    sparse_vectors = embed_keyword_documents(texts)

    client = get_client()
    collection_name = Settings.QDRANT_COLLECTION
    points = []

    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        payload = chunk.model_dump()
        payload["citation"] = {
            "file": chunk.file_name,
            "page": chunk.page_number,
        }

        points.append(
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    batch_size = 64

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
            timeout=120,
    )

    return len(points)
