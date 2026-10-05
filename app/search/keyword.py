from functools import lru_cache

from fastembed import SparseTextEmbedding
from langsmith import traceable
from qdrant_client import models


SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@traceable(name="Embed Keyword Documents")
def embed_keyword_documents(texts: list[str]):
    return list(get_sparse_model().embed(texts))


@traceable(name="Embed Keyword Query")
def embed_keyword_query(query: str) -> models.SparseVector:
    vector = next(get_sparse_model().query_embed(query))

    return models.SparseVector(
        indices=vector.indices.tolist(),
        values=vector.values.tolist(),
    )