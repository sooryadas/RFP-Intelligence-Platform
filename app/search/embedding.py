from functools import lru_cache

from fastembed import TextEmbedding
from langsmith import traceable


DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


@traceable(name="Embed Documents")
def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = get_dense_model().embed(texts)
    return [vector.tolist() for vector in vectors]


@traceable(name="Embed Search Query")
def embed_query(query: str) -> list[float]:
    vector = next(get_dense_model().query_embed(query))
    return vector.tolist()