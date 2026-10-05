from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langsmith import traceable


RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL_NAME)


@traceable(name="Rerank Search Results")
def rerank_documents(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if not documents:
        return []

    texts = [document["text"] for document in documents]
    scores = list(get_reranker().rerank(query, texts))

    ranked = []
    for document, score in zip(documents, scores):
        ranked.append({**document, "rerank_score": float(score)})

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k]