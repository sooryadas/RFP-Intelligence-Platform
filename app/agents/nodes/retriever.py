from langsmith import traceable

from app.search.hybrid import search_documents


@traceable(name="Retriever Node")
def retriever_node(state: dict) -> dict:
    """Search the selected bid and return passages with citation metadata."""
    results = search_documents(
        query=state.get("search_query", ""),
        bid_id=state.get("bid_id"),
        top_k=8,
    )

    return {"retrieved_documents": results}