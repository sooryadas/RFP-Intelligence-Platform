from langsmith import traceable

from app.agents.llm import llm

@traceable(name="Responder Node")
def responder_node(state: dict) -> dict:
    """Return the extracted JSON or answer a question with citations."""
    if state.get("task") == "extract":
        return {"answer": state.get("extracted_data", {})}

    documents = state.get("retrieved_documents", [])
    context = "\n\n".join(
        f"Source: {doc.get('file_name')}, page: {doc.get('page_number')}\n"
        f"{doc.get('text', '')}"
        for doc in documents
    )

    prompt = f"""
Answer the question using only the evidence below.
Cite sources by file name and page number.
If the evidence does not answer the question, say "Not found in documents."

Question:
{state.get('question', '')}

Evidence:
{context}
"""

    answer = llm.invoke(prompt).content
    return {"answer": answer}
