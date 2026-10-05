import json

from langsmith import traceable

from app.agents.llm import llm

def _parse_json(response: str) -> dict:
    response = response.strip()

    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0].strip()

    return json.loads(response)


@traceable(name="Addendum Reconciliation Node")
def reconcile_addenda_node(state: dict) -> dict:
    """Update extracted fields when retrieved addenda change them."""
    documents = state.get("retrieved_documents", [])
    extracted_data = state.get("extracted_data", {})

    addenda = [
        doc for doc in documents
        if doc.get("doc_type") == "addendum"
    ]

    if not addenda:
        return {"addendum_changes": []}

    context = "\n\n".join(
        f"Source: {doc.get('file_name')}, page: {doc.get('page_number')}\n"
        f"{doc.get('text', '')}"
        for doc in addenda
    )

    prompt = f"""
Compare the extracted fields with the addendum evidence.
Apply changes only when the addendum clearly changes a field.
Keep unchanged fields as they are.

Return valid JSON in this shape:
{{
  "extracted_data": {{}},
  "addendum_changes": []
}}

Each change must include field_name, previous_value, updated_value,
and sources with file and page. Do not guess.

Extracted fields:
{json.dumps(extracted_data)}

Addendum evidence:
{context}
"""

    result = _parse_json(llm.invoke(prompt).content)

    return {
        "extracted_data": result.get("extracted_data", extracted_data),
        "addendum_changes": result.get("addendum_changes", []),
    }
