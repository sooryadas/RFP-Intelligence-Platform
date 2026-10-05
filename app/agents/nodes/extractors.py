import json

from langsmith import traceable

from app.agents.llm import llm


def _parse_json(response: str) -> dict:
    """Parse JSON, including when the model wraps it in a code fence."""
    response = response.strip()

    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0].strip()

    return json.loads(response)


@traceable(name="Extraction Node")
def extractor_node(state: dict) -> dict:
    """Extract bid fields from retrieved passages."""
    documents = state.get("retrieved_documents", [])

    context = "\n\n".join(
        f"Source: {doc.get('file_name')}, page: {doc.get('page_number')}\n"
        f"{doc.get('text', '')}"
        for doc in documents
    )

    prompt = f"""
Extract bid information using only the evidence below.

Return valid JSON. Each field must have this shape:
{{"value": null, "sources": [], "confidence": 0.0, "notes": ""}}

Include these fields:
Bid Number, Title, Due Date, Bid Submission Type, Term of Bid,
Pre Bid Meeting, Installation, Bid Bond Requirement, Delivery Date,
Payment Terms, Any Additional Documentation Required,
MFG for Registration, Contract or Cooperative to use, Model_no,
Part_no, Product, contact_info, company_name, Bid Summary,
Product Specification.

For each found value, include at least one source with file and page.
If a value is missing, use null, an empty sources list, and
"Not found in documents" for notes. Do not guess.

Evidence:
{context}
"""

    extracted_data = _parse_json(llm.invoke(prompt).content)
    return {"extracted_data": extracted_data}
