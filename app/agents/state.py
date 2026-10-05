from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state for the RFP agent workflow."""

    bid_id: str
    question: str
    task: str  # "extract" or "question_answering"

    plan: str
    search_query: str
    retrieved_documents: list[dict]

    extracted_data: dict
    addendum_changes: list[dict]
    validation_result: dict

    answer: str
    retry_count: int