from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.agents.nodes.extractors import extractor_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.reconcile_addenda import reconcile_addenda_node
from app.agents.nodes.responder import responder_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.validator import validator_node
from app.agents.state import AgentState


MAX_VALIDATION_RETRIES = 2


@traceable(name="Prepare Extraction Retry")
def prepare_retry_node(state: AgentState) -> dict:
    """Add failed fields to the query before retrying retrieval."""
    failed_fields = state.get("validation_result", {}).get("failed_fields", [])
    original_query = state.get("search_query", "")

    return {
        "search_query": f"{original_query} {' '.join(failed_fields)}"
    }


def route_after_planner(state: AgentState) -> str:
    if state.get("task") == "extract":
        return "extract"
    return "question_answering"


def route_after_validator(state: AgentState) -> str:
    validation = state.get("validation_result", {})
    retries = state.get("retry_count", 0)

    if validation.get("failed", 0) and retries < MAX_VALIDATION_RETRIES:
        return "retry"
    return "finish"


def build_graph():
    """Build the RFP extraction and question-answering workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("reconcile_addenda", reconcile_addenda_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("responder", responder_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "extract": "retriever",
            "question_answering": "retriever",
        },
    )

    # Route by task after retrieval.
    workflow.add_conditional_edges(
        "retriever",
        lambda state: "extract" if state.get("task") == "extract" else "answer",
        {
            "extract": "extractor",
            "answer": "responder",
        },
    )

    workflow.add_edge("extractor", "reconcile_addenda")
    workflow.add_edge("reconcile_addenda", "validator")
    workflow.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "retry": "prepare_retry",
            "finish": "responder",
        },
    )
    workflow.add_edge("prepare_retry", "retriever")
    workflow.add_edge("responder", END)

    return workflow.compile()


graph = build_graph()
