from langsmith import traceable


@traceable(name="Planner Node")
def planner_node(state: dict) -> dict:
    """Choose the extraction or question-answering workflow."""
    task = state.get("task", "question_answering")
    question = state.get("question", "")

    if task == "extract":
        return {
            "task": task,
            "plan": "Search bid documents, extract fields, reconcile addenda, validate.",
            "search_query": (
                "bid number title due date submission term pre-bid installation "
                "bond delivery payment affidavits manufacturer contract model "
                "part product contact company specifications"
            ),
        }

    return {
        "task": "question_answering",
        "plan": "Search the bid documents and answer the question with citations.",
        "search_query": question,
    }