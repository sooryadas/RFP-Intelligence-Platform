from langsmith import traceable


@traceable(name="Build Citation")
def build_citation(result: dict) -> dict:
    """Return a compact file/page citation for a search result."""
    return {
        "file": result.get("file_name", "Unknown"),
        "page": result.get("page_number"),
    }