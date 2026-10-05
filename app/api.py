from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from langsmith import traceable
from pydantic import BaseModel

from app.agents.graph import graph
from app.ingestion.processor import process_documents
from app.search.hybrid import search_documents


app = FastAPI(title="RFP Intelligence Platform")

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"

BID_FIELDS = [
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
]


class IndexRequest(BaseModel):
    bid_id: str | None = None


class AskRequest(BaseModel):
    bid_id: str | None = None
    question: str
    mode: Literal["extract", "question_answering"] = "question_answering"


def _citation(source: dict) -> dict:
    return {
        "file": source.get("file") or source.get("file_name") or "Unknown",
        "page": source.get("page", source.get("page_number")),
    }


def _format_extraction(state: dict, bid_id: str) -> dict:
    extracted = state.get("extracted_data", {})
    fields = {}

    for name in BID_FIELDS:
        raw = extracted.get(name, {})
        if not isinstance(raw, dict):
            raw = {"value": raw}

        value = raw.get("value")
        sources = [_citation(source) for source in raw.get("sources", [])]
        confidence = raw.get("confidence", 0.0)

        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0

        notes = raw.get("notes")
        if value is None and not notes:
            notes = "Not found in documents"

        fields[name] = {
            "value": value,
            "sources": sources,
            "confidence": confidence,
            "notes": notes,
        }

    changes = []
    for change in state.get("addendum_changes", []):
        changes.append({
            **change,
            "sources": [
                _citation(source)
                for source in change.get("sources", [])
            ],
        })

    validation_state = state.get("validation_result", {})
    failed_fields = set(validation_state.get("failed_fields", []))
    not_found = sum(
        1 for field in fields.values()
        if field["value"] is None
    )
    failed = len(failed_fields)
    passed = len(BID_FIELDS) - not_found - failed

    return {
        "bid_id": bid_id,
        "fields": fields,
        "addendum_changes": changes,
        "validation": {
            "passed": max(0, passed),
            "failed": failed,
            "not_found": not_found,
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
@traceable(name="API Index Bid")
def index_bid(request: IndexRequest):
    """Index all DATA documents, or just one bid folder."""
    if request.bid_id:
        if Path(request.bid_id).name != request.bid_id:
            raise HTTPException(status_code=400, detail="Invalid bid_id")

        target_dir = DATA_DIR / request.bid_id
    else:
        target_dir = DATA_DIR

    if not target_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Bid folder not found: {request.bid_id}",
        )

    try:
        return process_documents(target_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/search")
@traceable(name="API Search")
def search(
    q: str = Query(..., min_length=1),
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum_number: int | None = Query(default=None, ge=1),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Search indexed chunks and return their citation metadata."""
    try:
        results = search_documents(
            query=q,
            bid_id=bid_id,
            doc_type=doc_type,
            addendum_number=addendum_number,
            top_k=top_k,
        )
        return {
            "query": q,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask")
@traceable(name="API Ask")
def ask(request: AskRequest):
    """Answer a question or return a structured bid extraction."""
    if request.mode == "extract" and not request.bid_id:
        raise HTTPException(
            status_code=400,
            detail="bid_id is required in extraction mode",
        )

    try:
        state = graph.invoke({
            "bid_id": request.bid_id,
            "question": request.question,
            "task": request.mode,
            "retry_count": 0,
        })

        if request.mode == "extract":
            return _format_extraction(state, request.bid_id)

        sources = []
        seen = set()

        for result in state.get("retrieved_documents", []):
            source = _citation(result)
            key = (source["file"], source["page"])
            if key not in seen:
                seen.add(key)
                sources.append(source)

        return {
            "bid_id": request.bid_id,
            "question": request.question,
            "answer": state.get("answer", ""),
            "sources": sources,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e