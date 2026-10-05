import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


DocumentType = Literal["bid_page", "rfp", "addendum", "specs", "affidavit"]


class DiscoveredDocument(BaseModel):
    path: str
    bid_id: str
    file_name: str
    doc_type: DocumentType
    addendum_number: int | None = None


class LoadedPage(BaseModel):
    text: str
    bid_id: str
    file_name: str
    doc_type: DocumentType
    page_number: int | None = Field(default=None, ge=1)
    addendum_number: int | None = Field(default=None, ge=1)
    document_date: str | None = None
    tables: list[str] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    text: str
    bid_id: str
    file_name: str
    doc_type: DocumentType
    page_number: int | None = None
    addendum_number: int | None = None
    document_date: str | None = None
    chunk_index: int
    chunk_id: str


def metadata_for(path: str | Path, bid_id: str | None = None) -> dict:
    source = Path(path)
    name = source.stem.casefold()

    if "addendum" in name or "addenda" in name:
        match = re.search(r"addend(?:um|a)\s*(\d+)", name)
        doc_type = "addendum"
        addendum_number = int(match.group(1)) if match else None
    elif "affidavit" in name:
        doc_type, addendum_number = "affidavit", None
    elif "spec" in name:
        doc_type, addendum_number = "specs", None
    elif source.suffix.casefold() in {".html", ".htm"}:
        doc_type, addendum_number = "bid_page", None
    else:
        doc_type, addendum_number = "rfp", None

    return {
        "bid_id": bid_id or source.parent.name,
        "file_name": source.name,
        "doc_type": doc_type,
        "addendum_number": addendum_number,
        "document_date": None,
    }