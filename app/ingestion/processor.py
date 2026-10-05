from pathlib import Path

from langsmith import traceable

from app.ingestion.chunking import chunk_pages
from app.ingestion.discover import discover_documents
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.normalize import normalize_pages
from app.search.index import index_chunks


@traceable(name="Load and Chunk Document")
def load_and_chunk_document(document) -> list:
    """Parse one discovered document, normalize its pages, and chunk them."""
    file_path = Path(document.path)

    if file_path.suffix.casefold() == ".pdf":
        pages = parse_pdf(
            file_path=document.path,
            bid_id=document.bid_id,
        )
    elif file_path.suffix.casefold() in {".html", ".htm"}:
        pages = parse_html(
            file_path=document.path,
            bid_id=document.bid_id,
        )
    else:
        return []

    pages = normalize_pages(pages)
    return chunk_pages(pages)


@traceable(name="Process Bid Documents")
def process_documents(data_dir: str | Path) -> dict:
    """Discover, parse, chunk, and index documents under a bid folder or DATA."""
    documents = discover_documents(data_dir)
    chunks = []
    errors = []

    for document in documents:
        try:
            chunks.extend(load_and_chunk_document(document))
        except Exception as e:
            errors.append({
                "file": document.file_name,
                "error": str(e),
            })

    chunks_indexed = index_chunks(chunks) if chunks else 0

    return {
        "documents_found": len(documents),
        "chunks_created": len(chunks),
        "chunks_indexed": chunks_indexed,
        "errors": errors,
    }


if __name__ == "__main__":
    result = process_documents("DATA")
    print(result)