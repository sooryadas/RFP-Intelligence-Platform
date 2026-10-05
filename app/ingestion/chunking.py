from langsmith import traceable

from app.ingestion.models import DocumentChunk, LoadedPage


@traceable(name="Chunk Pages")
def chunk_pages(
    pages: list[LoadedPage],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Split pages into paragraph-aware chunks with citation metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0.")

    chunks = []

    for page in pages:
        paragraphs = [
            part.strip()
            for part in page.text.split("\n\n")
            if part.strip()
        ]

        page_chunks = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if current:
                    page_chunks.append(current.strip())
                    current = ""

                start = 0
                while start < len(paragraph):
                    page_chunks.append(paragraph[start : start + chunk_size].strip())
                    start += chunk_size - overlap
                continue

            if current and len(current) + len(paragraph) + 2 > chunk_size:
                page_chunks.append(current.strip())
                current = current[-overlap:] if overlap else ""

            current = f"{current}\n\n{paragraph}".strip()

        if current:
            page_chunks.append(current.strip())

        for chunk_index, text in enumerate(page_chunks):
            chunks.append(
                DocumentChunk(
                    text=text,
                    bid_id=page.bid_id,
                    file_name=page.file_name,
                    doc_type=page.doc_type,
                    page_number=page.page_number,
                    addendum_number=page.addendum_number,
                    document_date=page.document_date,
                    chunk_index=chunk_index,
                    chunk_id=(
                        f"{page.bid_id}:{page.file_name}:"
                        f"{page.page_number or 'html'}:{chunk_index}"
                    ),
                )
            )

    return chunks