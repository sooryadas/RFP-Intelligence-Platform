import pdfplumber
from langsmith import traceable

from app.ingestion.models import LoadedPage, metadata_for


@traceable(name="PDF Parsing")
def parse_pdf(
    file_path: str,
    bid_id: str | None = None,
) -> list[LoadedPage]:
    """Extract PDF text and tables, preserving page citations."""
    metadata = metadata_for(file_path, bid_id)
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(layout=True) or ""
                raw_tables = page.extract_tables() or []

                tables = [
                    "\n".join(
                        " | ".join((cell or "").strip() for cell in row)
                        for row in table
                    )
                    for table in raw_tables
                ]
                tables = [table for table in tables if table.strip()]

                if tables:
                    text += "\n\n" + "\n\n".join(tables)

                errors = []
                if not text.strip():
                    errors.append("No embedded text found; OCR may be needed.")

            except Exception as e:
                text = ""
                tables = []
                errors = [f"Page extraction failed: {e}"]

            pages.append(
                LoadedPage(
                    text=text,
                    bid_id=metadata["bid_id"],
                    file_name=metadata["file_name"],
                    doc_type=metadata["doc_type"],
                    page_number=page_number,
                    addendum_number=metadata["addendum_number"],
                    document_date=metadata["document_date"],
                    tables=tables,
                    parse_errors=errors,
                )
            )

    return pages