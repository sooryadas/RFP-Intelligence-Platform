from bs4 import BeautifulSoup
from langsmith import traceable

from app.ingestion.models import LoadedPage, metadata_for


@traceable(name="HTML Parsing")
def parse_html(
    file_path: str,
    bid_id: str | None = None,
) -> list[LoadedPage]:
    """Extract readable text and tables from an HTML bid page."""
    metadata = metadata_for(file_path, bid_id)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()

    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.decompose()

    tables = []

    for table in soup.find_all("table"):
        rows = []

        for row in table.find_all("tr"):
            cells = [
                " ".join(cell.stripped_strings)
                for cell in row.find_all(["th", "td"])
            ]
            if cells:
                rows.append(" | ".join(cells))

        if rows:
            tables.append("\n".join(rows))

        table.decompose()

    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)

    if tables:
        text += "\n\n" + "\n\n".join(tables)

    errors = []
    if not text.strip():
        errors.append("No readable text found in HTML document.")

    return [
        LoadedPage(
            text=text,
            bid_id=metadata["bid_id"],
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            page_number=None,
            addendum_number=metadata["addendum_number"],
            document_date=metadata["document_date"],
            tables=tables,
            parse_errors=errors,
        )
    ]