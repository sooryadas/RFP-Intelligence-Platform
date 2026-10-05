from pathlib import Path

from langsmith import traceable

from app.ingestion.models import DiscoveredDocument, metadata_for


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}


@traceable(name="Discover Bid Documents")
def discover_documents(data_dir: str | Path) -> list[DiscoveredDocument]:
    """Find supported documents under bid folders in DATA."""
    root = Path(data_dir)
    documents = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            continue

        metadata = metadata_for(file_path)

        documents.append(
            DiscoveredDocument(
                path=str(file_path),
                **metadata,
            )
        )

    return documents