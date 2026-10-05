import re
from collections import Counter
from math import ceil

from langsmith import traceable

from app.ingestion.models import LoadedPage


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


@traceable(name="Normalize Pages")
def normalize_pages(pages: list[LoadedPage]) -> list[LoadedPage]:
    """Clean page text and remove repeated page-edge headers and footers."""
    pages_by_file: dict[str, list[LoadedPage]] = {}

    for page in pages:
        pages_by_file.setdefault(page.file_name, []).append(page)

    normalized = []

    for file_pages in pages_by_file.values():
        edge_lines = []

        for page in file_pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            edge_lines.extend(lines[:2])
            edge_lines.extend(lines[-2:])

        counts = Counter(edge_lines)
        minimum_count = max(2, ceil(len(file_pages) * 0.6))
        repeated_lines = {
            line for line, count in counts.items()
            if count >= minimum_count
        }

        for page in file_pages:
            lines = [line.strip() for line in page.text.splitlines()]
            cleaned_lines = []

            for index, line in enumerate(lines):
                is_page_edge = index < 3 or index >= len(lines) - 3
                if is_page_edge and line in repeated_lines:
                    continue
                cleaned_lines.append(line)

            clean_text = _clean_text("\n".join(cleaned_lines))
            normalized.append(page.model_copy(update={"text": clean_text}))

    return normalized