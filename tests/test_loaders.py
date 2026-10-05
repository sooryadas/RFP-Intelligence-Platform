from types import SimpleNamespace
from unittest.mock import MagicMock

import app.ingestion.loaders.pdf as pdf_loader
from app.ingestion.loaders.html import parse_html


def fake_pdf_context(*pages):
    """Create a mock PDF object that supports `with pdfplumber.open(...)`."""
    pdf = MagicMock()
    pdf.__enter__.return_value.pages = list(pages)
    pdf.__exit__.return_value = False
    return pdf


def test_parse_html_extracts_text_and_tables(tmp_path):
    html_file = tmp_path / "bid.html"
    html_file.write_text(
        """
        <html>
          <body>
            <h1>Bid deadline</h1>
            <p>Submit by July 9.</p>
            <script>ignore this</script>
            <table>
              <tr><th>Item</th><th>Qty</th></tr>
              <tr><td>Laptop</td><td>30</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    pages = parse_html(str(html_file), bid_id="Bid2")

    assert len(pages) == 1
    assert pages[0].bid_id == "Bid2"
    assert "Bid deadline" in pages[0].text
    assert "Submit by July 9." in pages[0].text
    assert "ignore this" not in pages[0].text
    assert "Laptop | 30" in pages[0].text
    assert pages[0].page_number is None


def test_parse_pdf_extracts_text_tables_and_page_numbers(tmp_path, monkeypatch):
    pdf_file = tmp_path / "rfp.pdf"

    page1 = SimpleNamespace(
        extract_text=lambda layout: "Due date: July 9",
        extract_tables=lambda: [[["Item", "Qty"], ["Laptop", "30"]]],
    )
    page2 = SimpleNamespace(
        extract_text=lambda layout: "Warranty: 3 years",
        extract_tables=lambda: [],
    )

    monkeypatch.setattr(
        pdf_loader.pdfplumber,
        "open",
        lambda path: fake_pdf_context(page1, page2),
    )

    pages = pdf_loader.parse_pdf(str(pdf_file), bid_id="Bid2")

    assert len(pages) == 2
    assert pages[0].bid_id == "Bid2"
    assert pages[0].page_number == 1
    assert "Due date: July 9" in pages[0].text
    assert "Laptop | 30" in pages[0].text
    assert pages[1].page_number == 2
    assert "Warranty: 3 years" in pages[1].text
    assert pages[0].parse_errors == []


def test_parse_pdf_records_empty_page_error(tmp_path, monkeypatch):
    pdf_file = tmp_path / "rfp.pdf"

    empty_page = SimpleNamespace(
        extract_text=lambda layout: None,
        extract_tables=lambda: [],
    )

    monkeypatch.setattr(
        pdf_loader.pdfplumber,
        "open",
        lambda path: fake_pdf_context(empty_page),
    )

    pages = pdf_loader.parse_pdf(str(pdf_file), bid_id="Bid1")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""
    assert "OCR may be needed" in pages[0].parse_errors[0]