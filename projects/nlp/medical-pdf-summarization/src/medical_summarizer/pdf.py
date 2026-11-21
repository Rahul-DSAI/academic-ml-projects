"""PDF text extraction utilities."""

from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract selectable text from a PDF.

    Scanned/image-only PDFs require OCR, which is intentionally not bundled.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No selectable text was extracted. The PDF may be scanned or image-only; "
            "run OCR before summarizing it."
        )
    return text
