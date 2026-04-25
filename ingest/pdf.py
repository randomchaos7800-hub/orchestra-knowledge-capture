"""Extract text from PDF files → raw markdown for the compile pipeline."""
from __future__ import annotations

from pathlib import Path


def extract_pdf(path: Path) -> str:
    """Return markdown-formatted text extracted from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is required for PDF ingestion: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"<!-- page {i} -->\n{text}")

    if not pages:
        raise ValueError(f"No text could be extracted from {path.name}. "
                         "The PDF may be image-based (scanned). Use OCR before ingesting.")

    body = "\n\n".join(pages)
    title = path.stem.replace("-", " ").replace("_", " ").title()

    return f"# {title}\n\n{body}"
