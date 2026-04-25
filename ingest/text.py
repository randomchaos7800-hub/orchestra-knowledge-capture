"""Normalize plain text, markdown, or docx content for the compile pipeline."""
from __future__ import annotations

import re
from pathlib import Path


def normalize_text(text: str, title: str = "") -> str:
    """Clean and normalize pasted text. Returns raw markdown."""
    text = text.strip()
    if not text:
        raise ValueError("Text is empty.")

    # Ensure it starts with a title heading
    if title and not text.startswith("# "):
        text = f"# {title.strip()}\n\n{text}"
    elif not text.startswith("# "):
        # Try to extract first line as title
        first_line = text.splitlines()[0].strip()
        if len(first_line) < 100 and not first_line.endswith("."):
            text = f"# {first_line}\n\n" + "\n".join(text.splitlines()[1:]).strip()

    return text


def extract_docx(path: Path) -> str:
    """Extract text from a .docx file. Returns raw markdown."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required for .docx ingestion: pip install python-docx")

    doc = Document(str(path))
    lines = []
    for para in doc.paragraphs:
        style = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            lines.append("")
            continue
        if style.startswith("Heading 1"):
            lines.append(f"# {text}")
        elif style.startswith("Heading 2"):
            lines.append(f"## {text}")
        elif style.startswith("Heading 3"):
            lines.append(f"### {text}")
        else:
            lines.append(text)

    # Collapse multiple blank lines
    content = "\n".join(lines)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    title = path.stem.replace("-", " ").replace("_", " ").title()
    if not content.startswith("# "):
        content = f"# {title}\n\n{content}"

    return content
