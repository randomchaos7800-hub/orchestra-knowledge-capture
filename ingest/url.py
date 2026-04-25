"""Extract readable text from a URL → raw markdown for the compile pipeline."""
from __future__ import annotations


def extract_url(url: str) -> tuple[str, str]:
    """
    Fetch and extract main content from a URL.
    Returns (markdown_text, page_title).
    """
    try:
        import trafilatura
    except ImportError:
        raise RuntimeError("trafilatura is required for URL ingestion: pip install trafilatura")

    import trafilatura
    from trafilatura.settings import use_config

    cfg = use_config()
    cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch content from {url}. "
                         "Check the URL and ensure the site is publicly accessible.")

    # Extract with metadata
    result = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        with_metadata=True,
    )

    if not result:
        result = trafilatura.extract(downloaded, favor_recall=True)

    if not result:
        raise ValueError(f"No readable content found at {url}.")

    # Try to get title from metadata
    meta = trafilatura.extract_metadata(downloaded)
    title = (meta.title if meta and meta.title else "") or _url_to_title(url)

    # Prepend title as H1 if not already present
    if not result.startswith("# "):
        result = f"# {title}\n\nSource: {url}\n\n{result}"

    return result, title


def _url_to_title(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return path.replace("-", " ").replace("_", " ").title() or parsed.netloc
