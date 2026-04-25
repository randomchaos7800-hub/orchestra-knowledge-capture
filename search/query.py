"""
search/query.py — Orchestra Knowledge Capture
Grounded Q&A using hybrid search results as context.

Public API:
    answer(question, top_n=5, max_tokens=1200) -> dict
        Returns {"answer": str, "sources": [slug, ...], "elapsed": float}
"""

import time
from pathlib import Path

import httpx

from .hybrid import hybrid_search


# ---------------------------------------------------------------------------
# Config helpers — resolved at call time so this module is side-effect-free.
# ---------------------------------------------------------------------------

def _base_url() -> str:
    import config
    return config.llm_base_url().rstrip("/")


def _model() -> str:
    import config
    return config.qa_model()


def _api_key() -> str:
    import config
    return config.llm_api_key()


def _wiki_dir() -> Path:
    import config
    return Path(config.wiki_dir())


# ---------------------------------------------------------------------------
# Article reading
# ---------------------------------------------------------------------------

def _read_article(rel_id: str, max_chars: int = 2000) -> str:
    """
    Read the full content of an article by its relative id (e.g.
    "concepts/my-article.md").  Returns up to max_chars characters.
    Returns an empty string if the file cannot be read.
    """
    path = _wiki_dir() / rel_id
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except OSError:
        return ""


def _slug(rel_id: str) -> str:
    """Return the stem (filename without extension) of a relative article id."""
    return Path(rel_id).stem


# ---------------------------------------------------------------------------
# LLM call via httpx
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Answer using ONLY the provided articles. "
    "Be specific — cite article names. "
    "If the knowledge base doesn't cover the question, say so."
)

_TIMEOUT_SECONDS = 60.0


def _chat_completion(
    messages: list[dict],
    max_tokens: int,
) -> str:
    """
    POST to the configured LLM endpoint.  Returns the assistant content string.
    Raises httpx.HTTPError on network/HTTP failure, or returns "" on empty
    content (allowing the caller to retry).
    """
    payload = {
        "model": _model(),
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    url = _base_url() + "/chat/completions"

    response = httpx.post(
        url,
        json=payload,
        headers=headers,
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer(
    question: str,
    top_n: int = 5,
    max_tokens: int = 1200,
) -> dict:
    """
    Answer a question grounded in the wiki knowledge base.

    Steps:
    1. hybrid_search(question, top_n) to find the most relevant articles.
    2. Read the full content of each article (first 2000 chars).
    3. Build a context string.
    4. Call the LLM via httpx POST.
    5. If the primary call returns empty content, retry once with the same model.
    6. Return {"answer": str, "sources": [slug, ...], "elapsed": float}.

    "sources" contains the stem slugs of the articles passed as context,
    ordered by fused relevance score (most relevant first).
    """
    t0 = time.monotonic()

    # --- Retrieve ---
    hits = hybrid_search(question, top_n=top_n)

    sources: list[str] = [_slug(h["id"]) for h in hits]

    # --- Read article bodies ---
    context_parts: list[str] = []
    for hit in hits:
        content = _read_article(hit["id"])
        if content:
            header = f"[Article: {hit['title']} | {hit['id']}]"
            context_parts.append(f"{header}\n{content}")

    if not context_parts:
        elapsed = time.monotonic() - t0
        return {
            "answer": (
                "No relevant articles were found in the knowledge base "
                "for this question."
            ),
            "sources": [],
            "elapsed": round(elapsed, 3),
        }

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION: {question}\n\n"
                f"KNOWLEDGE BASE ARTICLES:\n\n{context}"
            ),
        },
    ]

    # --- Call LLM with one retry on empty content ---
    answer_text = ""
    last_error: str = ""
    for attempt in range(2):
        try:
            answer_text = _chat_completion(messages, max_tokens=max_tokens)
        except httpx.HTTPStatusError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            break
        except httpx.RequestError as exc:
            last_error = f"Request error: {exc}"
            break

        if answer_text:
            break
        # Empty content on attempt 0 — retry once.

    if not answer_text:
        answer_text = (
            last_error
            if last_error
            else "The model returned an empty response after retry."
        )

    elapsed = time.monotonic() - t0
    return {
        "answer": answer_text,
        "sources": sources,
        "elapsed": round(elapsed, 3),
    }
