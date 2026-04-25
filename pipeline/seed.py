"""
pipeline/seed.py — Orchestra Knowledge Capture
===============================================
Research topic seeder: given a plain-language domain description, generate N
structured raw source files and optionally compile them into wiki articles.

Public API
----------
    seed_topic(
        topic,
        description="",
        n_articles=8,
        raw_only=False,
    ) -> dict
        Returns {"raw_files": [...], "compiled": [...], "failed": [...]}

Two LLM steps
-------------
Step 1 — Plan: LLM returns a JSON array of {slug, title, tags, summary} objects
          that together cover the domain.
Step 2 — Generate: For each planned topic, LLM writes a structured research note
          with sections: CORE CONCEPT, KEY CLAIMS, ARCHITECTURE/APPROACH,
          CONNECTIONS, SOURCES.

Each raw file is written to:
    raw_dir() / f"seed-{slug}-{date}" / f"{date}-{slug}.md"

If not raw_only, each file is compiled via compile_file(), then recompile_stale()
is called once at the end to resolve cross-links between newly created articles.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from openai import OpenAI

import config
from pipeline.compile import compile_file, recompile_stale

logger = logging.getLogger(__name__)

# ── Raw file template ──────────────────────────────────────────────────────────

_RAW_TEMPLATE = """\
---
source_agent: seed-topic
date: {date}
topic: {slug}
tags: [{tags}]
summary: "{summary}"
---

# {title}

## CORE CONCEPT

{core_concept}

## KEY CLAIMS

{key_claims}

## ARCHITECTURE/APPROACH

{architecture}

## CONNECTIONS

{connections}

## SOURCES

{sources}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_client() -> OpenAI:
    return OpenAI(
        base_url=config.llm_base_url(),
        api_key=config.llm_api_key(),
    )


def _llm_call(client: OpenAI, system: str, user: str, max_tokens: int) -> str:
    model = config.llm_model()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _extract_section(label: str, text: str, next_label: str = "") -> str:
    """Extract the text under a LABEL: heading up to the next heading or end."""
    if next_label:
        pattern = rf"{re.escape(label)}:\s*(.*?)(?={re.escape(next_label)}:|$)"
    else:
        pattern = rf"{re.escape(label)}:\s*(.*)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _format_bullets(raw_text: str) -> str:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    return "\n".join(f"- {l.lstrip('-•*').strip()}" for l in lines if l)


def _format_connections(raw_text: str) -> str:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    # Prefer lines that already contain ' — '; otherwise wrap them
    formatted = []
    for l in lines[:5]:
        if " — " in l or " - " in l:
            formatted.append(f"- {l.lstrip('-•*').strip()}")
        else:
            formatted.append(f"- {l}")
    return "\n".join(formatted)


# ── Plan generation ────────────────────────────────────────────────────────────


def _plan_topics(
    client: OpenAI,
    topic: str,
    description: str,
    n_articles: int,
) -> list[dict]:
    """
    Ask the LLM to identify n_articles distinct subtopics within the domain.
    Returns list of {slug, title, tags, summary} dicts.
    """
    system = (
        "You are a knowledge architect. Your job is to identify the key subtopics "
        "within a domain for a structured knowledge base. Be specific and concrete."
    )
    user = (
        f"Domain: **{topic}**\n"
        f"Description: {description}\n\n"
        f"Identify exactly {n_articles} distinct subtopics that together provide "
        "comprehensive coverage of this domain. Subtopics should span core concepts, "
        "frameworks or methodologies, tools or platforms, metrics, and common failure modes.\n\n"
        "Return ONLY a JSON array. Each object must have:\n"
        '  slug   — lowercase-hyphenated identifier (e.g. "retrieval-augmented-generation")\n'
        '  title  — title case name (e.g. "Retrieval-Augmented Generation")\n'
        "  tags   — array of 3-4 lowercase tags\n"
        "  summary — one sentence under 20 words describing the subtopic\n\n"
        "No markdown fences, no preamble. Pure JSON array only."
    )
    try:
        raw = _llm_call(client, system, user, max_tokens=1500)
        cleaned = _strip_json_fence(raw)
        topics: list[dict] = json.loads(cleaned)
        if not isinstance(topics, list):
            raise ValueError("Expected JSON array")
        return topics
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Topic planning failed for %r: %s", topic, exc)
        logger.debug("Raw plan response: %s", raw[:500] if "raw" in dir() else "")
        return []


# ── Content generation ─────────────────────────────────────────────────────────


def _generate_raw_content(
    client: OpenAI,
    domain: str,
    slug: str,
    title: str,
    summary: str,
    tags: list[str],
) -> str | None:
    """
    Generate structured research note content for a single subtopic.
    Returns the formatted raw file string, or None on failure.
    """
    system = (
        "You are a research writer building a structured knowledge base. "
        "Write dense, specific content. Every sentence must carry information. "
        "Name frameworks, authors, and companies. Use attribution for all claims."
    )
    user = (
        f"Write a structured research note for the knowledge base topic: **{title}**\n"
        f"Domain: {domain}\n"
        f"Summary: {summary}\n\n"
        "Write the following sections using EXACTLY these labels (in caps, followed by colon):\n\n"
        "CORE CONCEPT: 2-3 paragraphs explaining what this is, why it matters in this domain, "
        "and the key innovation or insight. Be specific — name frameworks, authors, companies.\n\n"
        "KEY CLAIMS: 5-6 bullet points. Each MUST have attribution: (Author/Source, Year) or "
        "(Organization, context). Include specific numbers, thresholds, or benchmarks where known.\n\n"
        "ARCHITECTURE/APPROACH: 2 paragraphs on how it works technically or operationally — "
        "components, data flows, decision points.\n\n"
        "CONNECTIONS: 3-4 lines naming related concepts in this domain and why they connect. "
        "Format: 'concept name — reason for connection'.\n\n"
        "SOURCES: 3-4 real references — papers, official docs, or well-known industry sources. "
        "Include URLs where available.\n\n"
        "Dense and specific. No filler. No hallucinated citations."
    )

    try:
        content = _llm_call(client, system, user, max_tokens=1500)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Content generation failed for %s: %s", slug, exc)
        return None

    core = _extract_section("CORE CONCEPT", content, "KEY CLAIMS")
    claims_raw = _extract_section("KEY CLAIMS", content, "ARCHITECTURE/APPROACH")
    arch = _extract_section("ARCHITECTURE/APPROACH", content, "CONNECTIONS")
    conns = _extract_section("CONNECTIONS", content, "SOURCES")
    sources = _extract_section("SOURCES", content)

    claims_formatted = _format_bullets(claims_raw) or f"- Key concept in {domain} domain."
    conns_formatted = _format_connections(conns) or f"- Relates to: {domain} fundamentals"

    return _RAW_TEMPLATE.format(
        date=date.today().isoformat(),
        slug=slug,
        tags=", ".join(tags),
        summary=summary,
        title=title,
        core_concept=core or f"Core concept for {title} in the {domain} domain.",
        key_claims=claims_formatted,
        architecture=arch or f"Standard implementation patterns for {title}.",
        connections=conns_formatted,
        sources=sources or f"- {domain} industry documentation",
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def seed_topic(
    topic: str,
    description: str = "",
    n_articles: int = 8,
    raw_only: bool = False,
) -> dict:
    """
    Seed a knowledge base domain from a plain-language topic name.

    Parameters
    ----------
    topic:
        Short domain name (e.g. "Retrieval-Augmented Generation", "Enterprise SaaS GTM").
    description:
        Optional 1-2 sentence description to guide topic selection. If omitted,
        a generic description is constructed from the topic name.
    n_articles:
        Number of subtopic raw files to generate (default: 8).
    raw_only:
        If True, write raw files but skip the compile step. Useful for reviewing
        generated content before committing it to the wiki.

    Returns
    -------
    dict with keys:
        raw_files  — list of absolute Path strings for written raw files
        compiled   — list of slugs successfully compiled (empty if raw_only)
        failed     — list of slugs that failed during content generation or compile
    """
    if not description:
        description = (
            f"Comprehensive knowledge base covering {topic} concepts, "
            "frameworks, tools, and best practices."
        )

    client = _make_client()
    raw_dir = config.raw_dir()
    today = date.today().isoformat()

    result: dict = {"raw_files": [], "compiled": [], "failed": []}

    # Step 1: Plan subtopics
    logger.info("Planning %d subtopics for domain: %s", n_articles, topic)
    topics = _plan_topics(client, topic, description, n_articles)
    if not topics:
        logger.error("Topic planning returned no results for %r", topic)
        return result

    logger.info("Planned %d topics — generating raw content...", len(topics))

    # Step 2: Generate raw content and write files
    written_files: list[tuple[str, Path]] = []  # (slug, path)

    for topic_entry in topics:
        slug = topic_entry.get("slug", "")
        title = topic_entry.get("title", slug.replace("-", " ").title())
        tags = topic_entry.get("tags", [topic.lower().split()[0]])
        summary = topic_entry.get("summary", f"Key concept in {topic}.")

        if not slug:
            logger.warning("Skipping topic entry with no slug: %s", topic_entry)
            result["failed"].append(title or "unknown")
            continue

        raw_content = _generate_raw_content(client, topic, slug, title, summary, tags)
        if not raw_content:
            result["failed"].append(slug)
            continue

        # Write raw file: raw_dir / seed-{slug}-{date} / {date}-{slug}.md
        batch_dir = raw_dir / f"seed-{slug}-{today}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        file_path = batch_dir / f"{today}-{slug}.md"

        try:
            file_path.write_text(raw_content, encoding="utf-8")
            logger.info("Wrote raw file: %s", file_path)
            result["raw_files"].append(str(file_path))
            written_files.append((slug, file_path))
        except OSError as exc:
            logger.error("Failed to write raw file %s: %s", file_path, exc)
            result["failed"].append(slug)

    if raw_only:
        logger.info(
            "raw_only=True — skipping compile. "
            "%d raw files written, %d failed.",
            len(result["raw_files"]),
            len(result["failed"]),
        )
        return result

    # Step 3: Compile each raw file
    logger.info("Compiling %d raw files...", len(written_files))
    for slug, file_path in written_files:
        try:
            touched = compile_file(file_path)
            if touched:
                logger.info("Compiled %s -> %s", slug, touched)
                result["compiled"].append(slug)
            else:
                logger.warning("Compile produced no articles for %s", slug)
                result["failed"].append(slug)
        except Exception as exc:  # noqa: BLE001
            logger.error("Compile failed for %s: %s", slug, exc)
            result["failed"].append(slug)

    # Step 4: Recompile stale to resolve cross-links between newly created articles
    if result["compiled"]:
        logger.info("Running recompile_stale() to resolve cross-links...")
        try:
            stale_touched = recompile_stale()
            if stale_touched:
                logger.info("Cross-link pass touched %d article(s)", len(stale_touched))
        except Exception as exc:  # noqa: BLE001
            logger.warning("recompile_stale() failed: %s", exc)

    logger.info(
        "seed_topic complete: %d raw files, %d compiled, %d failed.",
        len(result["raw_files"]),
        len(result["compiled"]),
        len(result["failed"]),
    )
    return result
