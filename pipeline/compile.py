"""
pipeline/compile.py — Orchestra Knowledge Capture
==================================================
Two-pass LLM compile pipeline: raw markdown → structured wiki articles.

Public API
----------
    compile_file(raw_path, dry_run=False, verbose=False) -> list[str]
        Process one raw file. Returns list of wiki paths touched.

    recompile_stale() -> list[str]
        Find and recompile articles whose source is newer than last_compiled.
        Returns list of wiki paths touched.

The LLM endpoint is BYO — set via config.llm_base_url() / config.llm_api_key().
See config.py at the package root for all tuneable settings.

Two-pass flow
-------------
Pass 1 — Plan: LLM reads raw file, returns JSON listing which articles to
         create or update (paths, titles, tags, sections, core_concepts).
Pass 2 — Write: LLM writes each article as full markdown with YAML frontmatter
         and typed [[type:slug]] wikilinks.

Post-write (per article)
------------------------
1. Inject last_compiled date into frontmatter.
2. Extract [[type:slug]] links from content → inject into frontmatter links block.
3. Inject reciprocal backlinks into every linked target article.
4. Call expand_concepts() to generate alternate search terms → inject as expanded_terms.

After all articles written
--------------------------
Rebuild _index.md from current wiki state.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

# config.py lives at the package root; the caller adds that dir to sys.path.
import config

logger = logging.getLogger(__name__)

# ── Link type constants ────────────────────────────────────────────────────────

LINK_TYPES = ["references", "depends_on", "extends", "contradicts", "related"]

# When injecting a reciprocal backlink, map forward type → inverse type.
_INVERSE_LINK_TYPE: dict[str, str] = {
    "depends_on": "referenced_by",
    "extends": "referenced_by",
    "contradicts": "related",
    "references": "referenced_by",
    "related": "related",
}

# Default wiki sections scanned for existing articles.
_DEFAULT_SECTIONS = ["concepts", "entities", "events", "research"]


# ── LLM client ────────────────────────────────────────────────────────────────


def _make_client() -> OpenAI:
    """Return a fresh OpenAI-compatible client using config values."""
    return OpenAI(
        base_url=config.llm_base_url(),
        api_key=config.llm_api_key(),
    )


def _llm_call(
    client: OpenAI,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    """Single chat completion. Returns stripped response text."""
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


# ── Wiki section discovery ─────────────────────────────────────────────────────


def _get_wiki_sections(wiki_dir: Path) -> list[str]:
    """Return all top-level directories under wiki_dir that contain .md files."""
    sections = set(_DEFAULT_SECTIONS)
    if wiki_dir.exists():
        for d in wiki_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name != "meta":
                if any(d.rglob("*.md")):
                    sections.add(d.name)
    return sorted(sections)


# ── Source tracking ────────────────────────────────────────────────────────────


def _sources_path() -> Path:
    return config.wiki_dir() / "_sources.json"


def _load_sources() -> dict:
    p = _sources_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load _sources.json: %s", exc)
    return {"processed": {}}


def _save_sources(sources: dict) -> None:
    _sources_path().write_text(json.dumps(sources, indent=2), encoding="utf-8")


def _mark_processed(
    sources: dict,
    raw_path: Path,
    articles_touched: list[str],
) -> None:
    rel = str(raw_path.relative_to(config.wiki_dir().parent))
    sources["processed"][rel] = {
        "processed_at": datetime.now().isoformat(),
        "articles": articles_touched,
    }


# ── Frontmatter helpers ────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict:
    """Extract and parse YAML frontmatter. Returns {} on missing or error."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    try:
        result = yaml.safe_load(fm_text)
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def _inject_metadata(path: Path, fields: dict) -> None:
    """
    Merge *fields* into an article's YAML frontmatter and rewrite the file.
    Creates a frontmatter block if none exists. Preserves body content exactly.
    Tag lists are serialised as inline YAML; link dicts use block style.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read %s for metadata injection: %s", path, exc)
        return

    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_raw = text[3:end].strip()
            body = text[end + 3:].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_raw) or {}
            except yaml.YAMLError:
                fm = {}
        else:
            fm, body = {}, text
    else:
        fm, body = {}, text

    fm.update(fields)

    class _Dumper(yaml.Dumper):
        pass

    def _list_repr(dumper: yaml.Dumper, data: list):  # type: ignore[override]
        if data and isinstance(data[0], dict):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=False
            )
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )

    _Dumper.add_representer(list, _list_repr)

    fm_out = yaml.dump(
        fm, Dumper=_Dumper, default_flow_style=False, allow_unicode=True
    ).strip()
    path.write_text(f"---\n{fm_out}\n---\n\n{body}", encoding="utf-8")


# ── JSON fence stripping ───────────────────────────────────────────────────────


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


# ── Link extraction and backlink injection ─────────────────────────────────────


def _extract_links_from_content(content: str) -> list[dict]:
    """
    Return deduplicated list of {target, type} dicts from [[type:slug]] and
    [[slug]] wikilinks in *content*. Bare [[slug]] defaults to 'references'.
    """
    seen: dict[str, dict] = {}
    typed_pattern = r"\[\[(" + "|".join(LINK_TYPES) + r"):([^\]]+)\]\]"
    for m in re.finditer(typed_pattern, content):
        link_type, target = m.group(1), m.group(2).strip()
        slug = (
            target.rsplit("/", 1)[-1].replace(".md", "")
            if "/" in target
            else Path(target).stem
        )
        if slug not in seen:
            seen[slug] = {"target": slug, "type": link_type}
    for m in re.finditer(r"\[\[([^\]:]+)\]\]", content):
        target = m.group(1).strip()
        slug = Path(target).stem
        if slug not in seen:
            seen[slug] = {"target": slug, "type": "references"}
    return list(seen.values())


def _inject_reciprocal_backlinks(
    source_slug: str,
    links: list[dict],
    wiki_dir: Path,
) -> None:
    """
    For each outbound link from *source_slug*, add an inverse backlink entry
    into the target article's frontmatter links block. No-ops if already present.
    """
    for link in links:
        target_slug = link.get("target", "")
        link_type = link.get("type", "references")
        if not target_slug or target_slug == source_slug:
            continue

        target_path: Path | None = None
        for section in _get_wiki_sections(wiki_dir):
            candidate = wiki_dir / section / f"{target_slug}.md"
            if candidate.exists():
                target_path = candidate
                break
        if not target_path:
            continue

        inverse_type = _INVERSE_LINK_TYPE.get(link_type, "related")

        try:
            text = target_path.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            existing: list = fm.get("links", [])
            if not isinstance(existing, list):
                existing = []

            existing_pairs = {
                (lnk.get("target"), lnk.get("type"))
                for lnk in existing
                if isinstance(lnk, dict)
            }
            if (source_slug, inverse_type) not in existing_pairs:
                existing.append({"target": source_slug, "type": inverse_type})
                _inject_metadata(target_path, {"links": existing})
                logger.debug(
                    "Backlink: %s <- %s (%s)", target_slug, source_slug, inverse_type
                )
        except OSError as exc:
            logger.warning("Failed to inject backlink into %s: %s", target_slug, exc)


# ── Concept expansion ──────────────────────────────────────────────────────────


def expand_concepts(client: OpenAI, concepts: list[str]) -> list[str]:
    """
    Ask the LLM for 3-5 alternate phrasings per concept.
    Returns a flat deduplicated list of expansion terms, excluding originals.
    """
    if not concepts:
        return []
    prompt = (
        "Generate 3-5 alternate phrasings or semantically related terms for each of "
        f"these concepts: {', '.join(concepts)}\n"
        "Return ONLY a flat comma-separated list of terms. No explanations, no numbering, "
        "no grouping. Example: term1, term2, term3"
    )
    try:
        raw = _llm_call(
            client,
            system="You are a semantic expansion tool.",
            user=prompt,
            max_tokens=300,
        )
        terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        originals = {c.lower() for c in concepts}
        return [t for t in dict.fromkeys(terms) if t not in originals]
    except Exception as exc:  # noqa: BLE001
        logger.warning("expand_concepts failed: %s", exc)
        return []


# ── Backlink index ─────────────────────────────────────────────────────────────


def _build_backlink_index(wiki_dir: Path) -> dict[str, list[str]]:
    """
    Scan all wiki articles. Return {slug → [slugs that link to it]}.
    Prefers the structured links: frontmatter block; falls back to content scan.
    """
    backlinks: dict[str, list[str]] = {}

    for section in _get_wiki_sections(wiki_dir):
        section_dir = wiki_dir / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            source_slug = md_file.stem
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue

            fm = _parse_frontmatter(text)
            links_fm = fm.get("links", [])

            if links_fm and isinstance(links_fm, list):
                for lnk in links_fm:
                    if isinstance(lnk, dict):
                        t = lnk.get("target", "")
                        slug = Path(t).stem if t else ""
                        if slug:
                            backlinks.setdefault(slug, [])
                            if source_slug not in backlinks[slug]:
                                backlinks[slug].append(source_slug)
            else:
                for lnk in _extract_links_from_content(text):
                    slug = lnk["target"]
                    backlinks.setdefault(slug, [])
                    if source_slug not in backlinks[slug]:
                        backlinks[slug].append(source_slug)

    return backlinks


def _gather_backlink_context(
    slug: str,
    backlink_index: dict[str, list[str]],
    wiki_dir: Path,
    max_articles: int = 10,
) -> str:
    """
    Walk backlinks up to depth 2. Return a 'Related Context' block for Pass 2.
    Depth-1 first, then depth-2. Capped at max_articles total.
    Each entry includes the first 2-3 sentences of the article's Overview.
    """
    depth1 = backlink_index.get(slug, [])
    depth2 = [
        d2
        for d1 in depth1
        for d2 in backlink_index.get(d1, [])
        if d2 != slug and d2 not in depth1
    ]
    # Deduplicate depth2
    depth2 = list(dict.fromkeys(depth2))

    candidates = [(s, 1) for s in depth1] + [(s, 2) for s in depth2]

    def _sort_key(item: tuple[str, int]) -> tuple[int, str]:
        s, depth = item
        for section in _get_wiki_sections(wiki_dir):
            p = wiki_dir / section / f"{s}.md"
            if p.exists():
                try:
                    fm = _parse_frontmatter(p.read_text(encoding="utf-8"))
                    lc = str(fm.get("last_compiled", "1970-01-01"))
                    return (depth, lc)
                except Exception:  # noqa: BLE001
                    pass
        return (depth, "1970-01-01")

    candidates.sort(key=_sort_key)
    candidates = candidates[:max_articles]
    if not candidates:
        return ""

    lines = [
        "## Related Context",
        "_Articles linking to this one — use for cross-reference enrichment "
        "only. Do not shift this article's focus._",
        "",
    ]
    for s, depth in candidates:
        article_path: Path | None = None
        for section in _get_wiki_sections(wiki_dir):
            p = wiki_dir / section / f"{s}.md"
            if p.exists():
                article_path = p
                break
        if not article_path:
            continue
        try:
            text = article_path.read_text(encoding="utf-8")
        except OSError:
            continue

        body = re.sub(r"^---.*?---\n", "", text, flags=re.DOTALL).strip()
        body = re.sub(r"^#[^\n]+\n", "", body).strip()
        body = re.sub(r"^\*\*[^\n]+\*\*\n?", "", body).strip()
        body = re.sub(r"^##[^\n]+\n", "", body).strip()

        sentences = re.split(r"(?<=[.!?])\s+", body)
        excerpt = " ".join(sentences[:3]).strip()
        if len(excerpt) > 400:
            excerpt = excerpt[:397] + "..."

        lines.append(f"- **[[{s}]]** (depth {depth}): {excerpt}")

    return "\n".join(lines)


# ── Index management ───────────────────────────────────────────────────────────


def _load_index(wiki_dir: Path) -> dict[str, dict]:
    """Return {slug: {path, title, tags, updated, section}} for all wiki articles."""
    index: dict[str, dict] = {}
    for section in _get_wiki_sections(wiki_dir):
        section_dir = wiki_dir / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            slug = md_file.stem
            rel_path = str(md_file.relative_to(wiki_dir))
            try:
                fm = _parse_frontmatter(md_file.read_text(encoding="utf-8"))
                index[slug] = {
                    "path": rel_path,
                    "title": fm.get("title", slug),
                    "tags": fm.get("tags", []),
                    "updated": fm.get("updated", ""),
                    "section": section,
                }
            except OSError:
                index[slug] = {
                    "path": rel_path,
                    "title": slug,
                    "tags": [],
                    "updated": "",
                    "section": section,
                }
    return index


def _read_existing_summaries(wiki_dir: Path) -> dict[str, str]:
    """Extract slug → summary from the current _index.md."""
    summaries: dict[str, str] = {}
    index_file = wiki_dir / "_index.md"
    if not index_file.exists():
        return summaries
    for line in index_file.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"\*\*\[\[([^\]]+)\]\]\*\* — ([^.]+(?:\.[^T][^a][^g][^s])*?)"
            r"(?:\s+Tags:|\s+Updated:|$)",
            line,
        )
        if m:
            slug, summary = m.group(1), m.group(2).strip().rstrip(".")
            summaries[slug] = summary
    return summaries


def _rebuild_index(wiki_dir: Path, summaries: dict[str, str]) -> None:
    """Rewrite _index.md from current wiki state."""
    index = _load_index(wiki_dir)
    total = len(index)
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# Wiki Index",
        f"_Last updated: {now} | {total} article{'s' if total != 1 else ''}_",
        "",
        "---",
        "",
    ]
    for section in _get_wiki_sections(wiki_dir):
        articles = {s: d for s, d in index.items() if d["section"] == section}
        lines.append(f"## {section}/ ({len(articles)})")
        lines.append("")
        if not articles:
            lines.append("*(empty)*")
            lines.append("")
            continue
        for slug, data in sorted(articles.items()):
            summary = summaries.get(slug) or data.get("title", slug)
            tags_str = ", ".join(data.get("tags", [])) if data.get("tags") else ""
            updated = data.get("updated", "")
            entry = f"**[[{slug}]]** — {summary}"
            if tags_str:
                entry += f" Tags: {tags_str}."
            if updated:
                entry += f" Updated: {updated}."
            lines.append(entry)
        lines.append("")

    index_file = wiki_dir / "_index.md"
    index_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Index rebuilt: %d articles", total)


# ── Staleness detection ────────────────────────────────────────────────────────


def _staleness_check(sources: dict, wiki_dir: Path) -> list[dict]:
    """
    Return list of {article, last_compiled, newest_source, stale_days} for
    wiki articles whose last_compiled is older than the newest contributing source.
    """
    # Invert sources: article_path_str → [{source, processed_at}, ...]
    article_sources: dict[str, list[dict]] = {}
    for source_rel, entry in sources.get("processed", {}).items():
        processed_at = entry.get("processed_at", "")
        for article_path in entry.get("articles", []):
            article_sources.setdefault(article_path, []).append(
                {"source": source_rel, "processed_at": processed_at}
            )

    stale = []
    for article_rel, contributing in article_sources.items():
        article_path = wiki_dir / article_rel
        if not article_path.exists():
            continue
        try:
            fm = _parse_frontmatter(article_path.read_text(encoding="utf-8"))
        except OSError:
            continue

        last_compiled = fm.get("last_compiled", "")
        if not last_compiled:
            continue

        newest = max(contributing, key=lambda x: x["processed_at"])
        newest_date = newest["processed_at"][:10]
        if newest_date <= str(last_compiled):
            continue

        try:
            lc_dt = datetime.strptime(str(last_compiled), "%Y-%m-%d")
            ns_dt = datetime.strptime(newest_date, "%Y-%m-%d")
            stale_days = (ns_dt - lc_dt).days
        except ValueError:
            stale_days = -1

        stale.append(
            {
                "article": article_rel,
                "last_compiled": str(last_compiled),
                "newest_source": newest["source"],
                "newest_source_date": newest_date,
                "stale_days": stale_days,
            }
        )

    stale.sort(key=lambda x: x["stale_days"], reverse=True)
    return stale


# ── Core compile logic ─────────────────────────────────────────────────────────


def compile_file(
    raw_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[str]:
    """
    Process one raw file through the two-pass LLM pipeline.

    Pass 1: JSON plan — which articles to create or update.
    Pass 2: Full markdown content for each planned article.
    Post-write: metadata injection, link extraction, reciprocal backlinks,
                concept expansion, index rebuild.

    Parameters
    ----------
    raw_path:
        Absolute path to the raw markdown file.
    dry_run:
        If True, log what would happen but do not write any files.
    verbose:
        If True, log LLM prompts and truncated responses.

    Returns
    -------
    List of relative wiki paths touched (e.g. ["concepts/foo.md"]).
    """
    wiki_dir = config.wiki_dir()
    config_dir = config.CONFIG_DIR

    try:
        compile_rules = (config_dir / "compile-rules.md").read_text(encoding="utf-8")
        wiki_style = (config_dir / "wiki-style.md").read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read config files from %s: %s", config_dir, exc)
        return []

    try:
        raw_content = raw_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read raw file %s: %s", raw_path, exc)
        return []

    client = _make_client()
    today = datetime.now().strftime("%Y-%m-%d")

    # Build article list for Pass 1 context
    index = _load_index(wiki_dir)
    article_list = "\n".join(
        f"- [{d['title']}]({d['path']})" for _, d in sorted(index.items())
    ) or "(wiki is empty — create new articles freely)"

    # Build existing-article block (matched articles only, for update awareness)
    existing_parts: list[str] = []
    raw_lower = raw_content.lower()
    for slug, data in index.items():
        slug_plain = slug.replace("-", " ")
        matched = slug_plain in raw_lower or slug in raw_lower
        if not matched:
            art_path = wiki_dir / data["path"]
            if art_path.exists():
                try:
                    afm = _parse_frontmatter(art_path.read_text(encoding="utf-8"))
                    for term in afm.get("expanded_terms", []):
                        if isinstance(term, str) and term.lower() in raw_lower:
                            matched = True
                            break
                except OSError:
                    pass
        if matched:
            art_path = wiki_dir / data["path"]
            if art_path.exists():
                try:
                    art_text = art_path.read_text(encoding="utf-8")
                    if len(art_text) > 2000:
                        logger.debug(
                            "Truncating existing article %s to 2000 chars", data["path"]
                        )
                    existing_parts.append(
                        f"### EXISTING: {data['path']}\n{art_text[:2000]}"
                    )
                except OSError:
                    pass

    existing_block = (
        "\n\n".join(existing_parts) if existing_parts else "(no existing articles matched)"
    )

    # ── Pass 1: Plan ──────────────────────────────────────────────────────────

    plan_system = f"""{compile_rules}

You must respond with ONLY a JSON object. No markdown fences, no prose.
Exact structure required:
{{
  "articles": [
    {{
      "path": "concepts/slug.md",
      "action": "create",
      "title": "Human Title",
      "summary": "One sentence, under 25 words.",
      "tags": ["tag1", "tag2"],
      "sections": ["Overview", "Key Claims", "Connections"],
      "core_concepts": ["concept1", "concept2", "concept3"]
    }}
  ]
}}
Or if nothing is worth writing: {{"articles": [], "skipped_reason": "reason"}}
All string values must be single-line. core_concepts: 3-5 key terms per article."""

    if len(raw_content) > 5000:
        logger.debug("Truncating raw source %s to 5000 chars for Pass 1", raw_path.name)
    plan_user = (
        f"Plan wiki articles to create or update from this raw source.\n\n"
        f"RAW SOURCE: {raw_path.name}\n---\n{raw_content[:5000]}\n---\n\n"
        f"EXISTING WIKI ARTICLES:\n{article_list}\n\n"
        f"MATCHED EXISTING CONTENT:\n{existing_block}\n\n"
        f"Today: {today}\n\n"
        "Return the JSON plan only — no content yet."
    )

    if verbose:
        logger.info("Pass 1 prompt (%d chars):\n%s...", len(plan_user), plan_user[:400])

    try:
        plan_raw = _llm_call(client, plan_system, plan_user, max_tokens=2000)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pass 1 LLM call failed for %s: %s", raw_path.name, exc)
        return []

    if verbose:
        logger.info("Pass 1 response:\n%s", plan_raw[:500])

    # Parse plan JSON — try direct, then regex extraction, then fail gracefully
    plan_data: dict = {}
    try:
        plan_data = json.loads(_strip_json_fence(plan_raw))
    except json.JSONDecodeError as exc:
        m = re.search(r"\{.*\}", plan_raw, re.DOTALL)
        if m:
            try:
                plan_data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.error("Pass 1 JSON parse failed for %s: %s", raw_path.name, exc)
                logger.debug("Raw Pass 1 response: %s", plan_raw[:500])
                return []
        else:
            logger.error("Pass 1 JSON parse failed for %s: %s", raw_path.name, exc)
            logger.debug("Raw Pass 1 response: %s", plan_raw[:500])
            return []

    articles_plan: list[dict] = plan_data.get("articles", [])
    if not articles_plan:
        reason = plan_data.get("skipped_reason", "no articles planned")
        logger.info("Skipped %s: %s", raw_path.name, reason)
        return []

    # Build backlink index once if any updates are planned (for context injection)
    has_updates = any(a.get("action") == "update" for a in articles_plan)
    backlink_index = _build_backlink_index(wiki_dir) if has_updates else {}

    # ── Pass 2: Write articles ────────────────────────────────────────────────

    content_system = f"""{wiki_style}

Write a wiki article as plain markdown starting with YAML frontmatter (---).
No preamble, no explanation — just the article.
Use [[type:slug]] for cross-references where type is one of:
references, depends_on, extends, contradicts, related
Bare [[slug]] is valid and defaults to references."""

    touched: list[str] = []
    summaries_update: dict[str, str] = {}

    for article in articles_plan:
        path_str = article.get("path", "")
        action = article.get("action", "create")
        title = article.get("title", "")
        summary = article.get("summary", "")
        tags = article.get("tags", [])
        sections = article.get("sections", ["Overview", "Key Claims", "Connections"])
        core_concepts = article.get("core_concepts", [])
        slug = Path(path_str).stem

        if not path_str or not title:
            logger.warning("Skipping malformed plan entry: %s", article)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would %s: %s — %s", action, path_str, summary)
            touched.append(path_str)
            continue

        article_path = wiki_dir / path_str

        # For updates, load existing article text for context
        existing_text = ""
        if action == "update" and article_path.exists():
            try:
                _existing = article_path.read_text(encoding="utf-8")
                if len(_existing) > 3000:
                    logger.debug("Truncating existing %s to 3000 chars", path_str)
                existing_text = (
                    f"\n\nEXISTING ARTICLE TO UPDATE:\n{_existing[:3000]}"
                )
            except OSError:
                pass

        # Gather backlink context for updates
        backlink_context = ""
        if action == "update" and backlink_index:
            backlink_context = _gather_backlink_context(slug, backlink_index, wiki_dir)

        tags_str = ", ".join(tags)
        sections_str = "\n".join(f"- {s}" for s in sections)

        if len(raw_content) > 4000:
            logger.debug("Truncating raw source %s to 4000 chars for Pass 2", raw_path.name)
        content_user = (
            f"Write a wiki article.\n\n"
            f"Title: {title}\n"
            f"Path: {path_str}\n"
            f"Tags: {tags_str}\n"
            f"Sections:\n{sections_str}\n\n"
            f"Source material:\n---\n{raw_content[:4000]}\n---\n"
            f"Today: {today}"
            f"{existing_text}"
            f"\n{backlink_context}\n\n"
            "Start with YAML frontmatter (title, tags, updated, sources). "
            "Then write each section. Use [[type:slug]] for cross-references."
        )

        if verbose:
            logger.info("Pass 2 prompt for %s (%d chars):\n%s...",
                        path_str, len(content_user), content_user[:400])

        try:
            content = _llm_call(client, content_system, content_user, max_tokens=4000)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pass 2 LLM call failed for %s: %s", path_str, exc)
            continue

        if verbose:
            logger.info("Pass 2 response for %s:\n%s...", path_str, content[:400])

        # Strip any accidental code fence wrapping the whole article
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()

        article_path.parent.mkdir(parents=True, exist_ok=True)
        article_path.write_text(content, encoding="utf-8")
        logger.info("%s: %s", action.upper(), path_str)
        touched.append(path_str)
        if summary:
            summaries_update[slug] = summary

        # ── Post-write: inject metadata ───────────────────────────────────────
        metadata: dict = {"last_compiled": today}

        links = _extract_links_from_content(content)
        if links:
            metadata["links"] = links

        # Expand core concepts (skip if concepts unchanged and expansion already cached)
        try:
            existing_fm = _parse_frontmatter(article_path.read_text(encoding="utf-8"))
        except OSError:
            existing_fm = {}

        needs_expansion = core_concepts and (
            not existing_fm.get("expanded_terms")
            or sorted(existing_fm.get("core_concepts", [])) != sorted(core_concepts)
        )
        if needs_expansion:
            expanded = expand_concepts(client, core_concepts)
            if expanded:
                metadata["expanded_terms"] = expanded
            if core_concepts:
                metadata["core_concepts"] = core_concepts

        _inject_metadata(article_path, metadata)

        # ── Post-write: reciprocal backlinks ──────────────────────────────────
        if links:
            _inject_reciprocal_backlinks(slug, links, wiki_dir)

    # Rebuild index after all articles written
    if not dry_run and touched:
        all_summaries = _read_existing_summaries(wiki_dir)
        all_summaries.update(summaries_update)
        _rebuild_index(wiki_dir, all_summaries)

    return touched


# ── Stale recompile ────────────────────────────────────────────────────────────


def recompile_stale(dry_run: bool = False, verbose: bool = False) -> list[str]:
    """
    Find wiki articles whose source material is newer than their last_compiled
    date and recompile the contributing source files.

    Returns a flat list of all wiki paths touched across all recompile runs.
    """
    wiki_dir = config.wiki_dir()
    sources = _load_sources()
    stale = _staleness_check(sources, wiki_dir)

    if not stale:
        logger.info("No stale articles found.")
        return []

    logger.info("Found %d stale article(s) — recompiling...", len(stale))

    # Collect source files that contributed to any stale article
    stale_articles = {s["article"] for s in stale}
    sources_to_rerun: set[str] = set()
    for source_rel, entry in sources.get("processed", {}).items():
        for article in entry.get("articles", []):
            if article in stale_articles:
                sources_to_rerun.add(source_rel)

    # Resolve paths relative to the wiki parent (project root)
    project_root = wiki_dir.parent

    all_touched: list[str] = []
    for source_rel in sorted(sources_to_rerun):
        raw_path = project_root / source_rel
        if not raw_path.exists():
            logger.warning("Source file not found: %s", source_rel)
            continue
        logger.info("Recompiling: %s", source_rel)
        try:
            touched = compile_file(raw_path, dry_run=dry_run, verbose=verbose)
            all_touched.extend(touched)
            if not dry_run and touched:
                _mark_processed(sources, raw_path, touched)
                _save_sources(sources)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed recompiling %s: %s", source_rel, exc)

    if not dry_run:
        # Write a fresh stale report after recompile
        sources = _load_sources()
        remaining = _staleness_check(sources, wiki_dir)
        _write_stale_report(remaining, wiki_dir)
        if remaining:
            logger.info("Stale after recompile: %d article(s)", len(remaining))
        else:
            logger.info("All articles current after recompile.")

    return all_touched


def _write_stale_report(stale: list[dict], wiki_dir: Path) -> None:
    meta_dir = wiki_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# Stale Articles", f"_Last updated: {now}_", ""]
    if not stale:
        lines.append("*(none — all compiled articles are current)*")
    else:
        lines.append(f"{len(stale)} article(s) have newer source material:\n")
        for s in stale:
            lines.append(
                f"- **{s['article']}** — last compiled {s['last_compiled']}, "
                f"newer source {s['newest_source_date']} "
                f"({s['stale_days']}d stale) via `{s['newest_source']}`"
            )
    (meta_dir / "stale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Stale report written: %d stale article(s)", len(stale))
