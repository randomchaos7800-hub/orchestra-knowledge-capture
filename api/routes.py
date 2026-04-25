"""FastAPI routes for Orchestra Knowledge Capture."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

logger = logging.getLogger(__name__)
router = APIRouter()

# Set by server.py after startup
templates: Optional[Jinja2Templates] = None
kb_name: str = "Knowledge Base"

WIKI_SECTIONS = ["concepts", "entities", "research", "events"]

# Health cache: recompute at most once per 60 seconds
_health_cache: Optional[dict] = None
_health_cache_ts: float = 0.0
_HEALTH_TTL = 60.0


# ── Pydantic models ───────────────────────────────────────────────────────────

class UrlIngest(BaseModel):
    url: str

class TextIngest(BaseModel):
    text: str
    title: str = ""

class SeedRequest(BaseModel):
    topic: str
    description: str = ""
    n_articles: int = 8

class AskRequest(BaseModel):
    question: str
    top_n: int = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _article_count_before() -> int:
    wiki = config.wiki_dir()
    return sum(
        len(list((wiki / s).rglob("*.md")))
        for s in WIKI_SECTIONS if (wiki / s).exists()
    )


def _count_new(before: int) -> int:
    return _article_count_before() - before


def _save_raw(content: str, slug: str) -> Path:
    """Write content to a timestamped raw file and return the path."""
    today = datetime.now().strftime("%Y-%m-%d")
    batch = config.raw_dir() / f"ingest-{today}"
    batch.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    path = batch / f"{today}-{ts}-{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _compile(raw_path: Path) -> list[str]:
    from pipeline.compile import compile_file
    return compile_file(raw_path)


def _invalidate_health_cache() -> None:
    global _health_cache
    _health_cache = None


def _reindex() -> None:
    """Re-index after compile so search reflects new articles immediately."""
    try:
        from search.hybrid import index_articles
        index_articles(force=False, verbose=False)
    except Exception:
        pass


# ── UI ────────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates is None:
        return HTMLResponse("<h1>Templates not configured</h1>", status_code=500)
    stats = _build_health()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "kb_name": kb_name, "stats": stats},
    )


# ── Health ────────────────────────────────────────────────────────────────────

def _build_health() -> dict:
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    if _health_cache is not None and (now - _health_cache_ts) < _HEALTH_TTL:
        return _health_cache

    wiki = config.wiki_dir()
    counts = {}
    total = 0
    for s in WIKI_SECTIONS:
        n = len(list((wiki / s).rglob("*.md"))) if (wiki / s).exists() else 0
        counts[s] = n
        total += n

    # Single-pass backlink index
    orphans = 0
    try:
        backlinks: dict[str, int] = {}
        all_stems: list[str] = []
        for s in WIKI_SECTIONS:
            d = wiki / s
            if not d.exists():
                continue
            for md in d.rglob("*.md"):
                all_stems.append(md.stem)
                text = md.read_text(encoding="utf-8")
                for m in re.finditer(r"\[\[(?:[a-z_-]+:)?([^\]]+)\]\]", text):
                    slug = m.group(1).strip()
                    backlinks[slug] = backlinks.get(slug, 0) + 1
        orphans = sum(1 for stem in all_stems if backlinks.get(stem, 0) == 0)
    except Exception:
        pass

    index_file = wiki / "_index.md"
    last_updated = ""
    if index_file.exists():
        ts = index_file.stat().st_mtime
        last_updated = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

    result = {
        "total": total,
        "counts": counts,
        "orphans": orphans,
        "last_updated": last_updated,
    }
    _health_cache = result
    _health_cache_ts = now
    return result


@router.get("/api/health")
async def health():
    try:
        return _build_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingest: file upload ───────────────────────────────────────────────────────

@router.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    t0 = time.monotonic()
    suffix = Path(file.filename or "upload").suffix.lower()
    tmp = Path("/tmp") / f"kc-upload-{int(t0)}{suffix}"
    try:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds 50 MB limit ({len(data) // 1024 // 1024} MB)")
        tmp.write_bytes(data)

        if suffix == ".pdf":
            from ingest.pdf import extract_pdf
            content = extract_pdf(tmp)
        elif suffix in (".docx", ".doc"):
            from ingest.text import extract_docx
            content = extract_docx(tmp)
        elif suffix in (".md", ".txt", ""):
            from ingest.text import normalize_text
            content = normalize_text(tmp.read_text(encoding="utf-8", errors="replace"),
                                     title=Path(file.filename or "upload").stem)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        slug = re.sub(r"[^a-z0-9]+", "-", Path(file.filename or "upload").stem.lower()).strip("-")
        before = _article_count_before()
        raw_path = _save_raw(content, slug)
        articles = _compile(raw_path)
        _reindex()
        _invalidate_health_cache()
        return {
            "articles_created": _count_new(before),
            "articles": articles,
            "elapsed": round(time.monotonic() - t0, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ingest/file failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)


# ── Ingest: URL ───────────────────────────────────────────────────────────────

@router.post("/api/ingest/url")
async def ingest_url(body: UrlIngest):
    t0 = time.monotonic()
    try:
        from ingest.url import extract_url
        content, title = extract_url(body.url)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:60].strip("-") or "url-ingest"
        before = _article_count_before()
        raw_path = _save_raw(content, slug)
        articles = _compile(raw_path)
        _reindex()
        _invalidate_health_cache()
        return {
            "title": title,
            "articles_created": _count_new(before),
            "articles": articles,
            "elapsed": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        logger.exception("ingest/url failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingest: text paste ────────────────────────────────────────────────────────

@router.post("/api/ingest/text")
async def ingest_text(body: TextIngest):
    t0 = time.monotonic()
    try:
        from ingest.text import normalize_text
        content = normalize_text(body.text, title=body.title)
        slug = re.sub(r"[^a-z0-9]+", "-", (body.title or "pasted-text").lower())[:60].strip("-")
        before = _article_count_before()
        raw_path = _save_raw(content, slug)
        articles = _compile(raw_path)
        _reindex()
        _invalidate_health_cache()
        return {
            "articles_created": _count_new(before),
            "articles": articles,
            "elapsed": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        logger.exception("ingest/text failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Seed ──────────────────────────────────────────────────────────────────────

@router.post("/api/seed")
async def seed(body: SeedRequest):
    t0 = time.monotonic()
    try:
        from pipeline.seed import seed_topic
        result = seed_topic(
            topic=body.topic,
            description=body.description,
            n_articles=max(1, min(body.n_articles, 20)),
        )
        _reindex()
        _invalidate_health_cache()
        result["elapsed"] = round(time.monotonic() - t0, 1)
        return result
    except Exception as e:
        logger.exception("seed failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Search ────────────────────────────────────────────────────────────────────

@router.get("/api/search")
async def search(q: str = "", top: int = 5):
    t0 = time.monotonic()
    if not q.strip():
        return {"results": [], "elapsed": 0}
    try:
        from search.hybrid import hybrid_search, index_articles
        index_articles(force=False, verbose=False)
        results = hybrid_search(q.strip(), top_n=min(top, 20))
        return {
            "results": results,
            "elapsed": round(time.monotonic() - t0, 2),
        }
    except Exception as e:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Ask ───────────────────────────────────────────────────────────────────────

@router.post("/api/ask")
async def ask(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        from search.query import answer
        return answer(body.question.strip(), top_n=min(body.top_n, 10))
    except Exception as e:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Articles ──────────────────────────────────────────────────────────────────

@router.get("/api/articles")
async def list_articles():
    wiki = config.wiki_dir()
    articles = []
    for s in WIKI_SECTIONS:
        d = wiki / s
        if not d.exists():
            continue
        for md in sorted(d.rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8")
                # Quick frontmatter parse
                title = md.stem
                tags: list[str] = []
                updated = ""
                if text.startswith("---"):
                    end = text.find("---", 3)
                    if end != -1:
                        for line in text[3:end].splitlines():
                            if line.startswith("title:"):
                                title = line.split(":", 1)[1].strip().strip('"')
                            elif line.startswith("tags:"):
                                tags = re.findall(r"[\w-]+", line.split(":", 1)[1])
                            elif line.startswith("updated:"):
                                updated = line.split(":", 1)[1].strip()
                articles.append({
                    "slug": md.stem,
                    "title": title,
                    "section": s,
                    "tags": tags,
                    "updated": updated,
                    "path": str(md.relative_to(wiki)),
                })
            except Exception:
                pass
    return {"articles": articles, "total": len(articles)}


@router.get("/api/articles/{slug}")
async def get_article(slug: str):
    wiki = config.wiki_dir()
    for s in WIKI_SECTIONS:
        p = wiki / s / f"{slug}.md"
        if p.exists():
            try:
                return {"slug": slug, "section": s, "content": p.read_text(encoding="utf-8")}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=404, detail=f"Article '{slug}' not found")
