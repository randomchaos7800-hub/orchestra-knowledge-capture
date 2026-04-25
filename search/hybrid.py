"""
search/hybrid.py — Orchestra Knowledge Capture
Hybrid BM25 + ChromaDB vector search with Reciprocal Rank Fusion.

Public API:
    index_articles(force=False, verbose=False) -> int
    hybrid_search(query, top_n=5) -> list[dict]
"""

import math
import os
import re
import sys
from pathlib import Path

# Suppress HuggingFace Hub noise before any HF imports.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# Config — resolved lazily so this module is importable without a live config.
# ---------------------------------------------------------------------------

def _wiki_dir() -> Path:
    import config
    return Path(config.wiki_dir())


def _chroma_dir() -> Path:
    import config
    return Path(config.chroma_dir())


COLLECTION_NAME = "wiki_articles"

# ---------------------------------------------------------------------------
# Frontmatter parser (pure-regex, no yaml dependency)
# ---------------------------------------------------------------------------

_FM_BLOCK_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
# Inline list:  key: [a, b, c]
_FM_INLINE_LIST_RE = re.compile(r"^([\w_]+):\s*\[([^\]]*)\]\s*$")
# Plain scalar: key: value
_FM_SCALAR_RE = re.compile(r"^([\w_]+):\s*(.+)$")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_text). Body has the frontmatter block stripped."""
    m = _FM_BLOCK_RE.match(text)
    if not m:
        return {}, text
    fm_block = m.group(1)
    body = text[m.end():]
    meta: dict[str, str] = {}
    for raw_line in fm_block.splitlines():
        line = raw_line.strip()
        # Skip blank lines and YAML list-item lines (- key: value).
        if not line or line.startswith("-"):
            continue
        lm = _FM_INLINE_LIST_RE.match(line)
        if lm:
            meta[lm.group(1)] = lm.group(2)
            continue
        sm = _FM_SCALAR_RE.match(line)
        if sm:
            meta[sm.group(1)] = sm.group(2)
    return meta, body


def _extract_snippet(body: str, max_chars: int = 120) -> str:
    """
    Return the first meaningful content from the body, skipping the H1 title
    line and any bold lead/opening heading.  Strips markdown markers.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip H1 title line.
        if stripped.startswith("# "):
            continue
        # Strip bold markers so the snippet is readable plain text.
        clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", stripped)
        # Strip remaining markdown link syntax [text](url) → text.
        clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
        # Strip bare inline code backticks.
        clean = clean.replace("`", "")
        return clean[:max_chars]
    return ""


# ---------------------------------------------------------------------------
# Article discovery
# ---------------------------------------------------------------------------

def _collect_articles(wiki_dir: Path) -> list[Path]:
    """Return all indexable .md files, excluding _index.md and meta/ directory."""
    result: list[Path] = []
    for p in wiki_dir.rglob("*.md"):
        if p.name == "_index.md":
            continue
        parts = p.relative_to(wiki_dir).parts
        if "meta" in parts[:-1]:
            continue
        result.append(p)
    return sorted(result)


def _rel(path: Path, wiki_dir: Path) -> str:
    return str(path.relative_to(wiki_dir))


# ---------------------------------------------------------------------------
# ChromaDB collection — persistent, cosine similarity
# ---------------------------------------------------------------------------

def _get_collection(chroma_dir: Path):
    """Return (client, collection). The embedding function uses all-MiniLM-L6-v2."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _build_entry(path: Path, rel: str, wiki_dir: Path) -> tuple[str, dict, str] | None:
    """
    Parse one article and return (doc_id, chroma_metadata, document_text).
    Returns None if the file cannot be read.

    doc_id          = relative path from wiki root, e.g. "concepts/my-article.md"
    document_text   = "{title}\\n\\n{body}" — used for embedding and BM25
    chroma_metadata = title, tags (comma-joined), section, updated, path, mtime
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"  Warning: cannot read {rel}: {exc}", file=sys.stderr)
        return None

    meta_fm, body = _parse_frontmatter(raw)

    title: str = meta_fm.get("title", path.stem.replace("-", " ").title())
    # Tags may come as an inline-list string "[a, b, c]" or plain "a, b, c".
    raw_tags: str = meta_fm.get("tags", "")
    tags: str = raw_tags.strip("[]")
    updated: str = str(meta_fm.get("updated", meta_fm.get("last_compiled", "")))
    section: str = path.parent.name  # concepts / entities / research / events

    document = f"{title}\n\n{body.strip()}"
    mtime = str(int(path.stat().st_mtime))

    chroma_meta = {
        "path": rel,
        "title": title,
        "tags": tags,
        "section": section,
        "updated": updated,
        "mtime": mtime,
    }
    return rel, chroma_meta, document


def index_articles(force: bool = False, verbose: bool = False) -> int:
    """
    Incrementally index all wiki articles into ChromaDB.

    Skips articles whose stored mtime matches the file system mtime (unless
    force=True).  Uses batch upserts of 50 documents for efficiency.

    Returns the number of articles added or updated.
    """
    wiki_dir = _wiki_dir()
    chroma_dir = _chroma_dir()
    chroma_dir.mkdir(parents=True, exist_ok=True)

    _, col = _get_collection(chroma_dir)
    articles = _collect_articles(wiki_dir)

    # Fetch already-indexed mtimes in one round-trip.
    all_ids = [_rel(p, wiki_dir) for p in articles]
    existing_mtime: dict[str, str] = {}
    try:
        fetched = col.get(ids=all_ids, include=["metadatas"])
        for doc_id, meta in zip(fetched["ids"], fetched["metadatas"]):
            existing_mtime[doc_id] = meta.get("mtime", "0")
    except Exception:
        pass  # collection may be empty or ids may not all exist yet

    to_upsert_ids: list[str] = []
    to_upsert_docs: list[str] = []
    to_upsert_metas: list[dict] = []

    for path in articles:
        rel = _rel(path, wiki_dir)
        current_mtime = str(int(path.stat().st_mtime))
        if not force and existing_mtime.get(rel) == current_mtime:
            continue  # file unchanged

        entry = _build_entry(path, rel, wiki_dir)
        if entry is None:
            continue
        doc_id, chroma_meta, document = entry
        to_upsert_ids.append(doc_id)
        to_upsert_docs.append(document)
        to_upsert_metas.append(chroma_meta)

    if not to_upsert_ids:
        if verbose:
            print(f"Index up to date — {len(articles)} articles tracked, 0 changed.")
        return 0

    batch_size = 50
    for i in range(0, len(to_upsert_ids), batch_size):
        batch_end = i + batch_size
        col.upsert(
            ids=to_upsert_ids[i:batch_end],
            documents=to_upsert_docs[i:batch_end],
            metadatas=to_upsert_metas[i:batch_end],
        )
        if verbose:
            indexed_so_far = min(batch_end, len(to_upsert_ids))
            print(f"  Indexed {indexed_so_far}/{len(to_upsert_ids)} articles...")

    if verbose:
        print(
            f"Done — {len(to_upsert_ids)} article(s) indexed/updated "
            f"({len(articles)} total)."
        )
    return len(to_upsert_ids)


# ---------------------------------------------------------------------------
# BM25 (stdlib-only, no external BM25 library)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric token split."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _bm25_scores(
    query_terms: list[str],
    docs: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """
    Compute BM25 score for each document against query_terms.

    IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)  — Robertson variant.
    """
    N = len(docs)
    if N == 0:
        return []

    tokenized: list[list[str]] = [_tokenize(d) for d in docs]
    avg_dl: float = sum(len(t) for t in tokenized) / N

    # Document frequency per query term.
    df: dict[str, int] = {}
    for term in query_terms:
        df[term] = sum(1 for t in tokenized if term in t)

    scores: list[float] = []
    for tokens in tokenized:
        dl = len(tokens)
        # Build term → frequency map for this document.
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(
                (N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0
            )
            tf_norm = (tf * (k1 + 1.0)) / (
                tf + k1 * (1.0 - b + b * dl / avg_dl)
            )
            score += idf * tf_norm
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    Score for doc d = sum_over_rankings( 1 / (k + rank(d) + 1) )
    where rank is 0-based.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


# ---------------------------------------------------------------------------
# Hybrid search — public API
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_n: int = 5) -> list[dict]:
    """
    Hybrid BM25 + vector search with Reciprocal Rank Fusion.

    Steps:
    1. Vector search via ChromaDB (up to 20 candidates).
    2. BM25 scoring on the same candidate set.
    3. RRF fusion of both ranked lists.
    4. Return top_n results sorted by fused score.

    Each result dict:
        id          — relative path from wiki root, e.g. "concepts/my-article.md"
        title       — article title
        section     — wiki section (concepts / entities / research / events)
        tags        — comma-joined tag string
        snippet     — first 120 chars of body (after H1/bold lead)
        fused_score — RRF fused score (float)
        vec_score   — cosine similarity (1 - cosine distance)
        bm25_score  — raw BM25 score
        relevance   — normalised relevance percentage (0-100, int)
    """
    chroma_dir = _chroma_dir()
    _, col = _get_collection(chroma_dir)

    total = col.count()
    if total == 0:
        return []

    n_candidates = min(20, total)

    # --- Vector search ---
    vec_results = col.query(
        query_texts=[query],
        n_results=n_candidates,
        include=["metadatas", "documents", "distances"],
    )
    vec_ids: list[str] = vec_results["ids"][0]
    vec_metas: list[dict] = vec_results["metadatas"][0]
    vec_docs: list[str] = vec_results["documents"][0]
    vec_dists: list[float] = vec_results["distances"][0]

    # Cosine distance ∈ [0, 2] for cosine space → similarity ∈ [-1, 1].
    # ChromaDB normalises embeddings so distance ∈ [0, 1]; map to [0, 1] similarity.
    vec_score_map: dict[str, float] = {
        vid: max(0.0, 1.0 - vd)
        for vid, vd in zip(vec_ids, vec_dists)
    }

    # --- BM25 on the candidate set ---
    query_terms = _tokenize(query)
    raw_bm25 = _bm25_scores(query_terms, vec_docs)
    bm25_score_map: dict[str, float] = dict(zip(vec_ids, raw_bm25))

    # Ranked lists (both already ordered from best to worst).
    vec_ranking: list[str] = vec_ids  # ChromaDB returns nearest first
    bm25_ranking: list[str] = [
        doc_id
        for doc_id, _ in sorted(bm25_score_map.items(), key=lambda x: x[1], reverse=True)
    ]

    # --- RRF fusion ---
    fused_scores = _rrf([vec_ranking, bm25_ranking], k=60)
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Normalise to relevance % relative to the top result.
    max_score = ranked[0][1] if ranked else 1.0

    # Lookup maps.
    meta_map: dict[str, dict] = dict(zip(vec_ids, vec_metas))
    doc_map: dict[str, str] = dict(zip(vec_ids, vec_docs))

    results: list[dict] = []
    for doc_id, fused_score in ranked:
        meta = meta_map.get(doc_id, {})
        doc_text = doc_map.get(doc_id, "")

        # Snippet: skip the title line that was prepended during indexing,
        # then grab the first meaningful body line.
        body_after_title = doc_text.split("\n", 2)[-1] if "\n" in doc_text else doc_text
        snippet = _extract_snippet(body_after_title)

        results.append(
            {
                "id": doc_id,
                "title": meta.get("title", doc_id),
                "section": meta.get("section", ""),
                "tags": meta.get("tags", ""),
                "snippet": snippet,
                "fused_score": fused_score,
                "vec_score": vec_score_map.get(doc_id, 0.0),
                "bm25_score": bm25_score_map.get(doc_id, 0.0),
                "relevance": int(fused_score / max_score * 100),
            }
        )

    return results
