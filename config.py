"""Central config — loads .env and exposes typed settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv  # optional, falls back to os.environ

# Load .env if present (fail silently if python-dotenv not installed)
try:
    load_dotenv()
except Exception:
    pass


def _require(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Required env var {key!r} is not set. Check your .env file.")
    return val


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── Wiki ──────────────────────────────────────────────────────────────────────

def wiki_dir() -> Path:
    p = Path(_require("WIKI_DIR"))
    if not p.exists():
        raise RuntimeError(f"WIKI_DIR={p} does not exist. Create it or point to an existing Orchestra wiki.")
    return p


def raw_dir() -> Path:
    """raw/ lives next to wiki/ in the same root."""
    d = wiki_dir().parent / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def chroma_dir() -> Path:
    d = wiki_dir().parent / ".chroma"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── LLM ───────────────────────────────────────────────────────────────────────

def llm_base_url() -> str:
    return _get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def llm_api_key() -> str:
    return _require("OPENAI_API_KEY")


def llm_model() -> str:
    return _get("OPENAI_MODEL", "openai/gpt-4o-mini")


def qa_model() -> str:
    return _get("QA_MODEL") or llm_model()


# ── Server ────────────────────────────────────────────────────────────────────

def server_host() -> str:
    return _get("HOST", "0.0.0.0")


def server_port() -> int:
    return int(_get("PORT", "8080"))


def kb_name() -> str:
    return _get("KB_NAME", "Knowledge Base")


# ── Internal paths ────────────────────────────────────────────────────────────

PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / "config"
TEMPLATES_DIR = PACKAGE_DIR / "ui" / "templates"
