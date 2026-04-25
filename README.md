# Orchestra Knowledge Capture

**Turn any document, URL, or research topic into a queryable knowledge base — wired directly into your Orchestra wiki.**

Orchestra Knowledge Capture is an addon to [Orchestra](https://github.com/randomchaos7800-hub/orchestra). Orchestra gives you the knowledge graph. This gives you the pipe: feed it a PDF, a URL, a paste of text, or a plain-language description of any domain, and it compiles the content into structured, cross-linked wiki articles that your AI can search and answer questions against.

---

## What It Does

```
PDF / URL / Text / Research Topic
          ↓
   Knowledge Capture
   (2-pass LLM compile)
          ↓
   Orchestra Wiki
   (cross-linked articles)
          ↓
   Hybrid Search + Q&A
   (BM25 + vector + grounded answers)
```

**Ingest anything:**
- PDF documents (contracts, policies, manuals, research papers)
- Web URLs (articles, documentation, industry resources)
- Pasted text (notes, SOPs, FAQs, meeting transcripts)
- Research topics — describe a domain in plain language and get 8+ structured articles generated automatically

**Query everything:**
- Hybrid search: BM25 + semantic vector search with Reciprocal Rank Fusion
- Grounded Q&A: natural-language questions answered from your knowledge base with citations
- Full REST API for agent integration

**Use cases:**
- Law firm: feed case law, contracts, and procedure manuals → query specific clauses, precedents, and workflows
- Medical practice: feed clinical guidelines, intake forms, and insurance policies → answer staff questions grounded in your protocols
- Home care agency: feed compliance requirements, care plans, and training materials → searchable knowledge base for caregivers
- Any business: turn your scattered documents into a structured, queryable knowledge graph

---

## Requirements

- Python 3.10+
- An [Orchestra](https://github.com/randomchaos7800-hub/orchestra) wiki directory (or any compatible wiki directory)
- An OpenAI-compatible LLM API (OpenRouter, OpenAI, local llama-server, etc.)

---

## Install

```bash
git clone https://github.com/randomchaos7800-hub/orchestra-knowledge-capture
cd orchestra-knowledge-capture
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Path to your Orchestra wiki root (the folder that contains wiki/)
WIKI_DIR=/path/to/your/orchestra

# Any OpenAI-compatible API
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
OPENAI_MODEL=openai/gpt-4o-mini

# Optional: separate model for Q&A
# QA_MODEL=openai/gpt-4o-mini

# Display name in the UI
KB_NAME=My Knowledge Base
```

**Works with any OpenAI-compatible endpoint:**
- [OpenRouter](https://openrouter.ai) — access GPT-4o, Claude, Mistral, and more
- OpenAI directly
- Local inference (llama.cpp, Ollama, LM Studio) — set `OPENAI_BASE_URL=http://localhost:8080/v1`

---

## Run

```bash
python3 server.py
```

Open `http://localhost:8080` in your browser.

---

## API

All endpoints return JSON.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Article counts, orphan count, last updated |
| `POST` | `/api/ingest/file` | Upload a file (PDF, .txt, .md, .docx) |
| `POST` | `/api/ingest/url` | Scrape and ingest a URL |
| `POST` | `/api/ingest/text` | Ingest pasted text |
| `POST` | `/api/seed` | Generate articles from a topic description |
| `GET` | `/api/search?q=...&top=5` | Hybrid semantic search |
| `POST` | `/api/ask` | Grounded Q&A |
| `GET` | `/api/articles` | List all articles |
| `GET` | `/api/articles/{slug}` | Get article content |

### Example: seed a domain

```bash
curl -X POST http://localhost:8080/api/seed \
  -H "Content-Type: application/json" \
  -d '{"topic": "HIPAA Compliance for Small Practices", "n_articles": 8}'
```

### Example: ask a question

```bash
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the required elements of a Business Associate Agreement?"}'
```

---

## How It Works

### The Compile Pipeline

Every piece of ingested content goes through a two-pass LLM pipeline:

**Pass 1 — Plan:** The LLM reads the source material and returns a JSON plan identifying which wiki articles to create or update (paths, titles, tags, core concepts).

**Pass 2 — Write:** For each planned article, the LLM writes a full structured wiki article with YAML frontmatter, typed cross-reference links (`[[type:slug]]`), sourced key claims, and a dense overview.

After writing, the pipeline:
- Injects reciprocal backlinks into linked articles
- Expands core concepts to alternate phrasings (improves search recall)
- Rebuilds the wiki index

### The Search Layer

Hybrid search combines two signals with Reciprocal Rank Fusion:
- **BM25** — keyword frequency scoring, strong for exact terminology
- **Vector search** — semantic similarity via `all-MiniLM-L6-v2` (local, no API cost)

First search loads the model (~6s); subsequent searches run in under 100ms.

### The Q&A Layer

Natural language questions → retrieve top-N relevant articles → LLM answers using only wiki content → answer is grounded and cited. No hallucination from outside the knowledge base.

---

## Relation to Orchestra

Orchestra captures knowledge from AI conversations and compiles it into a structured wiki. Knowledge Capture extends that by adding:

1. **External ingestion** — bring in documents and web content, not just conversations
2. **Domain seeding** — bootstrap a topic area from scratch with a plain-language description
3. **Semantic search** — find articles by meaning, not just filename
4. **Grounded Q&A** — ask questions and get answers anchored to your wiki

They share the same wiki format: `wiki/concepts/`, `wiki/entities/`, `wiki/research/`, `wiki/events/`, YAML frontmatter, `[[type:slug]]` cross-references. Point Knowledge Capture at any Orchestra wiki directory and it works.

---

## License

MIT
