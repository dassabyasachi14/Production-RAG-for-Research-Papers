# Production RAG for Research Papers

A production-grade Retrieval-Augmented Generation (RAG) application for analysing academic research papers. Upload a PDF, ask questions, and receive grounded answers with inline citations — backed by hybrid search, cross-encoder reranking, and hallucination prevention.

Built on **Google Gemini APIs** (free tier) with a **Streamlit** UI. Runs on Python 3.14+ with no proprietary vector database required.

---

## Features

- **Multi-modal ingestion** — extracts text, tables, and figures from PDFs. Figures are described by Gemini Vision and indexed as searchable text.
- **Hybrid retrieval** — combines dense vector search (semantic) and BM25 (lexical) fused with Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking** — a local `ms-marco-MiniLM-L-6-v2` model scores query–chunk pairs jointly for precise final selection.
- **Citation enforcement** — every factual claim in the answer carries an `[N]` inline citation traceable to a specific page. If the document lacks sufficient evidence, the model outputs `INSUFFICIENT_EVIDENCE` rather than hallucinating.
- **Versioned prompt management** — prompts are stored as YAML files and loaded at runtime, making iteration and A/B testing straightforward without code changes.
- **Zero external vector DB** — embeddings are stored in a numpy + pickle file, keeping the stack simple and fully Python 3.14 compatible.

---

## Architecture

```
PDF Upload
 │
 ├── Text extraction    (pymupdf4llm → markdown per page)
 ├── Image extraction   (PyMuPDF → Gemini Vision → text description)
 └── Table extraction   (pdfplumber → markdown pipe table)
           │
           ▼
     Recursive chunker  (512 tokens, 50-token overlap, tiktoken)
           │
           ├── Embed chunks  (gemini-embedding-001, 3072-dim, REST API)  →  Vector store (numpy + pickle)
           └── Tokenise      (BM25Okapi)                                 →  BM25 index   (pickle)

Query
 │
 ├── Vector search  (cosine similarity, top 30)  ─┐
 └── BM25 search    (top 30)                      ├──  RRF fusion (top 20)
                                                  ┘
                                                       │
                                              Cross-encoder reranker
                                           (ms-marco-MiniLM-L-6-v2, top 8)
                                                       │
                                             Gemini 2.5 Flash
                                          (citation-enforced generation)
                                                       │
                                             Answer  +  [N] citations
```

---

## Tech Stack

| Component | Library / Model |
|---|---|
| LLM & Vision | `gemini-2.5-flash` (Google AI Studio) |
| Embeddings | `gemini-embedding-001` (Google AI Studio, 3072-dim) |
| Vector store | numpy + pickle (custom, no external DB) |
| Lexical search | `rank-bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, ~90 MB) |
| PDF text | `pymupdf4llm` + `PyMuPDF` fallback |
| PDF tables | `pdfplumber` |
| Chunking | `tiktoken` (cl100k_base, 512 tokens) |
| UI | `Streamlit` |
| Data models | `Pydantic v2` |

---

## Quickstart

### Prerequisites

- Python 3.9+ (tested on 3.14)
- A free [Google AI Studio API key](https://aistudio.google.com/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/dassabyasachi14/Production-RAG-for-Research-Papers.git
cd Production-RAG-for-Research-Papers
```

### 2. Install dependencies

```bash
# Windows (use the py launcher if python/pip are not on PATH)
py -m pip install -r requirements.txt

# macOS / Linux
pip install -r requirements.txt
```

> **First run note:** `sentence-transformers` will download the cross-encoder model (~90 MB) on startup. This happens once and is cached locally.

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set your key:

```
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

### 4. Run the app

```bash
# Windows
py -m streamlit run app.py

# macOS / Linux
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Upload a PDF** using the sidebar file uploader.
2. Click **Process Document** — the pipeline extracts text, images, and tables, then indexes everything.
3. **Ask a question** in the chat input.
4. The answer appears with inline `[N]` citations. Expand the **Sources** panel to see the exact chunk, page number, and document section each citation points to.
5. If the document doesn't contain enough evidence, the app responds with an `INSUFFICIENT_EVIDENCE` message rather than guessing.

---

## Project Structure

```
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example                    # Environment variable template
├── config/
│   ├── prompts/
│   │   ├── active.yaml             # Points to the active prompt version
│   │   └── v1/
│   │       ├── rag_answer.yaml     # Citation-enforced RAG system prompt
│   │       └── image_description.yaml  # Vision description prompt
└── src/
    ├── document_processor/
    │   ├── pdf_extractor.py        # Text extraction (pymupdf4llm + fallback)
    │   ├── image_processor.py      # Image extraction + Gemini Vision description
    │   ├── table_extractor.py      # Table extraction → markdown
    │   └── chunker.py              # Recursive token-aware chunking
    ├── indexing/
    │   ├── embedder.py             # Google gemini-embedding-001 (REST API)
    │   ├── vector_store.py         # Cosine similarity store (numpy + pickle)
    │   └── bm25_store.py           # BM25 index (rank-bm25 + pickle)
    ├── retrieval/
    │   ├── hybrid_retriever.py     # BM25 + vector → RRF fusion
    │   └── reranker.py             # Cross-encoder reranking
    ├── generation/
    │   ├── llm_client.py           # Gemini 2.5 Flash wrapper (text + vision)
    │   ├── prompt_manager.py       # Versioned YAML prompt loader
    │   └── answer_generator.py     # Citation parsing + INSUFFICIENT_EVIDENCE handling
    └── utils/
        └── models.py               # Pydantic data models
```

---

## Prompt Versioning

Prompts are stored as YAML files under `config/prompts/`. The active version is controlled by `config/prompts/active.yaml`:

```yaml
active_version: "v1"
```

To create a new prompt version:

1. Copy `config/prompts/v1/` to `config/prompts/v2/`
2. Edit the YAML files in `v2/`
3. Update `active.yaml` to `active_version: "v2"`

No code changes needed — the `PromptManager` loads the active version at runtime.

---

## Retrieval Design Notes

### Why hybrid search?

- **Vector search** captures semantic similarity but misses exact keywords (e.g., model names, acronyms, numerical values).
- **BM25** excels at exact keyword matching but misses paraphrased queries.
- Combining both with RRF consistently outperforms either alone.

### Reciprocal Rank Fusion (RRF)

```
score(chunk) = 1 / (60 + rank_vector) + 1 / (60 + rank_bm25)
```

RRF operates on **ranks**, not raw scores, so there is no scale mismatch between cosine similarity floats and BM25 scores. The constant `k=60` reduces the impact of high-ranked results from a single modality.

### Two-stage retrieval

| Stage | Method | Candidates |
|---|---|---|
| 1 — Recall | BM25 + vector (RRF) | 20 |
| 2 — Precision | Cross-encoder reranker | 8 |

The cross-encoder sees the full (query, chunk) pair together, giving substantially more accurate relevance scores than the bi-encoder used for embedding — at the cost of higher latency, which is why it runs only on the 20 pre-filtered candidates.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key |
| `STORAGE_DIR` | No | Root directory for runtime data (default: `storage/`) |
| `BM25_INDEX_PATH` | No | BM25 index directory (default: `storage/bm25_indexes`) |
| `ACTIVE_PROMPT_VERSION` | No | Override the active prompt version (default: read from `active.yaml`) |

---

## Limitations

- **Single document per session** — the retrieval can be scoped to a specific document, but the app currently focuses on one paper at a time via the sidebar.
- **Free tier rate limits** — Google AI Studio free tier: Gemini 2.5 Flash at 15 RPM / 1,500 req/day; gemini-embedding-001 at 100 RPM / 1,500 req/day. Large papers with many figures may approach the embedding limit.
- **No evaluation framework** — a RAG evaluation suite (e.g., RAGAS) is not yet implemented.
- **In-memory reranker** — the cross-encoder model is loaded into RAM on startup (~90 MB). On machines with very limited RAM this may be a concern.
