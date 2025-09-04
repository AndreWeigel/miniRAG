# miniRAG

A minimal, end‑to‑end Retrieval‑Augmented Generation (RAG) pipeline for PDFs. It extracts and cleans text from PDFs, chunks it with token‑aware packing and overlap, embeds chunks with either a local SentenceTransformer or OpenAI embeddings, indexes them in FAISS, and answers questions by retrieving top‑k chunks and generating a cited answer with OpenAI Chat Completions.

## Features
- **PDF extraction**: Uses PyMuPDF to extract page text with optional header/footer cleaning.
- **Chunking**: Token‑aware packing via `tiktoken` with configurable target size and overlap.
- **Embeddings**: Local `all‑MiniLM‑L6‑v2` by default, or OpenAI `text-embedding-3-small` when enabled.
- **Vector index**: FAISS L2 index saved to disk with aligned `embeddings.npy` and `ids.npy`.
- **Retrieval**: Top‑k nearest neighbor search for a query.
- **Generation**: OpenAI Chat Completion with strict “use provided context” system message and bracketed citations.
- **Simple pipelines**: `pipelines/ingest.py` and `pipelines/query.py` runnable as scripts.

## Project Structure
```
/miniRAG
  data/
    raw/          # place your PDFs here
    processed/    # chunks.jsonl + meta.json
    index/        # faiss.index + embeddings.npy + ids.npy
  pipelines/
    ingest.py     # build chunks + embeddings + FAISS index
    query.py      # retrieve and generate an answer
  src/
    config.py     # configuration, paths, model choices
    preprocess/
      extract_pdf.py  # PDF -> pages
      cleaning.py     # page cleaning helpers
      chunking.py     # sentence split + token‑aware packing
      embed.py        # local or OpenAI embeddings
      index_store.py  # save/load FAISS + npy sidecars
    rag/
      retrieval.py    # top‑k search
      generator.py    # LLM answer with citations
    models/
      chunk.py        # Chunk dataclass
  rag_app.py       # small in‑memory demo (toy example)
  requirements.txt
  ReadME.md
```

## Requirements
- Python 3.10+
- macOS/Linux/Windows

Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate  # on macOS/Linux
# or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Configuration
Set environment variables in a `.env` file at the project root:
```
OPENAI_API_KEY=sk-...               # required for generation; also for OpenAI embeddings if enabled
USE_OPENAI_EMBEDDINGS=false         # set to true to use OpenAI embeddings; default uses SentenceTransformers
```
Models and defaults are defined in `src/config.py`:
- `MODEL_CHAT`: `gpt-4o-mini`
- `MODEL_EMBEDDING`: `text-embedding-3-small` (used only if `USE_OPENAI_EMBEDDINGS=true`)
- `TARGET_TOKENS`: 500
- `OVERLAP_TOKENS`: 64
- Paths for `data/processed` and `data/index` outputs

## Ingestion: Build the index from a PDF
`pipelines/ingest.py` will:
1) Extract and clean pages from the PDF
2) Chunk text into token‑bounded chunks with overlap
3) Embed chunks (local or OpenAI)
4) Build a FAISS index and save artifacts

Run (example uses the included sample `data/raw/oldmansea.pdf`):
```bash
python pipelines/ingest.py
```
By default, the script’s `__main__` runs with:
- `pdf_path="data/raw/oldmansea.pdf"`
- `doc_id="oldmansea"`

To call programmatically:
```python
from pipelines.ingest import main
main(pdf_path="data/raw/oldmansea.pdf", doc_id="oldmansea")
```
Artifacts produced:
- `data/processed/chunks.jsonl` — one JSON per chunk with ids and text
- `data/processed/meta.json` — basic dataset metadata
- `data/index/faiss.index` — FAISS L2 index
- `data/index/embeddings.npy` — embedding vectors
- `data/index/ids.npy` — parallel array of chunk_ids

## Query: Ask a question with retrieval + generation
`pipelines/query.py` loads chunks and the FAISS index, retrieves top‑k matches, and asks the chat model to answer strictly from the context with citations.

Run an example question:
```bash
python pipelines/query.py
```
`__main__` contains a few example questions; edit them or call programmatically:
```python
from pipelines.query import main
main("What kind of fish does Santiago catch?", k=4)
```
Output is a single answer string that includes citations like `[doc_id:chunk_number]`.

## In‑memory toy demo
`rag_app.py` shows a minimal, self‑contained RAG loop over an in‑memory list of strings using SentenceTransformers and OpenAI chat. It’s independent of the disk‑backed pipelines and useful for quick sanity checks.

Run:
```bash
python rag_app.py
```

## How it works (Data Flow)
1. PDF → pages: `extract_pdf.extract_pages` reads pages and applies `cleaning.clean_page`.
2. Pages → chunks: `chunking.pack_chunks` sentence‑splits and packs to ~`TARGET_TOKENS` with `OVERLAP_TOKENS`.
3. Chunks → embeddings: `embed.embed_texts` uses either local SentenceTransformers or OpenAI embeddings.
4. Embeddings → FAISS: `index_store.build_and_save_index` writes `faiss.index`, `embeddings.npy`, and `ids.npy`.
5. Query → top‑k: `rag.retrieval.top_k` embeds the query and searches FAISS.
6. Context → answer: `rag.generator.answer` calls OpenAI Chat with a system instruction to only use provided context and to cite chunks.

## Tips and Troubleshooting
- Ensure `.env` has a valid `OPENAI_API_KEY` before running `query.py` or `rag_app.py`.
- CPU‑only FAISS is used (`faiss-cpu`). If import errors occur, reinstall with the pinned version from `requirements.txt`.
- If you enable OpenAI embeddings, usage costs will apply; otherwise local embeddings are free but require downloading a small SentenceTransformers model on first run.
- Chunk sizes are approximate (tokenized by `tiktoken`). Adjust `TARGET_TOKENS`/`OVERLAP_TOKENS` in `src/config.py` to tune recall/precision.

## License
MIT (add your actual license if different).
