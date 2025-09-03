# pipelines/ingest.py
import json
from pathlib import Path
from src.preprocess.extract_pdf import extract_pages
from src.preprocess.chunking import pack_chunks  # (or detect_sections if you switch)
from src.preprocess.embed import embed_texts
from src.preprocess.index_store import build_and_save_index
from src.config import PROC_DIR, INDEX_DIR, CHUNKS_JSONL, META_JSON, TARGET_TOKENS, OVERLAP_TOKENS

def main(pdf_path: str, doc_id: str):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(str(pdf_path))
    sections = pages.copy()

    chunks = pack_chunks(sections, target_tokens=TARGET_TOKENS,
                         overlap_tokens=OVERLAP_TOKENS, doc_id=doc_id)

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            json.dump(c.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({"doc_id": doc_id, "num_chunks": len(chunks)}, f, ensure_ascii=False, indent=2)

    texts = [c.text for c in chunks]
    ids   = [c.chunk_id for c in chunks]
    build_and_save_index(embed_texts(texts), ids)

    print(f"Ingested {len(chunks)} chunks from {pdf_path}")
    print(f"PROC_DIR:     {PROC_DIR.resolve()}")
    print(f"CHUNKS_JSONL: {CHUNKS_JSONL.resolve()}")
    print(f"META_JSON:    {META_JSON.resolve()}")
    print(f"INDEX_DIR:    {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    path = r"/Users/andreweigel/PycharmProjects/miniRAG/data/raw/oldmansea.pdf"
    main(pdf_path=path, doc_id="oldmansea")
