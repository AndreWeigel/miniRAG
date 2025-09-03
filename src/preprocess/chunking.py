import regex as re
import tiktoken
from src.models.chunk import Chunk

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(s: str) -> int:
    return len(enc.encode(s))

def is_heading(line: str) -> bool:
    if re.match(r"^\d+(\.\d+)*\s+\S", line.strip()):  # 2.1 Introduction
        return True
    if len(line) < 80 and line.strip().isupper() and re.search(r"[A-Z]", line):
        return True
    return False

def sentence_split(paragraph: str):
    parts = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [p for p in parts if p]

def detect_sections(pages):
    sections = []
    current = {"title": "Front Matter", "page_start": 1, "page_end": 1, "text": []}
    for p in pages:
        lines = p["text"].splitlines()
        for line in lines:
            if is_heading(line):
                if current["text"]:
                    current["text"] = "\n".join(current["text"]).strip()
                    sections.append(current)
                current = {"title": line.strip(), "page_start": p["page"], "page_end": p["page"], "text": []}
            else:
                current["text"].append(line)
        current["page_end"] = p["page"]
    if current["text"]:
        current["text"] = "\n".join(current["text"]).strip()
        sections.append(current)
    return sections

def pack_chunks(sections, target_tokens=500, overlap_tokens=64, doc_id="doc"):
    chunks = []
    cid = 0
    for sec in sections:
        paragraphs = [p for p in sec["text"].split("\n\n") if p.strip()]
        sentences = []
        for para in paragraphs:
            sentences.extend(sentence_split(para))

        buf, buf_toks = [], 0
        for sent in sentences:
            st = count_tokens(sent)
            if buf_toks + st <= target_tokens:
                buf.append(sent); buf_toks += st
            else:
                if buf:
                    text = " ".join(buf).strip()
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}:{cid}",
                        doc_id=doc_id,
                        text=text,

                        meta={"source": "pdf", "ingest_version": 1}
                    ))
                    cid += 1
                # overlap tail
                if overlap_tokens > 0 and buf:
                    tail, tail_toks = [], 0
                    for s in reversed(buf):
                        t = count_tokens(s)
                        if tail_toks + t <= overlap_tokens:
                            tail.insert(0, s); tail_toks += t
                        else:
                            break
                    buf = tail[:]
                    buf_toks = sum(count_tokens(s) for s in buf)
                else:
                    buf, buf_toks = [], 0
                # new sentence
                buf.append(sent); buf_toks += st
        if buf:
            text = " ".join(buf).strip()
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{cid}",
                doc_id=doc_id,
                text=text,

                meta={"source": "pdf", "ingest_version": 1}
            ))
            cid += 1
    return chunks
