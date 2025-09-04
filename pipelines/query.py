# Usage: python pipelines/query.py "Your question?"
import sys, json
from src.config import CHUNKS_JSONL
from src.rag.retrieval import top_k
from src.rag.generator import answer

def load_id2chunk():
    id2chunk = {}
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            id2chunk[c["chunk_id"]] = c
    return id2chunk

def main(question: str, k: int = 4):
    id2chunk = load_id2chunk()
    hits = top_k(question, k=k, id2chunk=id2chunk)
    print(answer(question, hits))

if __name__ == "__main__":
    question = "What is the main characters name?"
    question = "Whats is the book about?"
    question = ("What kind of fish does santiago catch?")
    main(question)
