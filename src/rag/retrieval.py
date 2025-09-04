import numpy as np
from src.preprocess.embed import embed_texts
from src.preprocess.index_store import load_index

def top_k(query: str, k: int, id2chunk: dict):
    index, ids = load_index()
    q = embed_texts([query])
    D, I = index.search(q.astype(np.float32), k)
    hits = []
    for idx in I[0]:
        chunk_id = ids[idx]
        hits.append(id2chunk[str(chunk_id)])
    return hits
