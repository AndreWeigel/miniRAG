import os
import json
import faiss
import numpy as np
from src.config import FAISS_INDEX, EMB_NPY, IDS_NPY, INDEX_DIR

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)

def build_and_save_index(embeddings: np.ndarray, ids: list[str]):
    ensure_dirs()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX))
    np.save(EMB_NPY, embeddings)
    np.save(IDS_NPY, np.array(ids))
    return index

def load_index():
    index = faiss.read_index(str(FAISS_INDEX))
    ids = np.load(IDS_NPY, allow_pickle=True)
    return index, ids
