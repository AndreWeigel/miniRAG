import os
import numpy as np
from src.config import USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY, MODEL_EMBEDDING
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if USE_OPENAI_EMBEDDINGS:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_API_KEY)

    def embed_texts(texts):
        # OpenAI returns 1536-d vectors for text-embedding-3-small
        resp = _client.embeddings.create(model=MODEL_EMBEDDING, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)
else:
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(texts):
        return _embedder.encode(texts, normalize_embeddings=True).astype(np.float32)
