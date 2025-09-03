import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

MODEL_CHAT = "gpt-4o-mini"
MODEL_EMBEDDING = "text-embedding-3-small"  # used if USE_OPENAI_EMBEDDINGS=true

# Chunking defaults
TARGET_TOKENS = 500
OVERLAP_TOKENS = 64

# Paths
# --- Anchor to project root (parent of src/) ---
ROOT_DIR  = Path(__file__).resolve().parents[1]

DATA_DIR  = ROOT_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_JSONL = PROC_DIR / "chunks.jsonl"
META_JSON    = PROC_DIR / "meta.json"
FAISS_INDEX  = INDEX_DIR / "faiss.index"
EMB_NPY      = INDEX_DIR / "embeddings.npy"
IDS_NPY      = INDEX_DIR / "ids.npy"