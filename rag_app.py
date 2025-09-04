import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet HF tokenizers warning

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model (local & free for retrieval step)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Prepare data
documents = [
    "The internal project name for our next-generation recommendation engine is Firefly-X, and it will enter beta testing in Q4 of this year.",
    "Our data pipeline relies on a three-stage ETL process: extraction from Kafka topics, transformation via Spark jobs, and loading into Snowflake.",
    "The emergency server reboot procedure requires flipping the red toggle switch for 5 seconds before entering the passcode 91342.",
    "The company cafeteria changes its menu every Wednesday, and the most popular dish last quarter was the vegan mushroom lasagna.",
    "Customer support escalation beyond Tier 2 requires approval from the regional manager, except in cases involving GDPR data requests.",
    "Our proprietary chemical blend XZ-107 degrades at temperatures above 42°C, which is why it must always be stored in refrigerated containers."
]

# Step 2: Embed and store in FAISS
embeddings = embedder.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


# Step 3: Retrieve relevant chunks
def retrieve(query, k=2):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [documents[i] for i in indices[0]]


# Step 4: Ask ChatGPT with context
def rag_answer(query):
    context = "\n".join(retrieve(query))
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    print(rag_answer("What was the most popular dish last quarter?"))
