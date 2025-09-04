from openai import OpenAI
from src.config import OPENAI_API_KEY, MODEL_CHAT

_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers strictly using the provided context. "
    "If the answer is not in the context, say you don't know."
)

def answer(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['chunk_id']} • {c['section']} • p.{c['page_start']}-{c['page_end']}]\n{c['text']}"
        for c in context_chunks
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer with citations like [chunk_id]."
    )
    resp = _client.chat.completions.create(
        model=MODEL_CHAT,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
