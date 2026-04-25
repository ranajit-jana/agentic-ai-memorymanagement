import time
from google import genai
from config import GOOGLE_API_KEY, EMBEDDING_MODEL

_client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_texts(chunks: list[dict], batch_size: int = 50) -> list[dict]:
    """
    Embed chunk dicts using Gemini gemini-embedding-001 (dim=3072).
    Sends up to batch_size texts per API call for efficiency.
    Returns list of {"id": str, "values": list[float], "metadata": dict}.
    """
    result = []
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        response = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        for chunk, emb in zip(batch, response.embeddings):
            result.append({
                "id":       chunk["id"],
                "values":   emb.values,
                "metadata": chunk.get("metadata", {}),
            })
        print(f"  Embedded {min(i + batch_size, total)}/{total}", end="\r")
        # Brief pause between batches to respect rate limits
        if i + batch_size < total:
            time.sleep(0.5)
    print()
    return result


def embed_query(query_text: str) -> list[float]:
    """Embed a single query string for semantic search."""
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query_text,
    )
    return response.embeddings[0].values
