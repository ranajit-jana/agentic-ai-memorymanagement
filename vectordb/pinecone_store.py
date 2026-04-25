import time
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX
from embeddings.embed import embed_query

_index = None


def init_pinecone():
    global _index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
    _index = pc.Index(PINECONE_INDEX)
    return _index


def _get_index():
    global _index
    if _index is None:
        init_pinecone()
    return _index


def upsert_chunks(embedded_chunks: list[dict], batch_size: int = 100):
    """Batch upsert embedded chunks into Pinecone."""
    idx = _get_index()
    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i + batch_size]
        idx.upsert(vectors=batch)


def semantic_search(query_text: str, top_k: int = 5, filter: dict = None) -> list[dict]:
    """
    Embed query and retrieve top-k most similar chunks.
    Use filter to scope by source: {"source": "pdf_policy"} for policy-only search.
    """
    idx = _get_index()
    query_vec = embed_query(query_text)
    kwargs = {"vector": query_vec, "top_k": top_k, "include_metadata": True}
    if filter:
        kwargs["filter"] = filter
    result = idx.query(**kwargs)
    return [
        {"id": m.id, "score": m.score, "metadata": m.metadata}
        for m in result.matches
    ]


def policy_search(query_text: str, top_k: int = 5) -> list[dict]:
    """Semantic search scoped to PDF policy sections only."""
    return semantic_search(
        query_text,
        top_k=top_k,
        filter={"source": {"$eq": "pdf_policy"}},
    )


def ticket_search(query_text: str, top_k: int = 5) -> list[dict]:
    """Semantic search scoped to CSV tickets only."""
    return semantic_search(
        query_text,
        top_k=top_k,
        filter={"source": {"$eq": "csv_ticket"}},
    )


def hybrid_search(query_text: str, keyword: str, top_k: int = 5) -> list[dict]:
    """Semantic search with keyword filter across ticket_type or section metadata."""
    return semantic_search(
        query_text,
        top_k=top_k,
        filter={
            "$or": [
                {"ticket_type": {"$eq": keyword}},
                {"section":     {"$eq": keyword}},
            ]
        },
    )


def get_index_stats() -> dict:
    return _get_index().describe_index_stats()
