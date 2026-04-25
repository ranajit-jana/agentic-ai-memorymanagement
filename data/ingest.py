"""
Ingest all files in sample_data/ into Pinecone.
Run once (or re-run to refresh the index):
    uv run python data/ingest.py
"""
import os
from data.loader import load_csv, load_pdf, load_txt
from data.chunker import chunk_csv_tickets, chunk_pdf_pages, chunk_txt_segments
from embeddings.embed import embed_texts
from vectordb.pinecone_store import init_pinecone, upsert_chunks, get_index_stats

SAMPLE_DIR = "sample_data"


def ingest_csv(path: str):
    print(f"[CSV] Loading {path} ...")
    df = load_csv(path)
    print(f"[CSV] {len(df)} tickets loaded. Chunking ...")
    chunks = chunk_csv_tickets(df)
    print(f"[CSV] {len(chunks)} chunks created. Embedding (this may take a while) ...")
    embedded = embed_texts(chunks)
    print(f"[CSV] Upserting {len(embedded)} vectors to Pinecone ...")
    upsert_chunks(embedded)
    print(f"[CSV] Done.\n")


def ingest_pdf(path: str):
    print(f"[PDF] Loading {path} ...")
    pages = load_pdf(path)
    print(f"[PDF] {len(pages)} pages loaded. Chunking by section ...")
    chunks = chunk_pdf_pages(pages, source_name="pdf_policy")
    print(f"[PDF] {len(chunks)} chunks created. Embedding ...")
    embedded = embed_texts(chunks)
    print(f"[PDF] Upserting {len(embedded)} vectors to Pinecone ...")
    upsert_chunks(embedded)
    print(f"[PDF] Done.\n")


def ingest_txt(path: str):
    print(f"[TXT] Loading {path} ...")
    segments = load_txt(path)
    print(f"[TXT] {len(segments)} segments loaded. Chunking ...")
    chunks = chunk_txt_segments(segments, source_name="txt_notes")
    print(f"[TXT] {len(chunks)} chunks created. Embedding ...")
    embedded = embed_texts(chunks)
    print(f"[TXT] Upserting {len(embedded)} vectors to Pinecone ...")
    upsert_chunks(embedded)
    print(f"[TXT] Done.\n")


def ingest_all(sample_dir: str = SAMPLE_DIR):
    print("Initialising Pinecone ...")
    init_pinecone()
    print("Pinecone ready.\n")

    for filename in sorted(os.listdir(sample_dir)):
        path = os.path.join(sample_dir, filename)
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            ingest_csv(path)
        elif ext == "pdf":
            ingest_pdf(path)
        elif ext == "txt":
            ingest_txt(path)
        else:
            print(f"[SKIP] {filename} (unsupported type)\n")

    stats = get_index_stats()
    print("Ingest complete.")
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    ingest_all()
