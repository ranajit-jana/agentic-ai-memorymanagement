from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = "support-triage"
EMBEDDING_MODEL  = "models/gemini-embedding-001"  # dim=3072
GEMINI_MODEL     = "gemini-3-flash-preview"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
