"""
Shared Agno SQLite database for session + memory persistence.

A single SqliteDb instance is shared across all agents so that:
  - Chat history (session) is stored and replayed across Streamlit reruns
  - User memories extracted by MemoryManager persist between conversations
  - All agents share the same underlying db file (triage_memory.db)
"""

from agno.db.sqlite.sqlite import SqliteDb

# Single shared database file for the whole application
db = SqliteDb(db_file="memory/triage_memory.db")
