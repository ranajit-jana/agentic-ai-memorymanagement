"""
SupportTriageWorkflow
=====================
Orchestrates the full multi-agent pipeline:

    File Upload
        ↓
    load() → chunk() → embed() → upsert to Pinecone
                                        ↓
                               semantic_search()
                                        ↓
              SupervisorAgent coordinates 5 specialised sub-agents:
                  ┌─ SentimentAgent  → sentiment + urgency
                  ├─ IntentAgent     → intent + topic
                  ├─ PolicyAgent     → policy reference + refund eligibility
                  ├─ SearchAgent     → similar past tickets
                  └─ ReplyAgent      → suggested customer reply
                                        ↓
                        batch_triage(df) → enriched DataFrame
                                        ↓
                       generate_supervisor_report()
"""

import os
import pandas as pd
from agno.workflow import Workflow

from agent.supervisor_agent import SupervisorAgent
from data.loader import load_csv, load_pdf, load_txt, handle_uploaded_file
from data.chunker import chunk_csv_tickets, chunk_pdf_pages, chunk_txt_segments
from embeddings.embed import embed_texts
from vectordb.pinecone_store import init_pinecone, upsert_chunks


class SupportTriageWorkflow(Workflow):
    """End-to-end multi-agent customer support triage pipeline."""

    name: str = "SupportTriageWorkflow"
    description: str = (
        "Ingest support data, coordinate specialised triage agents, "
        "and generate supervisor reports."
    )

    def __init__(self, **kwargs):
        supervisor = SupervisorAgent()
        # Workflow.__init__ accepts only `agent`, not `team`.
        # We pass None here — all actual work is done via self._supervisor.team
        # (an Agno Team with 5 member agents) and self._supervisor.triage_ticket().
        super().__init__(agent=None, **kwargs)
        self._supervisor = supervisor
        init_pinecone()  # connect to Pinecone vector DB on startup

    # ------------------------------------------------------------------
    # Stage 1 — Ingest & Index
    # ------------------------------------------------------------------

    def ingest_and_index(self, file_path: str, file_type: str | None = None) -> dict:
        """
        Load a file, chunk it, embed it, and upsert to Pinecone.
        file_type: 'csv' | 'pdf' | 'txt'  (auto-detected if None)
        Returns summary dict with chunk and vector counts.
        """
        # auto-detect extension from the filename if caller didn't specify
        if file_type is None:
            file_type = file_path.rsplit(".", 1)[-1].lower()

        # route to the correct loader + chunker based on file format
        if file_type == "csv":
            df = load_csv(file_path)
            chunks = chunk_csv_tickets(df)
        elif file_type == "pdf":
            pages = load_pdf(file_path)
            chunks = chunk_pdf_pages(pages)
        elif file_type == "txt":
            segments = load_txt(file_path)
            chunks = chunk_txt_segments(segments)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        embedded = embed_texts(chunks)   # encode chunks into dense vectors
        upsert_chunks(embedded)          # push vectors + metadata into Pinecone

        return {
            "file":          os.path.basename(file_path),
            "file_type":     file_type,
            "chunks":        len(chunks),
            "vectors_added": len(embedded),
        }

    def ingest_uploaded_file(self, uploaded_file) -> dict:
        """
        Accept a Streamlit UploadedFile object, route and ingest it.
        Returns same summary dict as ingest_and_index.
        """
        # handle_uploaded_file normalises the Streamlit object into (data, type, filename)
        result = handle_uploaded_file(uploaded_file)
        data, file_type, filename = result["data"], result["type"], result["filename"]

        if file_type == "csv":
            chunks = chunk_csv_tickets(data)
        elif file_type == "pdf":
            chunks = chunk_pdf_pages(data)
        elif file_type == "txt":
            chunks = chunk_txt_segments(data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        embedded = embed_texts(chunks)
        upsert_chunks(embedded)

        return {
            "file":          filename,
            "file_type":     file_type,
            "chunks":        len(chunks),
            "vectors_added": len(embedded),
        }

    # ------------------------------------------------------------------
    # Stage 2 — Single Ticket Triage  (delegated to SupervisorAgent)
    # ------------------------------------------------------------------

    def run_triage(self, ticket_row: dict) -> dict:
        """
        Delegate triage of a single ticket to the SupervisorAgent,
        which fans out to the 5 specialised sub-agents in sequence.
        """
        # combine subject + description into one text block
        text = (
            str(ticket_row.get("Ticket Subject", ""))
            + " "
            + str(ticket_row.get("Ticket Description", ""))
        ).strip()

        metadata = {
            "ticket_type":      ticket_row.get("Ticket Type", "General"),
            "date_of_purchase": str(ticket_row.get("Date of Purchase", "")),
            "priority":         ticket_row.get("Ticket Priority", ""),
        }

        triage = self._supervisor.triage_ticket(text, metadata)

        # merge original row fields with all inferred fields
        return {
            **ticket_row,
            "inferred_sentiment":  triage["sentiment"],
            "inferred_urgency":    triage["urgency"],
            "inferred_intent":     triage["intent"],
            "inferred_topic":      triage["topic"],
            "refund_eligible":     triage["refund_eligible"],
            "refund_reason":       triage["refund_reason"],
            "policy_reference":    triage["policy_reference"],
            "similar_tickets":     triage["similar_tickets"],
            "suggested_reply":     triage["suggested_reply"],
            "escalate":            triage["escalate"],
        }

    # ------------------------------------------------------------------
    # Stage 3 — Batch Triage
    # ------------------------------------------------------------------

    def batch_triage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run triage on every row of a DataFrame via the SupervisorAgent."""
        results = []
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  Triaging {i}/{total} ...", end="\r")
            results.append(self.run_triage(row.to_dict()))
        print()
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Stage 4 — Supervisor Report  (SupervisorAgent's own Agno Agent)
    # ------------------------------------------------------------------

    def generate_supervisor_report(self, df: pd.DataFrame) -> str:
        """Generate an executive-level summary using the SupervisorAgent."""
        return self._supervisor.generate_summary(df)

    # ------------------------------------------------------------------
    # NL Query  (SupervisorAgent reasons with full tool set)
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Answer a natural language question about tickets or policies."""
        return self._supervisor.answer_query(question)
