"""
SupervisorAgent
===============
Uses Agno's Team (coordinate mode) to orchestrate five specialised member agents:

    SentimentAgent  → sentiment + urgency
    IntentAgent     → intent + topic
    PolicyAgent     → policy reference + refund eligibility
    ReplyAgent      → suggested customer reply
    SearchAgent     → similar past tickets

The Team leader (Gemini) routes free-form NL queries and supervisor reports
to the right member agents autonomously.

For structured triage (where deterministic, sequential output is required),
the supervisor calls sub-agents' run() methods directly and aggregates results.

Memory:
    - db (SqliteDb)        → persists chat sessions and extracted memories to disk
    - MemoryManager        → after each NL query, Gemini extracts key facts and stores them
    - add_history_to_context → injects past conversation turns into every new prompt
    - update_memory_on_run → automatically updates user memories after each team.run()
"""

import pandas as pd
from agno.memory import MemoryManager
from agno.models.google import Gemini
from agno.team import Team, TeamMode

from agent.sentiment_agent import SentimentAgent
from agent.intent_agent import IntentAgent
from agent.policy_agent import PolicyAgent
from agent.reply_agent import ReplyAgent
from agent.search_agent import SearchAgent
from memory.db import db          # shared SqliteDb — sessions + memories
from config import GEMINI_MODEL


class SupervisorAgent:
    """
    Coordinates all sub-agents via an Agno Team with persistent memory.
    - Structured triage: calls each member agent's run() directly (deterministic order).
    - NL queries & reports: delegates to self.team.run() so Gemini routes autonomously.
    - Memory: chat history and extracted facts are persisted across Streamlit reruns.
    """

    def __init__(self):
        # Instantiate each specialised sub-agent
        self.sentiment_agent = SentimentAgent()
        self.intent_agent    = IntentAgent()
        self.policy_agent    = PolicyAgent()
        self.reply_agent     = ReplyAgent()
        self.search_agent    = SearchAgent()

        # Register all sub-agents as members of an Agno Team with memory.
        # mode=coordinate → leader picks the best member(s), crafts their sub-tasks,
        # and synthesises a final response.
        self.team = Team(
            name="SupportTriageTeam",
            mode=TeamMode.coordinate,
            model=Gemini(id=GEMINI_MODEL),
            members=[
                self.sentiment_agent.agent,
                self.intent_agent.agent,
                self.policy_agent.agent,
                self.reply_agent.agent,
                self.search_agent.agent,
            ],
            description="YNC customer support triage team. Each member is a specialist.",
            instructions=[
                "You are the team leader of a customer support triage system for YNC e-commerce.",
                "Delegate tasks to the most appropriate specialist member based on what is being asked.",
                "For sentiment questions → SentimentAgent.",
                "For intent/topic questions → IntentAgent.",
                "For policy or refund questions → PolicyAgent.",
                "For drafting replies → ReplyAgent.",
                "For finding similar past tickets → SearchAgent.",
                "Synthesise member responses into a clear, concise final answer.",
            ],
            markdown=True,
            add_member_tools_to_context=True,   # lets leader see what each member can do
            # --- Agno Memory ---
            db=db,                              # SQLite: persists sessions + memories to disk
            memory_manager=MemoryManager(
                model=Gemini(id=GEMINI_MODEL),  # uses Gemini to extract facts from conversations
            ),
            update_memory_on_run=True,          # save extracted memories after every team.run()
            add_history_to_context=True,        # inject past turns into each new prompt
        )

    # ------------------------------------------------------------------
    # Structured triage — deterministic sequential calls to sub-agents
    # ------------------------------------------------------------------

    def triage_ticket(self, ticket_text: str, metadata: dict) -> dict:
        """
        Fans out to each sub-agent in a fixed sequence and merges results.
        Uses direct run() calls (not team.run()) to guarantee structured output.

        Order:
            1. SentimentAgent  → sentiment + urgency
            2. IntentAgent     → intent + topic
            3. PolicyAgent     → policy text + refund eligibility
            4. SearchAgent     → similar past tickets
            5. ReplyAgent      → suggested reply (uses policy text from step 3)
        """
        ticket_type      = metadata.get("ticket_type", "General")
        date_of_purchase = metadata.get("date_of_purchase", "")

        # Step 1 — Sentiment & urgency
        sentiment_data = self.sentiment_agent.run(ticket_text)

        # Step 2 — Intent & topic
        intent_data = self.intent_agent.run(ticket_text)

        # Step 3 — Policy + refund (query built from intent result)
        policy_query = f"{intent_data.get('intent', '')} {intent_data.get('topic', '')}"
        policy_text  = self.policy_agent.run_policy(policy_query)
        refund_data  = self.policy_agent.run_refund(date_of_purchase, ticket_type)

        # Step 4 — Similar past tickets
        similar = self.search_agent.run(ticket_text, top_k=3)

        # Step 5 — Suggested reply (informed by policy context from step 3)
        reply = self.reply_agent.run(ticket_type, ticket_text, policy_text)

        return {
            "sentiment":        sentiment_data.get("sentiment"),
            "urgency":          sentiment_data.get("urgency"),
            "intent":           intent_data.get("intent"),
            "topic":            intent_data.get("topic"),
            "refund_eligible":  refund_data.get("eligible"),
            "refund_reason":    refund_data.get("reason"),
            "policy_reference": policy_text[:300],
            "similar_tickets":  similar,
            "suggested_reply":  reply,
            "escalate":         sentiment_data.get("urgency") == "critical",
        }

    # ------------------------------------------------------------------
    # NL query — Team leader routes to the right member agent(s)
    # ------------------------------------------------------------------

    def answer_query(self, query: str) -> str:
        """Delegates to the Team; Gemini decides which member(s) to involve."""
        response = self.team.run(query)
        return response.content

    # ------------------------------------------------------------------
    # Supervisor report — Team synthesises stats across all domains
    # ------------------------------------------------------------------

    def generate_summary(self, tickets_df: pd.DataFrame) -> str:
        """Builds a stats prompt and lets the Team generate an executive summary."""
        total           = len(tickets_df)
        type_counts     = tickets_df["Ticket Type"].value_counts().head(5).to_dict() \
                          if "Ticket Type" in tickets_df.columns else {}
        priority_counts = tickets_df["Ticket Priority"].value_counts().to_dict() \
                          if "Ticket Priority" in tickets_df.columns else {}
        avg_sat         = tickets_df["Customer Satisfaction Rating"].mean() \
                          if "Customer Satisfaction Rating" in tickets_df.columns else None

        prompt = f"""You are a customer support manager. Summarise the following ticket statistics
and highlight the top issues, urgent patterns, and recommendations.

Total tickets: {total}
Ticket type distribution: {type_counts}
Priority distribution: {priority_counts}
Average customer satisfaction: {round(avg_sat, 2) if avg_sat else 'N/A'}

Write a concise executive summary (3-5 bullet points)."""

        response = self.team.run(prompt)
        return response.content
