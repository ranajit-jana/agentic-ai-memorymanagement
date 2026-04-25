from agno.agent import Agent
from agno.models.google import Gemini

from agent.tools import generate_suggested_reply, external_lookup
# external_lookup already wraps DuckDuckGo + Wikipedia with graceful fallback,
# so Agno's raw DuckDuckGoTools / WikipediaTools are not added here — they raise
# unhandled DDGSException on network failures and add noisy error logs.
from memory.db import db          # shared SqliteDb for session persistence
from config import GEMINI_MODEL


class ReplyAgent:
    """
    Responsible for: drafting empathetic, policy-compliant customer replies.
    Also handles external lookups (product info, consumer law) via DuckDuckGo / Wikipedia.
    Owns: generate_suggested_reply, external_lookup, DuckDuckGoTools, WikipediaTools.
    """

    def __init__(self):
        self.agent = Agent(
            name="ReplyAgent",
            role="Draft empathetic, policy-compliant customer replies. Uses external_lookup for product/policy context when needed.",
            model=Gemini(id=GEMINI_MODEL),
            tools=[
                generate_suggested_reply,
                external_lookup,   # handles DuckDuckGo + Wikipedia internally with fallback
            ],
            instructions=[
                "You are a customer reply specialist for YNC e-commerce.",
                "Draft concise, empathetic replies that are grounded in the provided policy context.",
                "If the ticket mentions a product, use external_lookup for additional context.",
                "Focus ONLY on reply generation — do not classify intent or assess refund eligibility.",
            ],
            markdown=True,
            # session history lets the agent recall earlier replies in the same session
            db=db,
            add_history_to_context=True,
        )

    def run(self, ticket_type: str, ticket_text: str, policy_context: str) -> str:
        """Returns a drafted reply string."""
        return generate_suggested_reply(ticket_type, ticket_text, policy_context)
