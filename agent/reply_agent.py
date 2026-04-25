from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools

from agent.tools import generate_suggested_reply, external_lookup
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
            role="Draft empathetic, policy-compliant customer replies. Can look up product information via DuckDuckGo or Wikipedia if needed.",
            model=Gemini(id=GEMINI_MODEL),
            tools=[
                generate_suggested_reply,
                external_lookup,
                DuckDuckGoTools(),
                WikipediaTools(),
            ],
            instructions=[
                "You are a customer reply specialist for YNC e-commerce.",
                "Draft concise, empathetic replies that are grounded in the provided policy context.",
                "If the ticket mentions a product, use external_lookup for additional context.",
                "Focus ONLY on reply generation — do not classify intent or assess refund eligibility.",
            ],
            markdown=True,
        )

    def run(self, ticket_type: str, ticket_text: str, policy_context: str) -> str:
        """Returns a drafted reply string."""
        return generate_suggested_reply(ticket_type, ticket_text, policy_context)
