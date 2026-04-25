from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.calculator import CalculatorTools

from agent.tools import search_similar_tickets
from config import GEMINI_MODEL


class SearchAgent:
    """
    Responsible for: finding semantically similar past tickets from Pinecone.
    Also exposes CalculatorTools for ticket-count arithmetic if needed.
    Owns: search_similar_tickets, CalculatorTools.
    """

    def __init__(self):
        self.agent = Agent(
            name="SearchAgent",
            role="Find semantically similar past support tickets from the vector database to identify patterns and precedents.",
            model=Gemini(id=GEMINI_MODEL),
            tools=[search_similar_tickets, CalculatorTools()],
            instructions=[
                "You are a ticket similarity and analytics specialist.",
                "Given a query, call search_similar_tickets to find related past tickets.",
                "Use CalculatorTools only when arithmetic on ticket counts is needed.",
                "Focus ONLY on retrieval and search — do not analyse sentiment or draft replies.",
            ],
            markdown=False,
        )

    def run(self, ticket_text: str, top_k: int = 3) -> list[dict]:
        """Returns a list of similar past ticket dicts."""
        return search_similar_tickets(ticket_text, top_k=top_k)
