from agno.agent import Agent
from agno.models.google import Gemini

from agent.tools import classify_intent
from config import GEMINI_MODEL


class IntentAgent:
    """
    Responsible for: intent (refund/delivery/billing/…) + topic (short issue phrase).
    Owns: classify_intent tool.
    """

    def __init__(self):
        self.agent = Agent(
            name="IntentAgent",
            role="Classify the customer's intent (refund/delivery/billing/technical/etc.) and extract the specific issue topic.",
            model=Gemini(id=GEMINI_MODEL),
            tools=[classify_intent],
            instructions=[
                "You are an intent classification specialist for customer support.",
                "Given a ticket, call classify_intent and return the result.",
                "Focus ONLY on intent and topic — do not analyse sentiment or generate replies.",
            ],
            markdown=False,
        )

    def run(self, ticket_text: str) -> dict:
        """Returns {intent, topic}."""
        return classify_intent(ticket_text)
