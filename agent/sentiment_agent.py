from agno.agent import Agent
from agno.models.google import Gemini

from agent.tools import analyze_sentiment_urgency
from config import GEMINI_MODEL


class SentimentAgent:
    """
    Responsible for: sentiment (positive/negative/neutral) + urgency (low/medium/high/critical).
    Owns: analyze_sentiment_urgency tool.
    """

    def __init__(self):
        self.agent = Agent(
            name="SentimentAgent",
            role="Analyse customer sentiment (positive/negative/neutral) and urgency (low/medium/high/critical) from ticket text.",
            model=Gemini(id=GEMINI_MODEL),
            tools=[analyze_sentiment_urgency],
            instructions=[
                "You are a sentiment and urgency analysis specialist for customer support.",
                "Given a ticket, call analyze_sentiment_urgency and return the result.",
                "Focus ONLY on sentiment and urgency — do not classify intent or suggest replies.",
            ],
            markdown=False,
        )

    def run(self, ticket_text: str) -> dict:
        """Returns {sentiment, urgency}."""
        return analyze_sentiment_urgency(ticket_text)
