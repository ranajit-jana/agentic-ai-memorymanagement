from agno.agent import Agent
from agno.models.google import Gemini

from agent.tools import find_policy_reference, check_refund_eligibility
from config import GEMINI_MODEL


class PolicyAgent:
    """
    Responsible for: policy retrieval from Pinecone + refund eligibility calculation.
    Owns: find_policy_reference, check_refund_eligibility tools.
    """

    def __init__(self):
        self.agent = Agent(
            name="PolicyAgent",
            role="Retrieve relevant YNC policy sections and determine refund eligibility based on purchase date and ticket type.",
            model=Gemini(id=GEMINI_MODEL),
            tools=[find_policy_reference, check_refund_eligibility],
            instructions=[
                "You are a policy and refund eligibility specialist for YNC e-commerce.",
                "Given a query, call find_policy_reference to retrieve relevant policy sections.",
                "Given a purchase date and ticket type, call check_refund_eligibility.",
                "Focus ONLY on policy and refund — do not generate replies or classify intent.",
            ],
            markdown=False,
        )

    def run_policy(self, query: str) -> str:
        """Retrieve the most relevant policy text for a given query string."""
        return find_policy_reference(query)

    def run_refund(self, date_of_purchase: str, ticket_type: str) -> dict:
        """Returns {eligible, days_since_purchase, reason}."""
        if not date_of_purchase:
            return {"eligible": None, "reason": "No purchase date provided."}
        return check_refund_eligibility(date_of_purchase, ticket_type)
