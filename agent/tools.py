import json
import re
from datetime import datetime, date

from google import genai
from config import GOOGLE_API_KEY, GEMINI_MODEL
from vectordb.pinecone_store import policy_search, ticket_search

_client = genai.Client(api_key=GOOGLE_API_KEY)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> str:
    response = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def _parse_json(text: str) -> dict:
    """Extract JSON from Gemini response, stripping markdown code fences if present."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Tool 1 — Sentiment & Urgency
# ---------------------------------------------------------------------------

def analyze_sentiment_urgency(ticket_text: str) -> dict:
    """Analyze sentiment and urgency level of a customer support ticket."""
    prompt = f"""Analyze the customer support ticket below and return a JSON object with:
- "sentiment": one of "positive", "negative", "neutral"
- "urgency": one of "low", "medium", "high", "critical"

Return ONLY valid JSON, no explanation.

Ticket: {ticket_text}"""
    try:
        return _parse_json(_call_gemini(prompt))
    except Exception:
        return {"sentiment": "neutral", "urgency": "medium"}


# ---------------------------------------------------------------------------
# Tool 2 — Intent Classification
# ---------------------------------------------------------------------------

def classify_intent(ticket_text: str) -> dict:
    """Classify the intent and topic of a customer support ticket."""
    prompt = f"""Classify the customer support ticket below and return a JSON object with:
- "intent": one of "refund", "delivery", "account", "technical", "billing", "product_inquiry", "cancellation", "other"
- "topic": a short phrase describing the specific issue (e.g. "cracked screen", "delayed shipment")

Return ONLY valid JSON, no explanation.

Ticket: {ticket_text}"""
    try:
        return _parse_json(_call_gemini(prompt))
    except Exception:
        return {"intent": "other", "topic": "unknown"}


# ---------------------------------------------------------------------------
# Tool 3 — Suggested Reply Generation
# ---------------------------------------------------------------------------

def generate_suggested_reply(ticket_type: str, ticket_text: str, policy_context: str) -> str:
    """Generate a professional, policy-compliant draft reply for a support ticket."""
    prompt = f"""You are a professional customer support agent for YNC e-commerce.
Write a concise, empathetic reply to the following customer ticket.
Use the policy context provided to ensure your reply is accurate and policy-compliant.

Ticket Type: {ticket_type}
Customer Message: {ticket_text}
Relevant Policy: {policy_context}

Write the reply directly without any preamble or subject line."""
    return _call_gemini(prompt)


# ---------------------------------------------------------------------------
# Tool 4 — Policy Reference Matching
# ---------------------------------------------------------------------------

def find_policy_reference(query: str) -> str:
    """Retrieve the most relevant policy sections for a given support query."""
    results = policy_search(query, top_k=3)
    if not results:
        return "No relevant policy section found."

    sections = []
    for r in results:
        section = r["metadata"].get("section", "Policy")
        text = r["metadata"].get("chunk_text", "")
        score = round(r["score"], 3)
        sections.append(f"[{section} | relevance={score}]\n{text}")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tool 5 — Refund Eligibility Calculator
# ---------------------------------------------------------------------------

def check_refund_eligibility(date_of_purchase: str, ticket_type: str) -> dict:
    """
    Calculate refund eligibility based on YNC's 30-day return policy.
    date_of_purchase must be in ISO format: YYYY-MM-DD.
    """
    try:
        purchase_date = datetime.strptime(date_of_purchase.strip(), "%Y-%m-%d").date()
        days_since = (date.today() - purchase_date).days

        if days_since <= 30:
            return {
                "eligible": True,
                "days_since_purchase": days_since,
                "reason": f"Purchase was {days_since} day(s) ago — within the 30-day return window.",
            }
        else:
            return {
                "eligible": False,
                "days_since_purchase": days_since,
                "reason": f"Purchase was {days_since} day(s) ago — outside the 30-day return window.",
            }
    except ValueError:
        return {
            "eligible": False,
            "days_since_purchase": -1,
            "reason": f"Could not parse date '{date_of_purchase}'. Expected format: YYYY-MM-DD.",
        }


# ---------------------------------------------------------------------------
# Tool 6 — Similar Ticket Search
# ---------------------------------------------------------------------------

def search_similar_tickets(query: str, top_k: int = 5) -> list[dict]:
    """Search for semantically similar past support tickets."""
    results = ticket_search(query, top_k=top_k)
    return [
        {
            "ticket_id":   r["metadata"].get("ticket_id", ""),
            "ticket_type": r["metadata"].get("ticket_type", ""),
            "priority":    r["metadata"].get("priority", ""),
            "status":      r["metadata"].get("status", ""),
            "score":       round(r["score"], 3),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Tool 7 — External Lookup (DuckDuckGo + Wikipedia fallback)
# ---------------------------------------------------------------------------

def external_lookup(query: str) -> str:
    """
    Search DuckDuckGo for external context (product info, consumer laws, etc.).
    Falls back to Wikipedia if DuckDuckGo returns no results.
    """
    try:
        from agno.tools.duckduckgo import DuckDuckGoTools
        result = DuckDuckGoTools().duckduckgo_search(query)
        if result and result.strip():
            return result
    except Exception:
        pass

    try:
        from agno.tools.wikipedia import WikipediaTools
        result = WikipediaTools().search_wikipedia(query)
        if result and result.strip():
            return result
    except Exception:
        pass

    return f"No external results found for: {query}"
