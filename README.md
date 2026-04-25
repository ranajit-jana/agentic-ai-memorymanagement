# YNC Customer Support Triage — Multi-Agent System

AI-powered customer support triage for YNC e-commerce using **Agno**, **Gemini**, **Pinecone**, and **Streamlit**.

Tickets are automatically triaged by a team of specialised agents — each owning a narrow concern — coordinated by a Supervisor via Agno's `Team` (coordinate mode).

---

## Architecture

```
SupportTriageWorkflow  (agent/workflow.py)
        │
        └── SupervisorAgent  (agent/supervisor_agent.py)
                │
                │  Agno Team — coordinate mode
                │  Gemini routes NL queries to the right member agent
                │
                ├── SentimentAgent   → analyze_sentiment_urgency
                ├── IntentAgent      → classify_intent
                ├── PolicyAgent      → find_policy_reference + check_refund_eligibility
                ├── ReplyAgent       → generate_suggested_reply + external_lookup + DuckDuckGo + Wikipedia
                └── SearchAgent      → search_similar_tickets + CalculatorTools
```

### Agent responsibilities

| Agent | Owns | Returns |
|---|---|---|
| `SentimentAgent` | `analyze_sentiment_urgency` | `{sentiment, urgency}` |
| `IntentAgent` | `classify_intent` | `{intent, topic}` |
| `PolicyAgent` | `find_policy_reference`, `check_refund_eligibility` | policy text + `{eligible, reason}` |
| `ReplyAgent` | `generate_suggested_reply`, `external_lookup`, DuckDuckGo, Wikipedia | draft reply string |
| `SearchAgent` | `search_similar_tickets`, CalculatorTools | list of similar tickets |
| `SupervisorAgent` | Agno `Team` with all 5 members | orchestrated triage dict |

### Two execution paths

| Path | Mechanism | When used |
|---|---|---|
| Structured triage | Supervisor calls each member's `run()` directly | `run_triage()`, `batch_triage()` |
| NL queries & reports | `team.run(prompt)` — Gemini routes autonomously | `query()`, `generate_supervisor_report()` |

---

## Memory System

The system uses **Agno's built-in memory** backed by a local SQLite database (`memory/triage_memory.db`).

### How it works

```
User asks a question (Tab 3 — Chat)
        │
        ▼
team.run(query)
        │
        ├── add_history_to_context=True
        │       Past N conversation turns are injected into the prompt.
        │       The team leader "remembers" what was asked earlier in the session.
        │
        ├── Gemini routes to the right member agent(s)
        │       e.g. "find refund complaints" → SearchAgent + PolicyAgent
        │
        ▼
Response returned to user
        │
        ├── update_memory_on_run=True
        │       MemoryManager (Gemini) reads the conversation and extracts
        │       key facts, e.g. "user is investigating Feb refund spike"
        │
        ▼
Facts written to SQLite (memory/triage_memory.db)
        │
        ▼
Next conversation — extracted facts are recalled automatically
```

### Memory components

| Component | Where configured | What it does |
|---|---|---|
| `SqliteDb` | `memory/db.py` | Single shared database file for all agents |
| `MemoryManager` | `SupervisorAgent` Team | Uses Gemini to extract facts from conversations and store them |
| `db=db` | `SupervisorAgent` Team + `ReplyAgent` | Links agent to the SQLite store for session + memory persistence |
| `update_memory_on_run=True` | `SupervisorAgent` Team | Automatically saves extracted memories after every `team.run()` |
| `add_history_to_context=True` | `SupervisorAgent` Team + `ReplyAgent` | Injects past conversation turns into every new prompt |

### What gets stored

| Data | Stored in | Lifespan |
|---|---|---|
| Full chat session (all turns) | SQLite `sessions` table | Persists across Streamlit reruns and server restarts |
| Extracted facts / memories | SQLite `memories` table | Persists permanently until manually cleared |
| Reply session history | SQLite `sessions` table | Allows ReplyAgent to avoid repeating itself in a session |

### Which agents use memory

| Agent | Memory | Why |
|---|---|---|
| `SupervisorAgent` (Team) | `db` + `MemoryManager` + history | Main conversation interface — needs full memory |
| `ReplyAgent` | `db` + history | Avoids repeating identical replies in the same session |
| `SentimentAgent` | None | Stateless — always called programmatically with a fresh ticket |
| `IntentAgent` | None | Stateless — deterministic classification, no history needed |
| `PolicyAgent` | None | Stateless — rule-based refund check + vector search |
| `SearchAgent` | None | Stateless — pure vector DB lookup |

### Memory database location

```
memory/
├── __init__.py
├── db.py               # SqliteDb(db_file="memory/triage_memory.db")
└── triage_memory.db    # auto-created on first run (git-ignored)
```

> `triage_memory.db` is listed in `.gitignore` and will never be committed.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | [Agno](https://github.com/agno-agi/agno) |
| LLM | Gemini (`gemini-3-flash-preview`) via `google-genai` |
| Embeddings | `models/gemini-embedding-001` (dim=3072) |
| Vector DB | Pinecone (serverless, cosine, AWS us-east-1) |
| UI | Streamlit |
| Package manager | `uv` |

---

## Project Structure

```
agentic-ai-mod-6-assignment-1/
├── .env                              # API keys (not committed)
├── config.py                         # Constants loaded from .env
├── app6.py                           # Streamlit UI — 4-tab application
│
├── agent/
│   ├── tools.py                      # 7 tool functions (Gemini + Pinecone)
│   ├── sentiment_agent.py            # SentimentAgent
│   ├── intent_agent.py               # IntentAgent
│   ├── policy_agent.py               # PolicyAgent
│   ├── reply_agent.py                # ReplyAgent
│   ├── search_agent.py               # SearchAgent
│   ├── supervisor_agent.py           # SupervisorAgent (Agno Team)
│   └── workflow.py                   # SupportTriageWorkflow
│
├── data/
│   ├── loader.py                     # CSV / PDF / TXT loaders + Streamlit upload handler
│   ├── chunker.py                    # Sliding-window + multi-turn chunking
│   └── ingest.py                     # One-shot ingest script
│
├── embeddings/
│   └── embed.py                      # Gemini batch embedding (dim=3072)
│
├── vectordb/
│   └── pinecone_store.py             # Pinecone init + upsert + scoped search
│
├── session/
│   └── state.py                      # JSON session save/restore
│
├── memory/
│   ├── __init__.py                   # Exports shared db instance
│   ├── db.py                         # SqliteDb(db_file="memory/triage_memory.db")
│   └── triage_memory.db              # Auto-created on first run (git-ignored)
│
├── sample_data/                      # Primary data files for ingestion
│   ├── customer_support_tickets.csv  # 8469 ticket rows (17 cols)
│   ├── internal_notes.txt            # Agent chat log entries
│   ├── support_policy.pdf            # YNC policy document (8 pages)
│   └── create_policy_pdf.py          # Script to regenerate policy PDF
│
└── validation_data/                  # Additional data for testing
    ├── tickets_electronics.csv       # 10 electronics support tickets
    ├── tickets_fashion.csv           # 10 fashion/apparel tickets
    ├── tickets_home_appliances.csv   # 10 home appliance tickets
    ├── agent_notes_jan_2024.txt      # Internal agent notes — January 2024
    ├── chat_logs_refund_cases.txt    # Live chat transcripts — refund cases
    └── escalation_notes_q1_2024.txt  # Escalation desk notes — Q1 2024
```

---

## Prerequisites

- Python 3.11+
- `uv` — `curl -Ls https://astral.sh/uv/install.sh | sh`
- `GOOGLE_API_KEY` — Gemini API key
- `PINECONE_API_KEY` — Pinecone API key

---

## Setup

**1. Clone and install dependencies**

```bash
uv sync
```

Or install from scratch:

```bash
uv init .
uv add agno "streamlit==1.35.0" "google-genai>=0.8.0" "pinecone>=8.0.0" \
    "pandas>=2.0.0" "pdfplumber>=0.10.0" "python-dotenv>=1.0.0" \
    "numpy>=1.26.0" fpdf2 fastapi ddgs wikipedia
```

**2. Create `.env`**

```
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

**3. Validate config**

```bash
uv run python -c "
from config import GOOGLE_API_KEY, PINECONE_API_KEY, GEMINI_MODEL
print('Keys loaded:', bool(GOOGLE_API_KEY), bool(PINECONE_API_KEY))
print('Model:', GEMINI_MODEL)
"
```

**4. Ingest sample data into Pinecone (run once)**

```bash
uv run python -m data.ingest
```

**5. Launch the app**

```bash
uv run streamlit run app6.py
```

---

## Streamlit UI — 4 Tabs

| Tab | What it does |
|---|---|
| **Dashboard** | KPI metrics, ticket type / priority / channel bar charts |
| **Ticket Triage** | Select or paste a ticket → sentiment badge, urgency badge, intent, refund status, editable reply |
| **Chat** | Natural language Q&A over the full ticket history and policy docs |
| **Supervisor Insights** | Executive summary report, issue spike table, avg satisfaction by type, JSON download |

**Sidebar** — multi-file uploader (CSV / TXT / PDF) with Ingest & Index button.

---

## Running Validation Tests

**Single ticket triage**

```bash
uv run python -c "
from agent.workflow import SupportTriageWorkflow
wf = SupportTriageWorkflow()
r = wf.run_triage({
    'Ticket Subject': 'Cracked screen on new GoPro',
    'Ticket Description': 'Screen cracked on arrival. I want a full refund.',
    'Ticket Type': 'Technical issue',
    'Ticket Priority': 'High',
    'Date of Purchase': '2026-04-15',
})
print('Sentiment :', r['inferred_sentiment'])
print('Urgency   :', r['inferred_urgency'])
print('Intent    :', r['inferred_intent'])
print('Refund    :', r['refund_eligible'])
print('Escalate  :', r['escalate'])
"
```

**Batch triage (5 rows)**

```bash
uv run python -c "
from agent.workflow import SupportTriageWorkflow
import pandas as pd
wf = SupportTriageWorkflow()
df = pd.read_csv('sample_data/customer_support_tickets.csv').head(5)
out = wf.batch_triage(df)
print(out[['Ticket ID','inferred_sentiment','inferred_urgency','inferred_intent','refund_eligible']].to_string())
"
```

**NL query (Team routing)**

```bash
uv run python -c "
from agent.workflow import SupportTriageWorkflow
wf = SupportTriageWorkflow()
print(wf.query('What are the top 3 customer complaints this month?'))
"
```

**Ingest validation data**

```bash
uv run python -c "
from agent.workflow import SupportTriageWorkflow
wf = SupportTriageWorkflow()
for f in ['validation_data/tickets_electronics.csv',
          'validation_data/agent_notes_jan_2024.txt',
          'validation_data/escalation_notes_q1_2024.txt']:
    r = wf.ingest_and_index(f)
    print(r)
"
```

---

## Key Files Reference

| File | Role |
|---|---|
| `app6.py` | Streamlit UI entrypoint |
| `config.py` | API keys + model constants |
| `agent/tools.py` | 7 tool functions (LLM + Pinecone calls) |
| `agent/sentiment_agent.py` | Agno Agent — sentiment & urgency |
| `agent/intent_agent.py` | Agno Agent — intent classification |
| `agent/policy_agent.py` | Agno Agent — policy lookup + refund check |
| `agent/reply_agent.py` | Agno Agent — reply drafting + web search |
| `agent/search_agent.py` | Agno Agent — similar ticket retrieval |
| `agent/supervisor_agent.py` | Agno Team coordinating all 5 agents |
| `agent/workflow.py` | End-to-end pipeline (ingest → triage → report) |
| `data/loader.py` | CSV / PDF / TXT loaders + Streamlit upload handler |
| `data/chunker.py` | Sliding-window + multi-turn text chunking |
| `embeddings/embed.py` | Gemini batch embedding (dim=3072) |
| `vectordb/pinecone_store.py` | Pinecone index + upsert + scoped search |
| `session/state.py` | JSON session save/restore |
| `memory/db.py` | Shared `SqliteDb` for Agno memory persistence |
| `.env` | `GOOGLE_API_KEY`, `PINECONE_API_KEY` |

---

## Quick Reference

```bash
# Run app
uv run streamlit run app6.py

# Re-ingest all sample data
uv run python -m data.ingest

# Add a dependency
uv add <package>

# Run a script
uv run python <script.py>
uv run python -m <module>
```
