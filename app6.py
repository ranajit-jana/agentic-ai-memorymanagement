import json
import os
import pandas as pd
import streamlit as st

from agent.workflow import SupportTriageWorkflow

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YNC Customer Support Triage",
    page_icon="🎯",
    layout="wide",
)

# ── Cached workflow (one instance across all reruns) ──────────────────────────
@st.cache_resource
def get_workflow():
    return SupportTriageWorkflow()

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "uploaded_df":       None,
    "chat_history":      [],
    "indexed":           False,
    "triage_results":    None,
    "supervisor_report": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Badge helpers ──────────────────────────────────────────────────────────────
_SENTIMENT = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
_URGENCY   = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}

def _badge(label: str, value: str, colour_map: dict) -> str:
    icon = colour_map.get(str(value).lower(), "⚪")
    return f"{icon} **{label}:** {value}"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — File Upload & Ingest
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📂 Data Ingestion")

    uploaded_files = st.file_uploader(
        "Upload support files",
        type=["csv", "txt", "pdf"],
        accept_multiple_files=True,
        help="CSV = tickets | TXT = agent notes | PDF = policies",
    )

    if uploaded_files and st.button("⚡ Ingest & Index", use_container_width=True):
        wf = get_workflow()
        progress = st.progress(0, text="Starting ingestion…")
        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / len(uploaded_files), text=f"Processing {f.name}…")
            summary = wf.ingest_uploaded_file(f)
            st.success(f"✅ {summary['file']} — {summary['vectors_added']} vectors")
            if summary["file_type"] == "csv":
                f.seek(0)
                st.session_state.uploaded_df = pd.read_csv(f)
        progress.empty()
        st.session_state.indexed = True
        st.balloons()

    st.caption(
        "🗄️ Pinecone index: **ready**" if st.session_state.indexed
        else "🗄️ Pinecone index: pending ingest"
    )

    st.divider()
    st.caption("YNC Support Triage Agent · v1.0")

# ── Load default CSV if nothing uploaded yet ──────────────────────────────────
if st.session_state.uploaded_df is None:
    try:
        st.session_state.uploaded_df = pd.read_csv("sample_data/customer_support_tickets.csv")
    except Exception:
        pass

df: pd.DataFrame | None = st.session_state.uploaded_df

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "🔍 Ticket Triage", "💬 Chat", "📋 Supervisor Insights", "🧠 Memory"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Customer Support Dashboard")

    if df is None:
        st.info("Upload a CSV file using the sidebar to populate the dashboard.")
    else:
        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tickets", f"{len(df):,}")

        if "Customer Satisfaction Rating" in df.columns:
            c2.metric("Avg Satisfaction", f"{df['Customer Satisfaction Rating'].mean():.2f} / 5")

        if "Ticket Priority" in df.columns:
            critical_n = int((df["Ticket Priority"].str.lower() == "critical").sum())
            c3.metric("Critical Tickets", f"{critical_n:,}")

        if "Ticket Status" in df.columns:
            open_n = int(df["Ticket Status"].str.lower().str.contains("open", na=False).sum())
            c4.metric("Open Tickets", f"{open_n:,}")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            if "Ticket Type" in df.columns:
                st.subheader("Ticket Type Distribution")
                st.bar_chart(df["Ticket Type"].value_counts())
        with col_b:
            if "Ticket Priority" in df.columns:
                st.subheader("Priority Distribution")
                st.bar_chart(df["Ticket Priority"].value_counts())

        if "Ticket Channel" in df.columns:
            st.subheader("Ticket Channel Distribution")
            st.bar_chart(df["Ticket Channel"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Ticket Triage
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Ticket Triage")

    mode = st.radio("Input mode", ["Select from CSV", "Paste ticket text"], horizontal=True)

    ticket_text = ""
    ticket_row  = {}

    if mode == "Select from CSV" and df is not None:
        ids = df["Ticket ID"].astype(str).tolist() if "Ticket ID" in df.columns else []
        sel = st.selectbox("Select Ticket ID", ids)
        if sel:
            row = df[df["Ticket ID"].astype(str) == sel].iloc[0]
            ticket_row  = row.to_dict()
            ticket_text = (
                str(row.get("Ticket Subject", "")) + " " +
                str(row.get("Ticket Description", ""))
            ).strip()
            st.text_area("Ticket text (preview)", ticket_text[:500], height=100, disabled=True)
    else:
        ticket_text = st.text_area(
            "Paste ticket text",
            height=120,
            placeholder="e.g. My order hasn't arrived and I want a refund...",
        )
        ticket_row = {
            "Ticket Subject": ticket_text,
            "Ticket Description": "",
            "Ticket Type": "General",
            "Ticket Priority": "Medium",
        }

    run_disabled = not ticket_text.strip()
    if st.button("🔍 Run Triage", use_container_width=True, disabled=run_disabled):
        wf = get_workflow()
        with st.spinner("Analysing ticket…"):
            result = wf.run_triage(ticket_row)

        st.divider()

        # Badges
        b1, b2, b3 = st.columns(3)
        b1.markdown(_badge("Sentiment", result.get("inferred_sentiment", "—"), _SENTIMENT))
        b2.markdown(_badge("Urgency",   result.get("inferred_urgency",   "—"), _URGENCY))
        b3.markdown(
            f"🏷️ **Intent:** {result.get('inferred_intent','—')}  •  "
            f"**Topic:** {result.get('inferred_topic','—')}"
        )

        # Refund eligibility
        eligible = result.get("refund_eligible")
        if eligible is True:
            st.success(f"✅ Refund Eligible — {result.get('refund_reason','')}")
        elif eligible is False:
            st.error(f"❌ Not Refund Eligible — {result.get('refund_reason','')}")
        else:
            st.info("ℹ️ Refund eligibility could not be determined (no purchase date).")

        if result.get("escalate"):
            st.warning("⚠️ ESCALATION REQUIRED — Critical urgency detected")

        # Policy reference
        with st.expander("📄 Relevant Policy Section"):
            st.markdown(result.get("policy_reference", "None found."))

        # Suggested reply (editable)
        st.subheader("✏️ Suggested Reply")
        st.text_area(
            "Edit before sending",
            value=result.get("suggested_reply", ""),
            height=220,
            key="reply_editor",
        )

        # Similar tickets
        similar = result.get("similar_tickets", [])
        if similar:
            with st.expander(f"🔗 Similar Past Tickets ({len(similar)})"):
                st.dataframe(pd.DataFrame(similar), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Semantic Search / Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Chat with Ticket Data")
    st.caption(
        "Ask anything about tickets, policies, or trends. "
        "Examples: *'Summarise refund complaints from last month'* · "
        "*'What are the top 3 customer pain points?'*"
    )

    # Render existing history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching…"):
                wf  = get_workflow()
                ans = wf.query(prompt)
            st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Supervisor Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Supervisor Insights")

    if df is None:
        st.info("Upload a CSV file to generate a supervisor report.")
    else:
        if st.button("📋 Generate Supervisor Report", use_container_width=True):
            wf = get_workflow()
            with st.spinner("Generating report…"):
                st.session_state.supervisor_report = wf.generate_supervisor_report(df)

        if st.session_state.supervisor_report:
            st.markdown(st.session_state.supervisor_report)
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                if "Ticket Type" in df.columns:
                    st.subheader("Issue Spike Table")
                    spike_df = (
                        df["Ticket Type"].value_counts()
                        .reset_index()
                        .rename(columns={"Ticket Type": "Type", "count": "Count"})
                    )
                    st.dataframe(spike_df, use_container_width=True)

            with col2:
                if "Customer Satisfaction Rating" in df.columns and "Ticket Type" in df.columns:
                    st.subheader("Avg Satisfaction by Type")
                    sat = df.groupby("Ticket Type")["Customer Satisfaction Rating"].mean().round(2)
                    st.bar_chart(sat)

            # Download
            report_payload = {
                "summary": st.session_state.supervisor_report,
                "ticket_type_counts": (
                    df["Ticket Type"].value_counts().to_dict()
                    if "Ticket Type" in df.columns else {}
                ),
                "avg_satisfaction": (
                    float(df["Customer Satisfaction Rating"].mean())
                    if "Customer Satisfaction Rating" in df.columns else None
                ),
                "total_tickets": len(df),
            }
            st.download_button(
                "⬇️ Download Report as JSON",
                data=json.dumps(report_payload, indent=2),
                file_name="supervisor_report.json",
                mime="application/json",
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Memory Viewer
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Agent Memory")
    st.caption(
        "Facts extracted by the MemoryManager from past conversations. "
        "These persist across sessions in `memory/triage_memory.db`."
    )

    import sqlite3 as _sqlite3
    import json as _json
    from datetime import datetime as _dt

    _MEMORY_DB_PATH = "memory/triage_memory.db"

    def _load_memories(keyword: str = "") -> list[dict]:
        """Read memories directly from SQLite — bypasses Agno's SQLAlchemy reflection bug."""
        if not os.path.exists(_MEMORY_DB_PATH):
            return []
        try:
            conn = _sqlite3.connect(_MEMORY_DB_PATH)
            cur  = conn.cursor()
            cur.execute(
                "SELECT memory_id, memory, topics, user_id, created_at "
                "FROM agno_memories ORDER BY created_at DESC"
            )
            raw_rows = cur.fetchall()
            conn.close()
        except Exception:
            return []

        rows = []
        for memory_id, memory, topics, user_id, created_at in raw_rows:
            text        = _json.loads(memory) if memory else ""
            topics_list = _json.loads(topics) if topics else []
            if keyword and keyword.lower() not in text.lower():
                continue
            created = (
                _dt.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                if created_at else "—"
            )
            rows.append({
                "Memory":    text,
                "Topics":    ", ".join(topics_list),
                "User ID":   user_id or "—",
                "Created":   created,
                "Memory ID": memory_id or "—",
            })
        return rows

    def _clear_memories() -> None:
        """Delete all rows from agno_memories via raw sqlite3."""
        conn = _sqlite3.connect(_MEMORY_DB_PATH)
        conn.execute("DELETE FROM agno_memories")
        conn.commit()
        conn.close()

    # ── Controls row ──────────────────────────────────────────────────────────
    col_search, col_clear = st.columns([4, 1])
    with col_search:
        keyword = st.text_input(
            "Search memories",
            placeholder="e.g. refund, escalation, billing…",
            label_visibility="collapsed",
        )
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            try:
                _clear_memories()
                st.success("All memories cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear memories: {e}")

    # ── Memory table ──────────────────────────────────────────────────────────
    rows = _load_memories(keyword)

    if not rows:
        st.info(
            "No memories stored yet. "
            "Ask questions in the **Chat** tab — the agent will extract and save facts automatically."
        )
    else:
        st.metric("Memories stored", len(rows))
        mem_df = pd.DataFrame(rows)

        # Show full memory text expanded, hide the ID column by default
        st.dataframe(
            mem_df[["Memory", "Topics", "User ID", "Created"]],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Show Memory IDs"):
            st.dataframe(
                mem_df[["Memory ID", "Memory"]],
                use_container_width=True,
                hide_index=True,
            )
