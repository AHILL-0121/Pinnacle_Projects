"""
Streamlit Web UI for the Competitor Intelligence System.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import config
from agent import get_agent, SYSTEM_PROMPT

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CompetitorIQ – Clothing Retail Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🏪 CompetitorIQ")
    st.caption("AI-Powered Competitor Intelligence for Clothing Retail")

    st.divider()

    st.subheader("⚙️ Settings")
    demo_mode = st.toggle("Demo Mode", value=config.DEMO_MODE,
                          help="Use synthetic data (no API keys needed)")
    if demo_mode != config.DEMO_MODE:
        config.DEMO_MODE = demo_mode

    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "openai"],
        index=0 if config.LLM_PROVIDER == "ollama" else 1,
    )
    if llm_provider != config.LLM_PROVIDER:
        config.LLM_PROVIDER = llm_provider
        if "agent" in st.session_state:
            del st.session_state["agent"]

    st.divider()

    st.subheader("💡 Example Queries")
    examples = [
        "List top clothing competitors near Koramangala, Bangalore",
        "What are the busiest hours for competitors near Indiranagar?",
        "Generate a competitor analysis report for HSR Layout",
        "Compare footfall trends of clothing stores near Jayanagar",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["pending_query"] = ex

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["lc_messages"] = []
        st.rerun()

    st.divider()
    st.caption(f"Mode: {'🟢 Demo' if config.DEMO_MODE else '🔵 Live'} | LLM: {config.LLM_PROVIDER}")

# ── Session State Init ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []          # For display (role + content)
    st.session_state["lc_messages"] = []       # LangChain message objects

if "agent" not in st.session_state:
    with st.spinner("Building agent graph..."):
        st.session_state["agent"] = get_agent()


# ── Chat Display ────────────────────────────────────────────
st.title("🏪 Competitor Intelligence Agent")
st.caption("Ask me about clothing store competitors, footfall trends, and market analysis.")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle Input ────────────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask about competitors...") or pending

if user_input:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build LangChain messages
    st.session_state["lc_messages"].append(HumanMessage(content=user_input))

    # Invoke agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                agent = st.session_state["agent"]
                result = agent.invoke(
                    {"messages": st.session_state["lc_messages"]},
                    config={"recursion_limit": 25},
                )
                # Update LangChain messages from agent output
                st.session_state["lc_messages"] = list(result["messages"])
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"⚠️ Error: {e}\n\nPlease check that your LLM provider ({config.LLM_PROVIDER}) is running."

        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
