"""
AI Mock Interviewer — Streamlit Frontend
==========================================
Run from the workspace root:

    streamlit run frontend/app.py

LLM: Ollama (llama3.1) ← primary | Groq ← secondary
     Toggle in the sidebar.
"""

import sys
import os

# Ensure workspace root is on the path so `backend.*` imports work
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from backend.utils.llm_config import (
    LLMProvider,
    check_ollama_available,
    check_groq_available,
    OLLAMA_MODEL,
    GROQ_MODEL,
)
from backend.interview.question_bank import AVAILABLE_ROLES, total_questions
from backend.interview.controller import InterviewController

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — minimal chat polish
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .question-box {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: #e2e8f0;
        font-size: 1.05rem;
    }
    .feedback-box {
        background: #134e4a;
        border-left: 4px solid #10b981;
        padding: 0.85rem 1.1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        color: #d1fae5;
        font-size: 0.95rem;
    }
    .score-badge {
        display: inline-block;
        background: #4f46e5;
        color: white;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .answer-box {
        background: #1e3a5f;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        color: #bfdbfe;
        font-style: italic;
        margin: 0.4rem 0;
    }
    .final-report {
        background: #1a1a2e;
        border: 1px solid #6366f1;
        border-radius: 0.75rem;
        padding: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "interview_started": False,
        "controller": None,
        "selected_role": AVAILABLE_ROLES[0],
        "provider": LLMProvider.OLLAMA,
        "chat_history": [],       # list of {"type": "q"|"a"|"feedback", "content": str, "score": float|None}
        "current_answer": "",
        "processing": False,
        "show_final": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    # ── LLM Provider Toggle ──────────────────────────────────────────────────
    st.subheader("🤖 LLM Provider")

    ollama_ok = check_ollama_available()
    groq_ok = check_groq_available()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ollama", "🟢 Online" if ollama_ok else "🔴 Offline")
    with col2:
        st.metric("Groq", "🟢 Ready" if groq_ok else "🔴 No Key")

    provider_label = st.toggle(
        f"Use Groq ({GROQ_MODEL})",
        value=(st.session_state.provider == LLMProvider.GROQ),
        help=(
            "OFF → Ollama (llama3.1) [primary, local]\n"
            "ON  → Groq [secondary, cloud API]"
        ),
    )
    new_provider = LLMProvider.GROQ if provider_label else LLMProvider.OLLAMA

    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        # Hot-swap if interview is in progress
        if st.session_state.controller is not None:
            st.session_state.controller.switch_provider(new_provider)
            st.toast(f"Switched to {new_provider.value.upper()} mid-session.", icon="🔄")

    active_model = GROQ_MODEL if new_provider == LLMProvider.GROQ else OLLAMA_MODEL
    st.caption(f"Active model: **{active_model}**")

    # Warnings
    if new_provider == LLMProvider.OLLAMA and not ollama_ok:
        st.warning("Ollama is not running. Start it with: `ollama serve`", icon="⚠️")
    if new_provider == LLMProvider.GROQ and not groq_ok:
        st.error("GROQ_API_KEY missing in .env file.", icon="🔑")

    st.divider()

    # ── Role Selection ───────────────────────────────────────────────────────
    st.subheader("🎯 Interview Role")
    selected_role = st.selectbox(
        "Choose role",
        AVAILABLE_ROLES,
        index=AVAILABLE_ROLES.index(st.session_state.selected_role),
        disabled=st.session_state.interview_started,
    )
    st.session_state.selected_role = selected_role
    st.caption(f"{total_questions(selected_role)} questions for this role.")

    st.divider()

    # ── Controls ─────────────────────────────────────────────────────────────
    if not st.session_state.interview_started:
        if st.button("▶ Start Interview", use_container_width=True, type="primary"):
            # Validate pre-conditions
            if new_provider == LLMProvider.OLLAMA and not ollama_ok:
                st.error("Cannot start: Ollama is offline.")
            elif new_provider == LLMProvider.GROQ and not groq_ok:
                st.error("Cannot start: GROQ_API_KEY not set.")
            else:
                st.session_state.controller = InterviewController(
                    role=selected_role,
                    provider=new_provider,
                )
                st.session_state.interview_started = True
                st.session_state.chat_history = []
                st.session_state.show_final = False
                st.rerun()
    else:
        if st.button("🔄 Restart Interview", use_container_width=True):
            for key in ["interview_started", "controller", "chat_history",
                        "current_answer", "processing", "show_final"]:
                del st.session_state[key]
            st.rerun()

    st.divider()
    st.caption("AI Mock Interviewer · CrewAI + Streamlit")
    st.caption("LLM: Ollama (primary) | Groq (secondary)")

# ─────────────────────────────────────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧠 AI Mock Interviewer")

if not st.session_state.interview_started:
    # ── Welcome Screen ───────────────────────────────────────────────────────
    st.markdown(
        """
        Welcome to the **AI Mock Interviewer** — a role-specific interview simulator
        powered by **CrewAI** multi-agent architecture.

        ### How it works
        1. Select your **interview role** in the sidebar
        2. Choose your **LLM provider** (Ollama local or Groq cloud)
        3. Click **▶ Start Interview**
        4. Answer each question and receive instant AI feedback
        5. Get a **final performance report** at the end

        ### LLM Options
        | Provider | Model | Type |
        |---|---|---|
        | 🦙 Ollama | llama3.1 | Local (primary) |
        | ⚡ Groq | llama3-8b-8192 | Cloud API (secondary) |

        > No OpenAI API is used anywhere in this application.
        """
    )
    st.info("👈 Configure settings in the sidebar, then click **Start Interview**.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Active Interview UI
# ─────────────────────────────────────────────────────────────────────────────

ctrl: InterviewController = st.session_state.controller

# Progress bar
progress_val = ctrl.current_index / ctrl.total if not ctrl.is_complete else 1.0
st.progress(progress_val, text=f"Question {min(ctrl.current_index + 1, ctrl.total)} of {ctrl.total}  |  Role: **{ctrl.role}**")

st.divider()

# ── Chat History ─────────────────────────────────────────────────────────────
for item in st.session_state.chat_history:
    if item["type"] == "q":
        st.markdown(
            f'<div class="question-box">🎤 <strong>Question {item["num"]}:</strong> {item["content"]}</div>',
            unsafe_allow_html=True,
        )
    elif item["type"] == "a":
        st.markdown(
            f'<div class="answer-box">💬 {item["content"]}</div>',
            unsafe_allow_html=True,
        )
    elif item["type"] == "feedback":
        score_html = (
            f'<span class="score-badge">Score: {item["score"]}/10</span> &nbsp;'
            if item.get("score") is not None else ""
        )
        st.markdown(
            f'<div class="feedback-box">📋 {score_html}{item["content"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")  # spacing

# ── Current Question ─────────────────────────────────────────────────────────
if not ctrl.is_complete and not st.session_state.show_final:
    current_q = ctrl.get_current_question()

    if current_q:
        q_num = ctrl.current_index + 1
        st.markdown(
            f'<div class="question-box">🎤 <strong>Question {q_num}:</strong> {current_q["question"]}<br>'
            f'<small style="color:#94a3b8">Topic: {current_q.get("topic","General")}</small></div>',
            unsafe_allow_html=True,
        )

        # Answer input
        answer = st.text_area(
            "Your Answer",
            key=f"answer_{ctrl.current_index}",
            height=140,
            placeholder="Type your answer here... Be thorough and include examples where possible.",
            disabled=st.session_state.processing,
        )

        submit_col, skip_col = st.columns([4, 1])
        with submit_col:
            submit_clicked = st.button(
                "✅ Submit Answer",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing or not answer.strip(),
            )
        with skip_col:
            skip_clicked = st.button(
                "⏭ Skip",
                use_container_width=True,
                disabled=st.session_state.processing,
                help="Submit a blank answer and move on.",
            )

        if submit_clicked or skip_clicked:
            submitted_answer = answer.strip() if submit_clicked else "(Skipped)"
            st.session_state.processing = True

            # Add Q + A to history immediately for visual continuity
            st.session_state.chat_history.append({
                "type": "q",
                "num": q_num,
                "content": current_q["question"],
            })
            st.session_state.chat_history.append({
                "type": "a",
                "content": submitted_answer,
            })

            with st.spinner("🤔 Evaluating your answer..."):
                evaluation, feedback = ctrl.submit_answer(submitted_answer)

            st.session_state.chat_history.append({
                "type": "feedback",
                "content": feedback,
                "score": evaluation.get("score"),
            })
            st.session_state.processing = False

            if ctrl.is_complete:
                st.session_state.show_final = True

            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Final Report
# ─────────────────────────────────────────────────────────────────────────────

if ctrl.is_complete or st.session_state.show_final:
    report = ctrl.final_report
    summary = ctrl.final_summary

    st.divider()
    st.subheader("🏁 Interview Complete — Final Report")

    # Overall score tile
    overall = report.get("overall_score", 0)
    grade = report.get("grade", "N/A")

    score_color = (
        "#10b981" if overall >= 7 else
        "#f59e0b" if overall >= 4 else
        "#ef4444"
    )

    col_score, col_grade = st.columns([1, 3])
    with col_score:
        st.markdown(
            f"""
            <div style="text-align:center;background:#1e293b;border-radius:1rem;padding:1.5rem;">
                <div style="font-size:3rem;font-weight:800;color:{score_color};">{overall}</div>
                <div style="color:#94a3b8;font-size:0.85rem;">out of 10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_grade:
        st.markdown(f"### Grade: **{grade}**")
        st.markdown(summary)

    st.divider()

    # Per-question breakdown
    st.subheader("📊 Question-by-Question Breakdown")
    per_q = report.get("per_question", [])
    for i, item in enumerate(per_q, 1):
        q_score = item.get("score", 0) or 0
        bar_pct = int(q_score * 10)
        bar_color = "#10b981" if q_score >= 7 else "#f59e0b" if q_score >= 4 else "#ef4444"
        with st.expander(f"Q{i}: {item['question'][:80]}...  —  Score: {q_score}/10"):
            st.markdown(f"**Feedback:** {item.get('feedback','—')}")
            st.markdown(
                f'<div style="background:#334155;border-radius:999px;height:8px;margin-top:0.5rem;">'
                f'<div style="background:{bar_color};width:{bar_pct}%;height:8px;border-radius:999px;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Strengths & Improvements
    strengths = report.get("strengths", [])
    improvements = report.get("improvements", [])

    col_s, col_i = st.columns(2)
    with col_s:
        st.subheader("✅ Strengths")
        if strengths:
            for s in strengths:
                st.markdown(f"- {s}")
        else:
            st.markdown("_No clear strengths identified._")
    with col_i:
        st.subheader("⚠️ Areas to Improve")
        if improvements:
            for imp in improvements:
                st.markdown(f"- {imp}")
        else:
            st.markdown("_No improvement areas identified._")

    st.divider()
    st.success(
        "Interview session complete! Click **🔄 Restart Interview** in the sidebar to try again.",
        icon="🎉",
    )
