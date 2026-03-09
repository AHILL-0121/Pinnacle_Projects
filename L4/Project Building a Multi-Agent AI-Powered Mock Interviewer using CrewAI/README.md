# 🧠 AI Mock Interviewer

A **role-specific AI mock interview platform** with a multi-agent architecture
and a **Streamlit** chat-style UI.

> **No OpenAI API is used anywhere.**
> LLMs are called directly: **Ollama llama3.1** (local, primary) and **Groq** (cloud, secondary).
> Switch between them with a toggle in the sidebar — hot-swappable mid-session.

---

## Features

| Feature | Status |
|---|---|
| Role selection (Data Scientist, Web Dev, PM) | ✅ |
| Sequential questioning — one question at a time | ✅ |
| Per-answer AI evaluation (Score, Correctness, Depth, Example, Clarity) | ✅ |
| Instant coaching feedback after every answer | ✅ |
| Final performance report with grade + breakdown | ✅ |
| Ollama ↔ Groq toggle with live availability status | ✅ |
| Hot-swap LLM provider mid-session | ✅ |
| Chat-style Streamlit UI with progress bar | ✅ |

---

## Architecture

```
Streamlit UI  (frontend/app.py)
      │
      ▼
InterviewController  (backend/interview/controller.py)
      │
      ├──► EvaluationAgent ──► call_llm() ──► Ollama HTTP  /  Groq SDK
      │
      └──► FeedbackAgent   ──► call_llm() ──► Ollama HTTP  /  Groq SDK
```

**Agents** (direct LLM calls — no framework overhead):

| Agent | Responsibility |
|---|---|
| `EvaluationAgent` | Scores each answer: 1–10, Correctness, Depth, Example, Clarity |
| `FeedbackAgent` | Converts evaluation into 2–4 sentence coaching feedback + final summary |
| `QuestionAgent` | Rephrases questions and generates follow-ups (optional, on demand) |

**LLM layer** (`backend/utils/llm_config.py`):

| Provider | Transport | Model |
|---|---|---|
| Ollama (primary) | Direct HTTP POST to `/api/chat` | `llama3.1` |
| Groq (secondary) | Official `groq` Python SDK | `llama3-8b-8192` |

---

## LLM Setup

### Ollama — Primary (Local, Free, No API Key)

1. Install: <https://ollama.com/download>
2. Pull the model:
   ```bash
   ollama pull llama3.1
   ```
3. Ollama auto-starts on login; or start manually:
   ```bash
   ollama serve
   ```

### Groq — Secondary (Cloud, Free Tier)

1. Get a free API key: <https://console.groq.com/>
2. Add to your `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```

---

## Quick Start

```powershell
# 1. Navigate to the project
cd "C:\Users\Asus\Desktop\AI INTERVIEWER"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
Copy-Item .env.example .env
# Open .env and set GROQ_API_KEY if you want to use Groq

# 5. Run the app
streamlit run frontend/app.py
```

Then open <http://localhost:8501> in your browser.

---

## Project Structure

```
AI INTERVIEWER/
│
├── frontend/
│   └── app.py                    ← Streamlit UI (sidebar toggle, chat, report)
│
├── backend/
│   ├── agents/
│   │   ├── evaluation_agent.py   ← Scores answers (direct LLM call)
│   │   ├── feedback_agent.py     ← Coaching feedback + final summary (direct LLM call)
│   │   └── question_agent.py     ← Question rephrasing / follow-ups (direct LLM call)
│   │
│   ├── interview/
│   │   ├── controller.py         ← Session orchestrator (sequencing, state, finalisation)
│   │   └── question_bank.py      ← Static question bank: Data Scientist, Web Dev, PM
│   │
│   └── utils/
│       ├── llm_config.py         ← Ollama (HTTP) + Groq (SDK) — no OpenAI
│       └── scoring.py            ← Structured output parser + final report builder
│
├── requirements.txt
├── .env.example                  ← Copy to .env and fill in GROQ_API_KEY
├── plan.md
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model to use |
| `GROQ_API_KEY` | _(required for Groq)_ | Free key from console.groq.com |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model to use |

---

## Adding a New Role

1. Open `backend/interview/question_bank.py`
2. Add an entry to `QUESTION_BANK`:
   ```python
   "My New Role": [
       {
           "id": 1,
           "question": "Your question here?",
           "topic": "Topic Name",
           "keywords": ["keyword1", "keyword2"],
       },
       # 4–5 more questions...
   ]
   ```
3. The role appears automatically in the Streamlit sidebar — no other changes needed.

---

## Evaluation Criteria

Every answer is scored across four dimensions:

| Criterion | What it checks |
|---|---|
| **Correctness** | Is the concept accurate? |
| **Depth** | Is the explanation thorough? |
| **Example** | Does the answer include a real-world example? |
| **Clarity** | Is the reasoning logically structured? |

A numeric score (1–10) is derived from these and aggregated into a final grade:
`Excellent` (9–10) · `Good` (7–8) · `Average` (5–6) · `Needs Improvement` (3–4) · `Poor` (1–2)

---

## Future Extensions

- Resume-based dynamic question generation
- Voice input / TTS output
- PDF report export
- Difficulty levels (Junior / Mid / Senior)
- Company-specific question packs
- Analytics dashboard across sessions
