# Conversational AI – Competitor Intelligence System for Clothing Retail

> An AI-powered decision-support assistant for clothing retailers to discover nearby competitors, estimate footfall trends, and generate actionable market intelligence reports — all through natural language conversation.

Built with **LangGraph** + **LangChain** + **Ollama (LLaMA 3.1)** / **OpenAI**, using free **OpenStreetMap** data (no paid API keys required).

---

## Features

| Feature | Description |
|---------|-------------|
| **Conversational Interface** | Ask questions in plain English about local clothing competition |
| **Competitor Discovery** | Location-based search via OpenStreetMap (Nominatim + Overpass API) |
| **Footfall Estimation** | Busy-hour analysis using distance-based heuristics and time-of-day patterns |
| **Report Generation** | Structured Markdown reports with competitor lists, footfall comparison, peak hours, and strategic takeaways |
| **Multi-Turn Context** | Remembers previous searches — ask follow-ups without repeating the location |
| **Dual Interface** | Rich CLI (terminal) + Streamlit web chat UI |
| **No Paid APIs** | Uses entirely free and open-source data sources |

---

## Architecture

```
User Query
  → LangGraph ReAct Agent (LLM with tool-calling)
    → competitor_fetch_tool     → find nearby clothing stores (OSM)
    → footfall_estimator_tool   → estimate busy hours & footfall levels
    → report_formatter_tool     → generate full Markdown analysis report
  → Formatted Response (Markdown tables, not raw JSON)
```

### Key Design Decisions

- **Tools return pre-formatted Markdown** — the LLM relays results directly instead of trying to format JSON (works reliably with small local models like LLaMA 3.1 8B).
- **Shared in-memory store** (`tools/_store.py`) — tools pass data to each other without JSON serialization through the LLM.
- **Post-processing guard** — if the LLM describes tool output instead of relaying it, the agent injects the actual tool output automatically.

### Agent Graph

```
┌──────────┐     tool calls     ┌──────────┐
│  agent   │ ──────────────►    │  tools   │
│  (LLM)   │ ◄──────────────    │ (4 tools)│
└──────────┘     results        └──────────┘
     │
     │ no more tool calls
     ▼
   [END] → formatted response
```

---

## Project Structure

```
├── main.py                     # CLI entry point (interactive / single-query / demo)
├── app.py                      # Streamlit web chat UI
├── config.py                   # Environment & path configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
│
├── agent/
│   ├── graph.py                # LangGraph agent definition + post-processing
│   ├── state.py                # Agent state schema
│   └── prompts.py              # System prompt (tool-use rules)
│
├── tools/
│   ├── _store.py               # Shared in-memory data store
│   ├── location_search.py      # Geocoding tool (Nominatim)
│   ├── competitor_fetch.py     # Nearby competitor search (Overpass API)
│   ├── footfall_estimator.py   # Busy-hour estimation (heuristic model)
│   └── report_formatter.py     # Full Markdown report generator
│
├── services/
│   ├── llm_service.py          # LLM provider factory (Ollama / OpenAI)
│   ├── places_service.py       # Nominatim geocoding + Overpass POI search
│   └── cache.py                # Disk-based response caching
│
├── models/
│   └── schemas.py              # Pydantic v2 data models
│
└── data/
    ├── demo/                   # Sample competitor data (for demo mode)
    ├── cache/                  # API response cache (auto-created, gitignored)
    └── reports/                # Generated reports (auto-created, gitignored)
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM) — or an OpenAI API key

### 1. Install Dependencies

```bash
cd "L3/Building your First AI Agent with LangGraph"
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
LLM_PROVIDER=ollama          # or "openai"
OLLAMA_MODEL=llama3.1        # local model
DEFAULT_RADIUS_KM=2          # search radius in km
DEMO_MODE=false              # set true for offline demo
```

### 3. Start Ollama (if using local LLM)

```bash
ollama pull llama3.1
ollama serve
```

### 4. Launch

**Interactive Chat (CLI):**
```bash
python main.py
```

**Single Query:**
```bash
python main.py --query "List clothing competitors near my area"
```

**Demo Mode:**
```bash
python main.py --demo
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
```

---

## Example Queries

| Query | What It Does |
|-------|-------------|
| *"List clothing stores near \<your area\>"* | Discovers nearby stores with distance and type |
| *"What are the peak hours for these stores?"* | Estimates busy-hour patterns and footfall levels |
| *"Generate a full competitor report"* | Creates a structured Markdown report with all sections |
| *"Now check \<another area\>"* | Searches a new location (remembers context) |
| *"Compare it with the previous area"* | Cross-location competitive comparison |

### Multi-Turn Conversation Flow

```
You > List clothing stores near my locality
      → [table: 25 stores with distance, type, address]

You > What are the peak hours?
      → [footfall table: High/Medium/Low per store, busiest days]

You > Generate a full competitor report
      → [5-section Markdown report saved to data/reports/]

You > Now check another nearby area
      → [new search, new table]

You > Compare both areas
      → [comparative analysis]
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Orchestration | LangGraph + LangChain |
| LLM | Ollama (LLaMA 3.1) / OpenAI (GPT-4o-mini) |
| Location Data | OpenStreetMap (Nominatim + Overpass API) — free, no key needed |
| Data Models | Pydantic v2 |
| Caching | diskcache |
| CLI | Rich (Markdown rendering, spinners, tables) |
| Web UI | Streamlit |
| Reports | Markdown (auto-saved to `data/reports/`) |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OPENAI_API_KEY` | — | OpenAI API key (if using openai provider) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `DEFAULT_LOCATION` | — | Default search area |
| `DEFAULT_RADIUS_KM` | `2` | Search radius in kilometers |
| `DEFAULT_BUSINESS_TYPE` | `clothing_store` | OSM business type filter |
| `DEMO_MODE` | `false` | Use synthetic demo data (no network needed) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CACHE_TTL_SECONDS` | `3600` | Cache expiry in seconds |

---

## How It Works

1. **User asks a question** → the ReAct agent decides which tool to call.
2. **`competitor_fetch_tool`** geocodes the location via Nominatim, then queries Overpass API for nearby clothing/fashion stores. Returns a formatted Markdown table and stores raw data in the shared `_store`.
3. **`footfall_estimator_tool`** reads competitor data from the store, generates distance-based synthetic popularity scores, and produces footfall estimates with peak hours. Returns formatted Markdown.
4. **`report_formatter_tool`** combines competitor + footfall data into a 5-section report (Area Overview, Competitor List, Footfall Comparison, Peak Hours, Strategic Takeaways). Saves to `data/reports/`.
5. **Post-processing** in the agent graph ensures tool output is always relayed to the user, even if the LLM tries to describe it instead.

---

## Data Disclaimer

All footfall and busy-hour data are **estimates** derived from publicly available proxy indicators (geographic distance, time-based heuristics). They do not represent actual measured foot traffic. The system uses only free, open-source data from OpenStreetMap.

---

## License

This project is part of the Pinnacle Projects collection and is intended for educational and portfolio purposes.
