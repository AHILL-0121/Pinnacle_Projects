# CrewAI Logistics Optimization System

A production-grade, multi-agent system that **proactively analyzes logistics data** and generates actionable optimization strategies using autonomous AI agents powered by [CrewAI](https://github.com/crewAIInc/crewAI) and a local Ollama LLM.

> **Key Design Principle:** All numerical metrics are **pre-computed in Python** and passed to the LLM as read-only facts. The LLM interprets and structures — it never calculates. This eliminates arithmetic hallucination entirely.

---

## Problem Statement

| Pain Point | Impact |
|---|---|
| Inefficient delivery routes | High fuel costs, slow deliveries |
| Poor inventory turnover | Overstocking waste, stockout risk |
| Manual, reactive decision-making | Losses discovered after the fact |

This system flips the script — it **proactively** identifies inefficiencies and proposes strategies *before* losses occur.

---

## Architecture

```
Input JSON (Products + Routes + Inventory)
        │
        ▼
┌─────────────────────────────────┐
│  Deterministic Metrics Engine   │  ← Python pre-computes speed, cost/km,
│  (tasks/definitions.py)        │    excess stock, waste, classifications
└────────────┬────────────────────┘
             │ facts (read-only)
             ▼
┌─────────────────────────────────┐
│  Logistics Analyst  (Agent 1)  │  → Interprets metrics, flags ALL
│                                │    inefficiencies exhaustively
└────────────┬────────────────────┘
             │ structured analysis
             ▼
┌─────────────────────────────────┐
│  Optimization Strategist       │  → Prioritized action plan with
│  (Agent 2)                     │    bounded savings estimates
└────────────┬────────────────────┘
             │
             ▼
       Markdown Report (auditable)
```

**Process:** Sequential — Agent 1 feeds context to Agent 2.

### Why This Architecture?

- **Safe** — LLM cannot hallucinate numbers; every figure is pre-computed
- **Explainable** — prompts contain guardrails; output is auditable
- **Extendable** — add agents/tasks without refactoring core logic

---

## Project Structure

```
├── main.py                  # CLI entry point (argparse)
├── crew.py                  # Crew assembly (sequential process)
├── config.py                # LLM & runtime config (env-var overridable)
├── requirements.txt         # Dependencies (crewai only)
├── .gitignore
├── agents/
│   ├── __init__.py
│   └── definitions.py       # Logistics Analyst & Optimization Strategist
├── tasks/
│   ├── __init__.py
│   └── definitions.py       # Pre-compute engines + task prompts
├── models/
│   ├── __init__.py
│   └── schemas.py           # Product, Route, Inventory dataclasses
├── data/
│   └── sample_logistics.json # Demo dataset (5 products, 6 routes, 5 inventory)
└── output/                  # Auto-generated reports (git-ignored)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | CrewAI ≥ 0.86.0 |
| Language | Python 3.10+ |
| LLM | Ollama — LLaMA 3.1 (local, no API key needed) |
| LLM Routing | Ollama's OpenAI-compatible endpoint (`/v1`) |
| Data Format | JSON |

---

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed with `llama3.1` pulled:

```bash
# Install Ollama → https://ollama.com/download
ollama pull llama3.1
ollama serve                # must be running on port 11434
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
# Default run — sample data, verbose output
python main.py

# Custom data + custom output path
python main.py --data path/to/your_data.json --output output/custom.md

# Quiet mode (suppress agent reasoning)
python main.py --quiet
```

Reports are saved to `output/` with a timestamp by default.

---

## Input Data Format

Supply a JSON file with three arrays — `products`, `routes`, and `inventory`:

```json
{
  "products": [
    { "product_id": "P001", "name": "Widget A", "category": "Hardware" }
  ],
  "routes": [
    {
      "route_id": "R001",
      "origin": "Warehouse A",
      "destination": "DC 1",
      "distance_km": 382,
      "delivery_time_hr": 5.5,
      "product_ids": ["P001"],
      "fuel_cost_usd": 145.00
    }
  ],
  "inventory": [
    {
      "product_id": "P001",
      "warehouse": "Warehouse A",
      "stock_level": 4500,
      "turnover_rate": 1.2,
      "reorder_point": 500,
      "holding_cost_per_unit_usd": 0.71
    }
  ]
}
```

> **Note:** `holding_cost_per_unit_usd` is the cost to hold **one unit** for one month.

---

## Configuration

All settings are environment-variable overridable (see `config.py`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_TEMPERATURE` | `0.2` | Lower → more deterministic |
| `CREW_VERBOSE` | `true` | Show agent reasoning in terminal |
| `MAX_EXECUTION_TIMEOUT` | `120` | Max seconds for full pipeline |

---

## Agents

| Agent | Role | What It Does |
|---|---|---|
| **Logistics Analyst** | Identify inefficiencies | Exhaustively flags every route violation (speed < 80 km/hr, cost/km > $0.45), overlapping routes, and inventory classifications (overstocked / stockout / borderline) |
| **Optimization Strategist** | Propose strategies | Prioritized action plans for route consolidation, re-routing, inventory right-sizing; bounded savings estimates; quick wins |

---

## Pre-Computed Metrics (Hallucination Prevention)

The system pre-computes all numerical values in Python before the LLM sees them:

| Metric | Computation |
|---|---|
| Route speed | `distance_km / delivery_time_hr` |
| Cost per km | `fuel_cost_usd / distance_km` |
| Route overlaps | Same origin→destination served by multiple route IDs |
| Excess stock | `stock_level - (turnover_rate × reorder_point)` |
| Monthly waste | `max(excess, 0) × holding_cost_per_unit_usd` |
| Classification | Turnover < 1.0 → Overstocked · > 6.0 → Stockout Risk · = 6.0 → Borderline |

The LLM prompt includes the directive:

> *"⚠️ CRITICAL: Do NOT recalculate any numbers. Use ONLY the pre-computed values."*

---

## Sample Output Highlights

Using the included demo dataset (`data/sample_logistics.json`):

| Metric | Value |
|---|---|
| Routes flagged (speed < 80) | R001, R002, R003, R006 |
| Cost-inefficient route | R002 ($0.49/km) |
| Overlapping routes | R001 + R006 (Chicago → Detroit) |
| Most overstocked product | P003 — 7,400 excess units |
| Monthly holding waste | $10,301 |
| Annual holding waste | $123,612 |
| Strategist savings estimate | ≤ $123,612 (bounded) |

---

## Extending the System

The modular design makes it easy to add:

- **New agents** — e.g., Cost Modeler, Emissions Analyst
- **New tasks** — e.g., carbon footprint, demand forecasting
- **New data sources** — swap `LogisticsData.from_json_file()` with a DB or API loader
- **Different LLMs** — change `OLLAMA_MODEL` or point to OpenAI/Anthropic via CrewAI's LLM abstraction

---

## License

MIT
