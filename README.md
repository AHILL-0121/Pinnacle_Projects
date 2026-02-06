<div align="center">

# 🏔️ Pinnacle Projects

### A Portfolio of Production-Grade AI & Financial Engineering Systems

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LLM](https://img.shields.io/badge/LLM-Multi--Provider-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Four interconnected projects spanning REST API design, OCR-powered document intelligence, retrieval-augmented generation, and autonomous AI agents — each built from scratch with enterprise-grade architecture.*

</div>

---

## 📋 Table of Contents

- [Repository Overview](#-repository-overview)
- [Architecture Map](#-architecture-map)
- [Project 1 — Flask Stock Intelligence API](#-project-1--flask-stock-intelligence-api)
- [Project 2 — Financial Document Analyzer](#-project-2--financial-document-analyzer)
- [Project 3 — RAG Systems Essentials](#-project-3--rag-systems-essentials)
- [Project 4 — AI Web Research Agent](#-project-4--ai-web-research-agent)
- [Shared Technical Concepts](#-shared-technical-concepts)
- [Global Prerequisites](#-global-prerequisites)
- [Environment Variables Reference](#-environment-variables-reference)
- [Troubleshooting](#-troubleshooting)
- [License & Author](#-license--author)

---

## 🎯 Repository Overview

This monorepo contains **four full-stack, independently deployable projects** organized by learning complexity:

| Level | Project | Domain | Core Technologies |
|:-----:|---------|--------|-------------------|
| **L2** | [Flask Stock Intelligence API](#-project-1--flask-stock-intelligence-api) | Financial Data & Analysis | Flask, yfinance, NumPy, Pandas |
| **L2** | [Financial Document Analyzer](#-project-2--financial-document-analyzer) | Document AI & OCR | FastAPI, Tesseract, Streamlit, Multi-LLM |
| **L2** | [RAG Systems Essentials](#-project-3--rag-systems-essentials) | Research Paper Q&A | FAISS, Sentence Transformers, Multi-LLM |
| **L3** | [AI Web Research Agent](#-project-4--ai-web-research-agent) | Autonomous Research | ReAct Pattern, Tavily, Multi-LLM |

### What Makes These Production-Grade

- **Separation of concerns** — Every project follows layered architecture (routes → services → utils)
- **Multi-LLM support** — Gemini, Groq, Ollama, and OpenAI/OpenRouter interchangeably across all AI projects
- **Input validation** — Comprehensive sanitization at every entry point
- **Structured error handling** — JSON error responses with appropriate HTTP status codes
- **Hallucination prevention** — Confidence gating, deterministic validation, constrained generation
- **Stateless design** — No server-side sessions; horizontally scalable

---

## 🗺️ Architecture Map

```
Pinnacle_Projects/
│
├── L2/                                    # Intermediate-level projects
│   │
│   ├── Coding Essentials for Agent/
│   │   └── flask-stock-api/               # PROJECT 1: REST API for stock intelligence
│   │       ├── run.py                     #   Entry point (port 5000)
│   │       ├── config.py                  #   Environment-based configuration
│   │       └── app/
│   │           ├── __init__.py            #   Flask application factory
│   │           ├── routes/                #   Blueprint-based endpoint handlers
│   │           │   ├── company.py         #     GET /api/company/<symbol>
│   │           │   ├── stock.py           #     GET /api/stock/<symbol>
│   │           │   ├── history.py         #     POST /api/history
│   │           │   └── analysis.py        #     POST /api/analyze
│   │           ├── services/              #   Business logic layer
│   │           │   ├── yahoo_service.py   #     Yahoo Finance data fetching
│   │           │   └── analysis_service.py#     Quantitative computations
│   │           └── utils/                 #   Validation & error handling
│   │               ├── errors.py          #     Custom exception classes
│   │               └── validators.py      #     Input sanitization
│   │
│   ├── Prompt Engineering Essentials/
│   │   └── financial-document-analyzer/   # PROJECT 2: OCR + LLM document AI
│   │       ├── Jupiter NB/               #   Jupyter Notebook prototype
│   │       │   └── Financial_Report_Analysis.ipynb
│   │       └── OCR-Tessaract/            #   Production system
│   │           ├── backend/              #     FastAPI backend (port 8000)
│   │           │   ├── run.py            #       Entry point
│   │           │   └── app/
│   │           │       ├── main.py       #       FastAPI routes + CORS
│   │           │       ├── config.py     #       Pydantic settings (env vars)
│   │           │       ├── models/       #       Pydantic schemas
│   │           │       └── services/     #       Core processing services
│   │           │           ├── ocr_service.py        # Tesseract/PaddleOCR extraction
│   │           │           ├── llm_service.py        # Multi-provider LLM abstraction
│   │           │           ├── entity_extractor.py   # Deterministic financial NER
│   │           │           ├── chart_analyzer.py     # Vision API chart interpretation
│   │           │           ├── summarizer.py         # Role-aware report generation
│   │           │           └── table_extractor.py    # Table structure extraction
│   │           ├── frontend/             #     Streamlit UI (port 8501)
│   │           │   └── app.py            #       Upload, configure, visualize
│   │           └── docs/                 #     Sample outputs
│   │
│   └── RAG Systems Essentials/           # PROJECT 3: Research paper Q&A
│       ├── main.py                       #   Entry point
│       ├── cli.py                        #   Interactive CLI commands
│       ├── config.py                     #   Dataclass-based configuration
│       ├── test_edge_cases.py            #   5 edge-case regression tests
│       ├── src/
│       │   ├── document_processor.py     #   PDF extraction + semantic chunking
│       │   ├── embeddings.py             #   MiniLM-L6-v2 sentence embeddings
│       │   ├── vector_store.py           #   FAISS IndexFlatIP (cosine similarity)
│       │   ├── retriever.py              #   MMR + section-aware dual retrieval
│       │   ├── llm_providers.py          #   Gemini/Groq/Ollama provider manager
│       │   └── rag_pipeline.py           #   Orchestration with confidence gating
│       └── data/
│           ├── papers/                   #   Place PDF research papers here
│           └── index/                    #   Persisted FAISS index + chunk metadata
│
└── L3/                                   # Advanced-level project
    └── Building AI Agents from Scratch/  # PROJECT 4: Autonomous research agent
        ├── agent.py                      #   Complete single-file agent (841 lines)
        ├── requirements.txt
        └── reports/                      #   Auto-generated research reports (MD + HTML)
```

---

## 📈 Project 1 — Flask Stock Intelligence API

### Purpose

A **stateless, backend-only REST API** that provides real-time stock market intelligence — company metadata, live quotes, historical OHLCV data, and quantitative analysis — for integration with AI agents, trading bots, and dashboards.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Framework** | Flask 3.0+ with Application Factory pattern |
| **Data Source** | Yahoo Finance via `yfinance` 0.2.36+ |
| **Numerical Engine** | NumPy (statistical computations), Pandas (time-series operations) |
| **Production Server** | Gunicorn 21.0+ |
| **Port** | `5000` (development) |
| **Python Version** | 3.10+ |

### API Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `GET` | `/health` | Health check | — |
| `GET` | `/api/company/<symbol>` | Company metadata (name, sector, officers, market cap) | Path: `symbol` |
| `GET` | `/api/stock/<symbol>` | Real-time quote (price, change, volume, 52-week range, market state) | Path: `symbol` |
| `POST` | `/api/history` | Historical OHLCV data | Body: `symbol`, `start_date`, `end_date`, `interval` |
| `POST` | `/api/analyze` | Quantitative analysis (volatility, trend, drawdown, insights) | Body: `symbol`, `start_date`, `end_date`, `interval` |

### Data Flow

```
Client Request
    │
    ▼
Route Blueprint ──── Input Validation (validators.py)
    │                    │
    │              ◄─────┘ (rejects invalid symbol/date/interval)
    ▼
Service Layer
    ├── YahooFinanceService: get_company_info(), get_stock_data(), get_historical_data()
    └── AnalysisService: analyze_stock()
            ├── _calculate_volatility()     → Annualized std. dev. of daily returns (√252)
            ├── _detect_trend()             → Linear regression slope + start-end % comparison
            ├── _calculate_max_drawdown()   → Peak-to-trough via running maximum
            ├── _calculate_return()         → Total return percentage
            └── _generate_insight()         → Rule-based natural language summary
    │
    ▼
JSON Response (with structured error handling via custom exceptions)
```

### Analysis Algorithms

**Trend Detection** uses a dual-signal approach:
1. *Start-end percentage comparison* — classifies >5% as bullish/bearish
2. *Linear regression slope* via `numpy.polyfit(x, prices, 1)` — confirms direction

Combined classification: `bullish | mildly_bullish | sideways | mildly_bearish | bearish`

**Volatility** is computed as annualized standard deviation:

$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252} \times 100$$

**Maximum Drawdown** uses the running-maximum method:

$$MDD = \min\left(\frac{P_t - \max_{s \leq t}(P_s)}{\max_{s \leq t}(P_s)}\right) \times 100$$

### Configuration

Three environment profiles via `config.py`:

| Profile | `DEBUG` | `TESTING` | Use Case |
|---------|---------|-----------|----------|
| `DevelopmentConfig` | `True` | `False` | Local development |
| `ProductionConfig` | `False` | `False` | Production deployment |
| `TestingConfig` | `True` | `True` | Automated tests |

Key settings: `YAHOO_TIMEOUT=10s`, `VALID_INTERVALS=['1d','1wk','1mo']`, `MAX_DATE_RANGE_DAYS=3650`, `MAX_HISTORY_RECORDS=5000`

### Quick Start

```powershell
cd "L2/Coding Essentials for Agent/flask-stock-api"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
# Server at http://localhost:5000
```

### Example Requests

```bash
# Company info
curl http://localhost:5000/api/company/AAPL

# Real-time quote
curl http://localhost:5000/api/stock/TSLA

# Historical data
curl -X POST http://localhost:5000/api/history \
  -H "Content-Type: application/json" \
  -d '{"symbol":"MSFT","start_date":"2024-01-01","end_date":"2024-12-31","interval":"1d"}'

# Quantitative analysis
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"GOOGL","start_date":"2024-01-01","end_date":"2024-06-30"}'
```

---

## 📊 Project 2 — Financial Document Analyzer

### Purpose

An **AI-powered document intelligence system** that converts scanned financial documents (PDFs, images) into structured, role-aware financial summaries with hallucination-resistant architecture. Features a **FastAPI backend** and **Streamlit frontend**.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Backend Framework** | FastAPI 0.109 + Uvicorn ASGI server |
| **Frontend Framework** | Streamlit 1.31 |
| **OCR Engine** | Tesseract (primary) / PaddleOCR (optional) |
| **PDF Processing** | PyMuPDF (fitz) + pdfplumber |
| **Image Processing** | Pillow 10.2 + NumPy |
| **LLM Providers** | Gemini, Groq, Ollama, OpenAI (switchable per request) |
| **Vision Model** | Gemini Vision (chart/graph interpretation) |
| **Validation** | Pydantic 2.5 (schemas) + Pydantic-Settings (env config) |
| **Ports** | Backend: `8000`, Frontend: `8501` |

### Architecture — Hallucination-Resistant Pipeline

The system's key innovation is separating **deterministic data extraction** from **LLM reasoning**:

```
Document Upload (PDF/Image/JSON)
         │
         ▼
┌──────────────────────────────────┐
│  Stage 1: Image Preprocessing    │  Pillow: enhancement, DPI normalization (300 DPI)
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Stage 2: Text Extraction        │  pdfplumber (tables) + Tesseract OCR (text)
│                                  │  Returns: OCRBlock[] with confidence & bounding boxes
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Stage 3: Chart Analysis         │  Gemini Vision API detects chart type, trends,
│  (Optional)                      │  key values from visual graphs
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Stage 4: Deterministic NER      │  entity_extractor.py: 100+ regex patterns for
│  (CODE decides, NOT LLM)         │  revenue, profit, EPS, ratios, currency, period
│                                  │  detection (Q1-Q4, FY, YTD), YoY/QoQ changes
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Stage 5: Period-Keyed JSON      │  Structured source-of-truth with latest_period,
│  (Source of Truth)               │  earliest_period, all extracted metrics
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Stage 6: LLM Report Generation  │  Single LLM call with pre-templated values
│  (Role-Aware)                    │  Role: Investor | Analyst | Auditor | Executive
└──────────────────────────────────┘
         │
         ▼
  Structured Summary + Confidence Score (JSON/Markdown export)
```

### Core Services

| Service | File | Responsibility |
|---------|------|----------------|
| **OCRService** | `ocr_service.py` | Tesseract/PaddleOCR text extraction with block-level confidence scoring, bounding box detection, auto-detects Tesseract on Windows |
| **LLMService** | `llm_service.py` | Abstract base class with `GeminiProvider`, `GroqProvider`, `OllamaProvider` implementations; generates `LLMResponse` with token accounting |
| **FinancialEntityExtractor** | `entity_extractor.py` | 100+ compiled regex patterns for currency (8 symbols), periods (Q/FY/H/YTD/TTM), metrics (revenue, profit, EPS, ratios), change detection (YoY/QoQ/MoM), value normalization (mn/bn/cr/lakh) |
| **ChartAnalyzer** | `chart_analyzer.py` | Vision-LLM chart interpretation: detects `BAR|LINE|PIE|AREA|STACKED_BAR|COMBO|WATERFALL` chart types, extracts `UP|DOWN|STABLE|VOLATILE` trends |
| **TableExtractor** | `table_extractor.py` | Structural table detection from OCR blocks |
| **FinancialSummarizer** | `summarizer.py` | Orchestrates all services, assembles role-aware prompts |

### Role-Aware Summarization

The LLM adapts its output based on the selected user role:

| Role | Focus Areas | Key Metrics | Tone |
|------|-------------|-------------|------|
| **Investor** | Growth potential, profitability, risk | Revenue Growth, NPM, EPS, ROE, P/E | Decision-oriented |
| **Analyst** | Ratios, trends, valuation | ROE, ROA, D/E, Current Ratio, Op. Margin | Technical, data-driven |
| **Auditor** | Compliance, anomalies, red flags | Asset Quality, CAR, CET1, NPL | Scrutinizing |
| **Executive** | Strategy, competitive position | Revenue, EBITDA, Market Share | High-level, strategic |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check with service status |
| `GET` | `/health` | Detailed health check (LLM + Vision availability) |
| `GET` | `/providers` | List available LLM providers and their status |
| `POST` | `/analyze` | Upload document + role + provider → structured analysis |

### Quick Start

```powershell
# Terminal 1: Backend
cd "L2/Prompt Engineering Essentials/financial-document-analyzer/OCR-Tessaract/backend"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
# Backend at http://localhost:8000 | Docs at http://localhost:8000/docs

# Terminal 2: Frontend
cd "L2/Prompt Engineering Essentials/financial-document-analyzer/OCR-Tessaract/frontend"
pip install -r requirements.txt
streamlit run app.py --server.port 8501
# Frontend at http://localhost:8501
```

### System Requirements

- **Tesseract OCR** must be installed separately:
  - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`
- At least **one LLM provider** configured (see [Environment Variables](#-environment-variables-reference))

---

## 🔍 Project 3 — RAG Systems Essentials

### Purpose

A **portfolio-grade Retrieval-Augmented Generation system** for question-answering over AI research papers, featuring confidence gating, hallucination prevention, section-aware retrieval, and multi-LLM support. Achieves **5/5 (Grade A)** on edge-case tests.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| **Vector Store** | FAISS `IndexFlatIP` (cosine similarity via normalized inner product) |
| **Retrieval Strategy** | Dual-strategy: MMR for diversity + section-aware boosting |
| **LLM Providers** | Gemini, Groq, Ollama (**no OpenAI dependency**) |
| **PDF Processing** | PyMuPDF (primary) + pdfminer.six (backup) |
| **Python Version** | 3.9+ |

### Architecture Pipeline

```
User Question
     │
     ├─── Query Type Detection
     │    (factual | table | section | cross-paper)
     │
     ├─── Query Encoding (MiniLM-L6-v2, 384-dim)
     │
     ├─── FAISS Retrieval (top_k=6)
     │    ├── Maximal Marginal Relevance (λ=0.3)
     │    └── Section-Aware Score Boosting:
     │         abstract=1.2x, introduction=1.1x,
     │         architecture=1.3x, method=1.25x
     │
     ├─── Confidence Gate
     │    ├── General queries:  threshold = 0.50
     │    ├── Factual queries:  threshold = 0.75
     │    └── Below threshold → "This information is not present..."
     │
     ├─── LLM Generation (query-type-specific prompts)
     │    ├── DEFAULT_SYSTEM_PROMPT      → conceptual questions
     │    ├── FACTUAL_SYSTEM_PROMPT      → exact values/numbers
     │    ├── CROSS_PAPER_SYSTEM_PROMPT  → multi-paper synthesis
     │    └── SECTION_QUERY_SYSTEM_PROMPT → structural questions
     │
     └─── Answer + Source Citations (max 2)
```

### Core Components

| Component | File | Technical Detail |
|-----------|------|------------------|
| **DocumentProcessor** | `document_processor.py` | PDF extraction with semantic chunking (`chunk_size=400 tokens`, `overlap=75 tokens`, `min=100 tokens`) |
| **EmbeddingModel** | `embeddings.py` | Sentence Transformers wrapper with batch encoding (`batch_size=32`) |
| **FAISSVectorStore** | `vector_store.py` | FAISS `IndexFlatIP` with L2 normalization for cosine similarity, persistence to disk (JSON metadata + `.index` file) |
| **Retriever** | `retriever.py` | Query-type detection via keyword matching (`FACTUAL_KEYWORDS`, `TABLE_KEYWORDS`, `SECTION_KEYWORDS`), MMR diversity, section-aware score boosting |
| **RAGPipeline** | `rag_pipeline.py` | Full orchestration with 4 specialized system prompts, confidence-gated generation, structured `RAGResponse` output with timing metrics |
| **LLMManager** | `llm_providers.py` | Multi-provider abstraction for Gemini, Groq, Ollama |

### Hallucination Prevention (5 Layers)

| Layer | Mechanism | Implementation |
|-------|-----------|----------------|
| 1 | **Confidence Gating** | Refuses when retrieval confidence < threshold (0.50 general / 0.75 factual) |
| 2 | **Strict System Prompts** | "Answer ONLY from context" instruction in all 4 prompt templates |
| 3 | **Citation Discipline** | Max 2 citations, filtered to query-relevant papers only |
| 4 | **Table Query Detection** | Extra-strict 0.75 threshold for numeric/table data requests |
| 5 | **Cross-Paper Isolation** | Prevents data leakage between separate papers |

### Configuration Reference (`config.py`)

```python
# Chunking
chunk_size = 400          # tokens per chunk
chunk_overlap = 75        # overlap tokens
min_chunk_size = 100      # minimum tokens

# Embedding
model_name = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384

# Retrieval
top_k = 6                 # chunks to retrieve
similarity_threshold = 0.3
use_mmr = True
mmr_diversity = 0.3       # λ (0=max diversity, 1=max relevance)

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.50
FACTUAL_CONFIDENCE_THRESHOLD = 0.75

# Token Limits by Query Type
cross_paper_max_tokens = 256
factual_max_tokens = 150
general_max_tokens = 1024
```

### Edge Case Test Results

| Test | What It Validates | Result |
|------|-------------------|--------|
| Table Exactness | Refuses to hallucinate missing table data | ✅ PASS |
| Negative Refusal | Clean refusal with proper citations | ✅ PASS |
| Cross-Paper Reasoning | Concise synthesis without extra metrics | ✅ PASS |
| Section Precision | Identifies Section 3.2 for multi-head attention | ✅ PASS |
| Knowledge Boundaries | No data leakage across papers | ✅ PASS |

### Quick Start

```powershell
cd "L2/RAG Systems Essentials"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Place PDFs in data/papers/
# Then run:
python main.py --provider ollama
# Or single question:
python main.py -q "What is multi-head attention?"
# Run tests:
python test_edge_cases.py
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/ingest` | Ingest PDFs from `data/papers/` into vector index |
| `/stats` | Show index statistics (chunk count, papers indexed) |
| `/provider <name>` | Switch LLM provider at runtime |
| `/help` | Show available commands |
| `/quit` | Exit |

---

## 🔬 Project 4 — AI Web Research Agent

### Purpose

An **autonomous research agent** implementing the **ReAct (Reason + Act) pattern** that automates end-to-end web research: generating research questions, searching the web, synthesizing findings, and producing formatted reports in both Markdown and HTML.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Design Pattern** | ReAct (Reason + Act) — alternating LLM reasoning and tool use |
| **Web Search** | Tavily API (`search_depth="advanced"`, 3 results/question) |
| **LLM Providers** | Gemini, Groq, Ollama, OpenRouter (4 providers) |
| **CLI Framework** | Rich (spinners, panels, progress bars, colored output) |
| **Output Formats** | Markdown (`.md`) + styled HTML (`.html`, auto-opens in browser) |
| **Architecture** | Single-file agent (`agent.py`, 841 lines) with clean class hierarchy |

### ReAct Execution Flow

```
User Topic: "AI in Healthcare"
│
├── PHASE 1: PLANNING (Reason)
│   LLM generates 4 targeted research questions
│   JSON output: ["Q1?", "Q2?", "Q3?", "Q4?"]
│   Fallback: Template-based question generation on JSON parse failure
│
├── PHASE 2: ACTING (Act)
│   For each question → Tavily Web Search (advanced depth)
│   Returns: SearchResult(title, url, content, score) × 3 per question
│   Total: ~12 web results gathered
│
├── PHASE 3: SYNTHESIS (Reason)
│   For each question → LLM summarizes search results
│   Input truncated to 300 chars/result for token efficiency
│   Output: Markdown bullet-point summaries (150-200 words each)
│
├── PHASE 4: FRAMING (Reason)
│   LLM generates introduction (50-80 words) and conclusion (50-80 words)
│
└── PHASE 5: REPORT GENERATION
    ├── Markdown report with TOC, 4-6 sections, source citations
    ├── HTML conversion with styled template (CSS custom properties)
    └── Auto-opens HTML in default browser
```

### Class Hierarchy

```
LLMProvider (ABC)               # Abstract base for all LLM providers
├── GeminiProvider              #   google.genai SDK, exponential backoff (3 retries)
├── GroqProvider                #   groq SDK, chat completions
├── OllamaProvider              #   ollama SDK, local inference
└── OpenRouterProvider          #   HTTP REST API, supports 50+ models

WebSearchTool                   # Tavily API wrapper
├── search()                    #   Returns List[SearchResult]

ResearchAgent                   # Main orchestrator
├── plan()                      #   REASON: Generate research questions (JSON)
├── act()                       #   ACT: Execute web searches
├── reason()                    #   REASON: Synthesize single question's results
├── synthesize_all()            #   REASON: Batch synthesis with progress bars
├── generate_introduction()     #   REASON: Create report intro
├── generate_conclusion()       #   REASON: Create report conclusion
├── generate_report()           #   Assemble final Markdown
├── _generate_filename()        #   LLM-generated meaningful filename
├── _save_and_open_html()       #   Markdown→HTML conversion + browser open
└── research()                  #   Full workflow orchestration
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Content truncation (300 chars) | Optimizes for free-tier API token limits |
| 4 questions (configurable) | Balances coverage vs. API costs |
| Exponential backoff (Gemini) | Handles rate limiting (429/RESOURCE_EXHAUSTED) gracefully |
| Separate intro/conclusion LLM calls | Keeps each prompt focused and token-efficient |
| `rich` CLI library | Professional terminal UX with spinners, panels, progress indicators |

### Quick Start

```powershell
cd "L3/Building AI Agents from Scratch"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Create .env with at least TAVILY_API_KEY + one LLM provider key
# Then:
python agent.py "Artificial Intelligence in Healthcare"

# Custom output:
python agent.py "Climate Change" -o climate_report.md

# Override provider:
python agent.py "Quantum Computing" --provider groq
```

---

## 🔗 Shared Technical Concepts

### Multi-LLM Provider Pattern

All three AI projects (Projects 2, 3, 4) implement the same provider abstraction:

```
┌──────────────────────────────────────┐
│        Abstract LLM Interface         │
│  generate(prompt, system_prompt) → str│
└──────────────────────────────────────┘
         ▲         ▲         ▲         ▲
         │         │         │         │
    ┌────┴───┐ ┌───┴───┐ ┌──┴────┐ ┌──┴────────┐
    │ Gemini │ │  Groq │ │Ollama │ │ OpenRouter │
    │ (Cloud)│ │(Cloud)│ │(Local)│ │  (Cloud)   │
    └────────┘ └───────┘ └───────┘ └────────────┘
```

| Provider | Type | Speed | Cost | Best For |
|----------|------|-------|------|----------|
| **Ollama** | Local | Medium | Free | Development, privacy |
| **Groq** | Cloud | Fastest | Free tier | Speed-critical tasks |
| **Gemini** | Cloud | Fast | Free tier | Good balance |
| **OpenAI** | Cloud | Fast | Paid | Best quality |
| **OpenRouter** | Cloud | Varies | Pay-per-use | Model variety |

### Shared Design Patterns

| Pattern | Used In | Implementation |
|---------|---------|----------------|
| Application Factory | Project 1 | `create_app()` in Flask `__init__.py` |
| Blueprint/Router | Projects 1, 2 | Flask Blueprints / FastAPI routers |
| Service Layer | All | Business logic separated from routes |
| Abstract Base Class | Projects 2, 3, 4 | `BaseLLMProvider(ABC)` / `LLMProvider(ABC)` |
| Factory Function | Project 4 | `get_llm_provider()` returns configured instance |
| Dataclass Models | Projects 3, 4 | `@dataclass` for structured data (RAGResponse, SearchResult) |
| Pydantic Settings | Project 2 | Environment variable configuration with validation |

---

## ⚙️ Global Prerequisites

| Requirement | Version | Required By | Notes |
|-------------|---------|-------------|-------|
| **Python** | 3.10+ | All projects | 3.9+ for RAG only |
| **pip** | Latest | All projects | Package manager |
| **Tesseract OCR** | Latest | Project 2 | System-level install |
| **Internet** | — | All (except Ollama) | API access |
| **Git** | Latest | — | Optional, for cloning |

### Recommended: Virtual Environments

Each project should use its own virtual environment to avoid dependency conflicts:

```powershell
# Create and activate per-project
cd "L2/Coding Essentials for Agent/flask-stock-api"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🔐 Environment Variables Reference

Create a `.env` file in each project that uses LLMs:

```env
# ═══════════════════════════════════════════════════
#  LLM PROVIDER SELECTION
# ═══════════════════════════════════════════════════
LLM_PROVIDER=gemini               # gemini | groq | ollama | openrouter

# ═══════════════════════════════════════════════════
#  GOOGLE GEMINI
#  Get key: https://makersuite.google.com/app/apikey
# ═══════════════════════════════════════════════════
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.0-flash     # or gemini-1.5-pro

# ═══════════════════════════════════════════════════
#  GROQ (Fast Inference)
#  Get key: https://console.groq.com/keys
# ═══════════════════════════════════════════════════
GROQ_API_KEY=gsk_your-key
GROQ_MODEL=llama-3.3-70b-versatile

# ═══════════════════════════════════════════════════
#  OLLAMA (Local, Free)
#  Install: https://ollama.ai → ollama pull llama3.2
# ═══════════════════════════════════════════════════
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ═══════════════════════════════════════════════════
#  OPENROUTER (Multi-Model, Project 4 only)
#  Get key: https://openrouter.ai/keys
# ═══════════════════════════════════════════════════
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=google/gemma-2-9b-it:free

# ═══════════════════════════════════════════════════
#  OPENAI (Project 2 only)
#  Get key: https://platform.openai.com/api-keys
# ═══════════════════════════════════════════════════
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o

# ═══════════════════════════════════════════════════
#  TAVILY WEB SEARCH (Project 4 only)
#  Get key: https://tavily.com (1000 searches/month free)
# ═══════════════════════════════════════════════════
TAVILY_API_KEY=tvly-your-key

# ═══════════════════════════════════════════════════
#  VISION MODEL (Project 2 — Chart Analysis)
# ═══════════════════════════════════════════════════
VISION_PROVIDER=gemini
VISION_MODEL=gemini-1.5-flash
```

---

## 🔧 Troubleshooting

| Problem | Project | Solution |
|---------|---------|----------|
| `ModuleNotFoundError` | Any | Ensure virtual environment is activated and `pip install -r requirements.txt` ran |
| `TesseractNotFoundError` | 2 | Install Tesseract system binary; set `TESSERACT_CMD` in `.env` if non-standard path |
| `RESOURCE_EXHAUSTED` / 429 | 2, 3, 4 | Rate limited — wait 60s, switch to Ollama (`--provider ollama`), or use Groq |
| `yfinance returns None` | 1 | Check internet connection; symbol may be delisted or invalid |
| Low confidence / refusals | 3 | Expected behavior — confidence gating rejects uncertain answers. Add more relevant PDFs to `data/papers/` |
| `TAVILY_API_KEY` error | 4 | Sign up at [tavily.com](https://tavily.com) for free key (1000/month) |
| No charts detected | 2 | Requires `GEMINI_API_KEY` for Vision model; text analysis still works without it |
| FAISS import error | 3 | Install with `pip install faiss-cpu` (not `faiss`) |
| Ollama not responding | 2, 3, 4 | Run `ollama serve` in a separate terminal; pull model with `ollama pull llama3.2` |

---

## 📄 License & Author

**License:** MIT — Use freely for educational and portfolio purposes.

**Author:** AHILL S

---

<div align="center">

*Built with dedication to production-grade software engineering, clean architecture, and responsible AI design.*

</div>