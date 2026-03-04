<div align="center">

# 🏔️ Pinnacle Projects

### A Portfolio of Production-Grade AI & Financial Engineering Systems

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6F00?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF5722?style=for-the-badge)](https://github.com/crewAIInc/crewAI)
[![n8n](https://img.shields.io/badge/n8n-Workflow-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io)
[![LLM](https://img.shields.io/badge/LLM-Multi--Provider-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Eighteen interconnected projects spanning REST API design, OCR-powered document intelligence, retrieval-augmented generation, autonomous AI agents, multi-agent systems with AutoGen, graph-based workflows with LangGraph, no-code automation, machine learning classification and regression, deep learning, NLP with transformers, LLM fine-tuning, and custom tokenizer development — each built with enterprise-grade architecture.*

</div>

---

## 📋 Table of Contents

- [Repository Overview](#-repository-overview)
- [Architecture Map](#-architecture-map)
- [Project 1 — Flask Stock Intelligence API](#-project-1--flask-stock-intelligence-api)
- [Project 2 — Financial Document Analyzer](#-project-2--financial-document-analyzer)
- [Project 3 — RAG Systems Essentials](#-project-3--rag-systems-essentials)
- [Project 4 — AI Web Research Agent](#-project-4--ai-web-research-agent)
- [Project 5 — Intelligent Travel Assistant](#-project-5--intelligent-travel-assistant)
- [Project 6 — AI Content Creator Agent (n8n)](#-project-6--ai-content-creator-agent-n8n)
- [Project 7 — Competitor Intelligence System (LangGraph)](#-project-7--competitor-intelligence-system-langgraph)
- [Project 8 — Logistics Optimization System (CrewAI)](#-project-8--logistics-optimization-system-crewai)
- [Project 9 — Health Classification ML Model](#-project-9--health-classification-ml-model)
- [Project 10 — NYC Taxi Trip Duration](#-project-10--nyc-taxi-trip-duration)
- [Project 11 — Water Quality Prediction (Deep Learning)](#-project-11--water-quality-prediction-deep-learning)
- [Project 12 — Comparative Sentiment Analysis (NLP)](#-project-12--comparative-sentiment-analysis-nlp)
- [Project 13 — Parameter-Efficient Fine-Tuning with QLoRA](#-project-13--parameter-efficient-fine-tuning-with-qlora)
- [Project 14 — Building Custom Tokenizers from Scratch](#-project-14--building-custom-tokenizers-from-scratch)
- [Project 15 — Advanced AI Agents with AutoGen](#-project-15--advanced-ai-agents-with-autogen)
- [Project 16 — LangGraph Multi-Agent Research System](#-project-16--langgraph-multi-agent-research-system)
- [Shared Technical Concepts](#-shared-technical-concepts)
- [Global Prerequisites](#-global-prerequisites)
- [Environment Variables Reference](#-environment-variables-reference)
- [Troubleshooting](#-troubleshooting)
- [License & Author](#-license--author)

---

## 🎯 Repository Overview

This monorepo contains **eighteen production-grade projects** (including 4 AutoGen sub-projects) organized by learning complexity:

| Level | Project | Domain | Core Technologies |
|:-----:|---------|--------|-------------------|
| **L2** | [Flask Stock Intelligence API](#-project-1--flask-stock-intelligence-api) | Financial Data & Analysis | Flask, yfinance, NumPy, Pandas |
| **L2** | [Financial Document Analyzer](#-project-2--financial-document-analyzer) | Document AI & OCR | FastAPI, Tesseract, Streamlit, Multi-LLM |
| **L2** | [RAG Systems Essentials](#-project-3--rag-systems-essentials) | Research Paper Q&A | FAISS, Sentence Transformers, Multi-LLM |
| **L3** | [AI Web Research Agent](#-project-4--ai-web-research-agent) | Autonomous Research | ReAct Pattern, Tavily, Multi-LLM |
| **L3** | [Intelligent Travel Assistant](#-project-5--intelligent-travel-assistant) | Travel AI Agent | LangChain Agent, WeatherAPI, DuckDuckGo |
| **L3** | [AI Content Creator Agent](#-project-6--ai-content-creator-agent-n8n) | No-Code Automation | n8n, Ollama, Tavily, Google Sheets |
| **L3** | [Competitor Intelligence System](#-project-7--competitor-intelligence-system-langgraph) | Retail Intelligence | LangGraph, LangChain, OpenStreetMap, Streamlit |
| **L3** | [Logistics Optimization System](#-project-8--logistics-optimization-system-crewai) | Supply Chain Optimization | CrewAI, Ollama, Deterministic Metrics Engine |
| **L4** | [Health Classification ML Model](#-project-9--health-classification-ml-model) | Insurance Risk Prediction | scikit-learn, XGBoost, Pandas, Seaborn |
| **L4** | [NYC Taxi Trip Duration](#-project-10--nyc-taxi-trip-duration) | Geospatial Regression | scikit-learn, Pandas, NumPy, Matplotlib |
| **L4** | [Water Quality Prediction (Deep Learning)](#-project-11--water-quality-prediction-deep-learning) | Environmental AI | MLP Neural Networks, scikit-learn, Pandas, Seaborn |
| **L4** | [Comparative Sentiment Analysis (NLP)](#-project-12--comparative-sentiment-analysis-nlp) | NLP & Deep Learning | PyTorch, Transformers (BERT), RNN/LSTM/GRU, HuggingFace |
| **L4** | [Parameter-Efficient Fine-Tuning with QLoRA](#-project-13--parameter-efficient-fine-tuning-with-qlora) | LLM Fine-Tuning | QLoRA, PEFT, BitsAndBytes, Transformers, W&B |
| **L4** | [Building Custom Tokenizers from Scratch](#-project-14--building-custom-tokenizers-from-scratch) | Tokenization & NLP | BPE, HuggingFace Tokenizers, WikiText-2, Subword Processing |
| **L4** | [Advanced AI Agents with AutoGen](#-project-15--advanced-ai-agents-with-autogen) | Multi-Agent Systems | AutoGen, Groq, Vision API, FSM, Reflection Pattern |
| **L4** | [LangGraph Multi-Agent Research System](#-project-16--langgraph-multi-agent-research-system) | Graph-Based Agents | LangGraph, ChromaDB, Tavily, RAG, Intelligent Routing |

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
└── L3/                                   # Advanced-level projects
    ├── Building AI Agents from Scratch/  # PROJECT 4: Autonomous research agent
    │   ├── agent.py                      #   Complete single-file agent (841 lines)
    │   ├── requirements.txt
    │   └── reports/                      #   Auto-generated research reports (MD + HTML)
    │
    ├── Building AI Agents with LangChain/# PROJECT 5: LangChain Travel Assistant
    │   ├── main.py                       #   CLI entry point (interactive loop)
    │   ├── agent.py                      #   Agent factory (multi-LLM + tool binding)
    │   ├── config.py                     #   Central configuration (loads .env)
    │   ├── requirements.txt
    │   └── tools/                        #   Modular tool registry
    │       ├── weather.py                #     @tool – WeatherAPI.com
    │       └── attractions.py            #     @tool – DuckDuckGo search
    │
    ├── Automate Anything with n8n/       # PROJECT 6: No-code content creator
    │   ├── AI Content Creator Agent.json #   n8n workflow definition
    │   └── readme.md                     #   Full documentation
    │
    ├── Building your First AI Agent with LangGraph/  # PROJECT 7: Competitor intelligence
    │   ├── main.py                       #   CLI entry point (interactive / demo)
    │   ├── app.py                        #   Streamlit web chat UI
    │   ├── config.py                     #   Environment configuration
    │   ├── requirements.txt
    │   ├── agent/
    │   │   ├── graph.py                  #   LangGraph ReAct agent + post-processing
    │   │   ├── state.py                  #   Agent state schema
    │   │   └── prompts.py                #   System prompt (tool-use rules)
    │   ├── tools/
    │   │   ├── _store.py                 #   Shared in-memory data store
    │   │   ├── location_search.py        #   Geocoding (Nominatim)
    │   │   ├── competitor_fetch.py        #   Nearby competitor search (Overpass API)
    │   │   ├── footfall_estimator.py      #   Busy-hour estimation
    │   │   └── report_formatter.py        #   Markdown report generator
    │   ├── services/
    │   │   ├── llm_service.py            #   LLM provider factory (Ollama / OpenAI)
    │   │   ├── places_service.py          #   Nominatim + Overpass API
    │   │   └── cache.py                  #   Disk-based response caching
    │   ├── models/
    │   │   └── schemas.py                #   Pydantic v2 data models
    │   └── data/
    │       ├── demo/                     #   Sample competitor data
    │       ├── cache/                    #   API response cache
    │       └── reports/                  #   Generated reports
    │
    └── Building your First AI Agent with CrewAI/     # PROJECT 8: Logistics optimization
        ├── main.py                       #   CLI entry point (argparse)
        ├── crew.py                       #   Crew assembly (sequential process)
        ├── config.py                     #   LLM & runtime config (env-var overridable)
        ├── requirements.txt
        ├── agents/
        │   ├── __init__.py
        │   └── definitions.py            #   Logistics Analyst & Optimization Strategist
        ├── tasks/
        │   ├── __init__.py
        │   └── definitions.py            #   Pre-compute engines + task prompts
        ├── models/
        │   ├── __init__.py
        │   └── schemas.py                #   Product, Route, Inventory dataclasses
        ├── data/
        │   └── sample_logistics.json      #   Demo dataset (5 products, 6 routes)
        └── output/                       #   Auto-generated reports (git-ignored)
│
└── L4/                                   # Machine Learning projects
    ├── Building your First ML Model/      # PROJECT 9: Health classification pipeline
    │   ├── README.md
    │   └── anova_insurance_health_classification(1).ipynb
    │       ├── Section 1–2: Imports & EDA #   numpy, pandas, sklearn, seaborn
    │       ├── Section 3: Preprocessing   #   Imputation → scaling → stratified split
    │       ├── Section 4: Model Training  #   6 models, 5-fold StratifiedKFold CV
    │       ├── Section 5: Evaluation      #   Confusion matrix, ROC curves, AUC
    │       └── Section 6: Feature Importance # RF + GB averaged importances
    │
    ├── Foundational ML Algorithms/        # PROJECT 10: NYC Taxi trip duration
    │   ├── README.md
    │   └── nyc_taxi_trip_duration(1).ipynb  # Regression: LR / RF / GBR
    │
    ├── Introduction to Deep Learning using PyTorch/  # PROJECT 11: Water quality prediction
    │   ├── README.md
    │   ├── water_quality.csv               #   CPCB dataset (19,029 records)
    │   └── water_quality_prediction.ipynb  #   Dual MLP: WQI regression + classification
    │       ├── Section 1–2: Imports & Load #     pandas, numpy, sklearn, seaborn
    │       ├── Section 3: Preprocessing    #     Cleaning → encode → split → scale
    │       ├── Section 4: EDA              #     Heatmap, WQI distributions, correlations
    │       ├── Section 5: Regression MLP   #     15→512→256→128→64→1 (Adam, early stop)
    │       ├── Section 6: Classification MLP #   15→512→256→128→64→5 (Softmax)
    │       └── Section 7: Final Summary    #     Side-by-side performance table
    │
    └── Natural Language Processing using PyTorch/    # PROJECT 12: Comparative NLP
        ├── README.md
        ├── requirements.txt
        └── Natural Language Processing using PyTorch/
            └── sentiment_analysis_comparative.ipynb  # BERT vs LSTM vs GRU vs RNN
                ├── Sections 1–2: Imports & Config  #   PyTorch, Transformers, device check
                ├── Section 3: Load & Preprocess    #   Sentiment140, clean, balance, split
                ├── Section 4: Vocabulary           #   Top-10K word vocab (RNN-family)
                ├── Section 5: DataLoaders          #   TextDataset + BertDataset
                ├── Section 6: Model Definitions    #   RNN / LSTM / GRU / BertSentiment
                ├── Section 7: Train/Eval Helpers   #   train_rnn, eval_bert, compute_metrics
                ├── Sections 8–9: Model Training    #   All 4 models (5 epochs each)
                ├── Section 10: Metrics Table       #   7-column comparative summary
                ├── Section 11: Visualizations      #   5 saved charts → outputs/
                ├── Section 12: Classification Rpts #   Per-class P/R/F1 per model
                └── Section 13: Final Summary       #   Recommendations by use case
    │
    ├── Finetuning LLMs/                   # PROJECT 13: Parameter-efficient fine-tuning
    │   ├── README.md
    │   └── qlora_bert_local_gpu.ipynb      #   QLoRA (4-bit quantization + LoRA adapters)
    │       ├── Section 1: Dependencies     #     torch, transformers, peft, bitsandbytes
    │       ├── Section 2: GPU Check        #     CUDA availability, VRAM detection
    │       ├── Section 3: Configuration    #     Hyperparameters, LoRA config, batch sizes  
    │       ├── Section 4: W&B Setup        #     Experiment tracking integration
    │       ├── Section 5: Data Prep        #     IMDB sentiment tokenization & mapping
    │       ├── Section 6: Model Setup      #     4-bit quantized BERT + LoRA adapters
    │       ├── Section 7: Training         #     Parameter-efficient fine-tuning (~0.16% params)
    │       └── Section 8: Evaluation       #     Accuracy, F1, memory usage analysis
    │
    └── Training LLMs from Scratch/         # PROJECT 14: Custom tokenizer creation  
        ├── README.md
        └── bpe_tokenizer_wikitext2.ipynb   #   BPE tokenizer from WikiText-2 corpus
            ├── Section 1: Dependencies     #     datasets, tokenizers, transformers
            ├── Section 2: Dataset Loading  #     WikiText-2 exploration & statistics
            ├── Section 3: Data Cleaning    #     Remove duplicates, normalize text
            ├── Section 4: Tokenizer Training #   BPE algorithm, 30K vocab, special tokens
            ├── Section 5: Analysis         #     Vocabulary coverage, compression ratio
            ├── Section 6: Evaluation       #     Sample tokenization, subword quality
            └── Section 7: Export           #     Save tokenizer for downstream usage
    │
    ├── Building Advanced AI Agents with AutoGen/  # PROJECT 15: Multi-agent systems collection
    │   ├── README.md                       #   Collection overview & patterns guide
    │   ├── Bill Managing Agent/            #   Group Chat pattern
    │   │   ├── README.md
    │   │   └── bill_management_agent_autogen.ipynb
    │   │       ├── Section 1: Setup        #     autogen, groq, pillow installations
    │   │       ├── Section 2: LLM Config   #     Groq API (vision + text models)
    │   │       ├── Section 3: Agents       #     User Proxy, Bill Processor, Summarizer
    │   │       ├── Section 4: Group Chat   #     GroupChatManager orchestration
    │   │       └── Section 5: Execution    #     Bill image processing workflow
    │   │
    │   ├── Financial Portfolio Manager/    #   Stateful FSM pattern
    │   │   ├── README.md
    │   │   └── financial_portfolio_manager.ipynb
    │   │       ├── Section 1: Setup        #     autogen, groq installation
    │   │       ├── Section 2: State Machine #    INIT → ANALYSIS → RECOMMEND → REPORT
    │   │       ├── Section 3: Agents       #     Portfolio Analyst, Advisor, Reporter
    │   │       ├── Section 4: User Profile #     Risk tolerance, goals, holdings
    │   │       └── Section 5: Execution    #     State-driven workflow progression
    │   │
    │   ├── Smart Content Creation/         #   Reflection pattern
    │   │   ├── README.md
    │   │   └── agentic_ai_autogen_reflection.ipynb
    │   │       ├── Section 1: Setup        #     autogen installation
    │   │       ├── Section 2: Agents       #     Content Creator, Content Critic
    │   │       ├── Section 3: Reflection   #     Bidirectional dialogue loop
    │   │       ├── Section 4: Scoring      #     Quality threshold (8.5/10)
    │   │       └── Section 5: Execution    #     Iterative content improvement
    │   │
    │   └── Smart Health Assistant/         #   Sequential with function calling
    │       ├── README.md
    │       └── smart_health_assistant.ipynb
    │           ├── Section 1: Setup        #     autogen installation
    │           ├── Section 2: Function Tools #   calculate_bmi() registration
    │           ├── Section 3: Agents       #     User Proxy, BMI Agent, Meal Planner
    │           ├── Section 4: Workflow     #     Sequential handoffs
    │           └── Section 5: Execution    #     Health assessment pipeline
    │
    └── Building Advanced AI Agents with LangGraph/ # PROJECT 16: Graph-based coordination
        ├── README.md
        └── langgraph_multi_agent_groq.ipynb    #   Router → LLM/RAG/Web → Summarizer
            ├── Section 1: Dependencies     #     langgraph, langchain, chroma, tavily
            ├── Section 2: Configuration    #     Groq, HuggingFace embeddings, Tavily
            ├── Section 3: State Definition #     AgentState schema with typed dict
            ├── Section 4: LLM & Tools Init #     Groq LLM, Tavily search, embeddings
            ├── Section 5: RAG Setup        #     ChromaDB knowledge base (AI/ML docs)
            ├── Section 6: Agent Nodes      #     Router, LLM, RAG, Web Research, Summarizer
            ├── Section 7: Graph Build      #     Conditional routing, edge definitions
            ├── Section 8: Execution        #     Query examples (RAG, Web, LLM)
            └── Section 9: Memory           #     MemorySaver for conversation persistence
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

## 🌍 Project 5 — Intelligent Travel Assistant

### Purpose

An **AI-powered travel assistant** built with **LangChain's tool-calling agent architecture** that accepts a destination city and autonomously fetches real-time weather data and top tourist attractions, then synthesises them into a unified travel briefing.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Framework** | LangChain ≥ 1.2 (`create_tool_calling_agent` + `AgentExecutor`) |
| **LLM Providers** | OpenAI GPT-4o-mini, Google Gemini, Ollama (llama3.1) — switchable via env var |
| **Weather API** | WeatherAPI.com (free tier) |
| **Search Engine** | DuckDuckGo via `ddgs` package (free, no key) |
| **Architecture** | Modular tool-based agent — add tools without touching agent logic |
| **Python Version** | 3.10+ |

### How the LLM Reasoning Works

The LLM acts as an **autonomous planner**, not hard-coded logic:

```
User: "Paris"
     │
     ▼
LLM reads system prompt + wrapped query
     │  "I'm planning a trip to Paris..."
     ▼
LLM reasons: need weather AND attractions
     │
     ▼
LLM generates tool calls:
  ├── get_weather("Paris")    → WeatherAPI.com → temperature, humidity, wind
  └── get_attractions("Paris") → DuckDuckGo    → top 5 tourist spots
     │
     ▼
AgentExecutor runs both tools, adds results to scratchpad
     │
     ▼
LLM sees all tool outputs, synthesises single travel briefing
     │
     ▼
Final formatted response displayed to user
```

The LLM **dynamically decides** which tools to call based on the query. Asking "What's the weather in Tokyo?" calls only the weather tool. Asking "What should I see in Rome?" calls only the attractions tool. This is **tool-based reasoning** — the intelligence comes from the LLM, not from `if/else` branches.

### Program Flow

```
1.  python main.py
2.  config.py loads .env (API keys, LLM_PROVIDER)
3.  agent.py builds: LLM → create_tool_calling_agent → AgentExecutor
4.  User types city name → wrapped into natural language prompt
5.  AgentExecutor runs reasoning chain:
      a. LLM selects tools  b. Tools execute  c. LLM merges results
6.  Response printed → loop continues until "quit"
```

### Project Structure

```
L3/Building AI Agents with LangChain/
├── .env.example        # API key template
├── config.py           # Central config (loads .env)
├── agent.py            # Agent factory (multi-LLM + tool binding)
├── main.py             # CLI entry point
├── requirements.txt    # Dependencies
├── README.md           # Full report with reasoning explanation
└── tools/
    ├── __init__.py     # Exports ALL_TOOLS
    ├── weather.py      # @tool — WeatherAPI.com
    └── attractions.py  # @tool — DuckDuckGo search
```

### Quick Start

```powershell
cd "L3/Building AI Agents with LangChain"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # Edit .env: set LLM_PROVIDER + API keys
python main.py
```

---

## 🤖 Project 6 — AI Content Creator Agent (n8n)

### Purpose

An **intelligent, no-code automation system** built on n8n that researches trending topics via Tavily, generates platform-specific content (LinkedIn, X/Twitter, Blog) using a local Ollama LLM, and publishes results to Google Sheets — all on a 6-hour automated schedule.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Platform** | n8n v1.0+ (workflow automation) |
| **LLM** | Ollama — Llama 3.1 (8B+, local inference) |
| **Web Search** | Tavily Search API (5 sources per topic) |
| **Data Store** | Google Sheets (read topics + write results) |
| **Scheduling** | Every 6 hours (cron trigger) |
| **Deployment** | npm global or Docker container |
| **Architecture** | JSON workflow definition (importable into any n8n instance) |

### Workflow Architecture

```
┌─────────────────┐
│ Schedule Trigger│  (Every 6 hours)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Google Sheets   │  ← Read all rows
│ (Read Topics)   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Filter Pending  │  ← Status = "Pending"
└────────┬────────┘
         ▼
┌─────────────────┐
│ Tavily Research │  ← Web search (5 sources/topic)
└────────┬────────┘
         │
         ├────────────────┬────────────────┐
         ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LinkedIn Gen │  │   X Gen      │  │  Blog Gen    │
│ (Ollama)     │  │  (Ollama)    │  │  (Ollama)    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       └────────┬────────┴────────┬────────┘
                ▼
         ┌─────────────┐
         │ Update Sheet│  ← Write content + timestamp + "Completed"
         └─────────────┘
```

### Content Specifications

| Platform | Length | Tone |
|----------|--------|------|
| **LinkedIn** | 120–200 words | Professional, insightful |
| **X (Twitter)** | Max 280 characters | Concise, engaging |
| **Blog** | 150–200 words | Informative, neutral |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| No-code (n8n) | Accessible to non-developers; visual debugging |
| Local LLM (Ollama) | Privacy-focused; no per-token costs |
| Google Sheets as store | Free, collaborative, no database setup |
| Status tracking | "Pending" → "Completed" prevents duplicate processing |
| Parallel content generation | LinkedIn / X / Blog generated simultaneously for speed |

### Quick Start

```powershell
# Install n8n
npm install -g n8n

# Install Ollama + model
# Download from https://ollama.ai, then:
ollama pull llama3.1

# Start n8n
n8n start
# Open http://localhost:5678
# Import "AI Content Creator Agent.json" workflow

# Configure credentials in n8n UI:
#   - Tavily API key
#   - Google Sheets OAuth2
#   - Ollama connection (localhost:11434)
```

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **n8n** | v1.0+ | Workflow automation platform |
| **Ollama** | Latest | Local AI model runtime |
| **Llama 3.1** | 8B+ | Language model for generation |
| **Node.js** | 18+ | n8n runtime |
| **RAM** | 8 GB min (16 GB recommended) | Llama 3.1 inference |

---

## 🏪 Project 7 — Competitor Intelligence System (LangGraph)

### Purpose

A **conversational AI decision-support assistant** for clothing retailers to discover nearby competitors, estimate footfall trends, and generate actionable market intelligence reports — all through natural language conversation. Built with **LangGraph** + **LangChain** + **Ollama/OpenAI**, using entirely free OpenStreetMap data.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **AI Framework** | LangGraph ≥ 0.2 + LangChain ≥ 0.3 |
| **LLM Providers** | Ollama (LLaMA 3.1) / OpenAI (GPT-4o-mini) |
| **Location Data** | OpenStreetMap — Nominatim (geocoding) + Overpass API (POI search) |
| **Data Models** | Pydantic v2 |
| **Caching** | diskcache (disk-based response caching) |
| **CLI** | Rich (Markdown rendering, spinners, tables) |
| **Web UI** | Streamlit (chat interface) |
| **Paid APIs** | None required — fully free data sources |
| **Python Version** | 3.10+ |

### Agent Architecture

```
User Query
  → LangGraph ReAct Agent (LLM with tool-calling)
    → competitor_fetch_tool     → find nearby clothing stores (OSM)
    → footfall_estimator_tool   → estimate busy hours & footfall levels
    → report_formatter_tool     → generate full Markdown analysis report
    → location_search_tool      → geocode location via Nominatim
  → Formatted Response (Markdown tables, not raw JSON)
```

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

### Core Tools

| Tool | File | Responsibility |
|------|------|----------------|
| **location_search** | `tools/location_search.py` | Geocodes user-specified areas via Nominatim |
| **competitor_fetch** | `tools/competitor_fetch.py` | Queries Overpass API for nearby clothing/fashion stores within configurable radius |
| **footfall_estimator** | `tools/footfall_estimator.py` | Generates distance-based popularity scores & busy-hour estimates |
| **report_formatter** | `tools/report_formatter.py` | Compiles competitor + footfall data into a 5-section Markdown report |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Tools return pre-formatted Markdown | LLM relays results directly — works reliably even with small local models |
| Shared in-memory store (`_store.py`) | Tools pass data to each other without JSON serialization through the LLM |
| Post-processing guard | If the LLM describes tool output instead of relaying it, the agent injects actual output automatically |
| Free data sources only | No paid API keys required for location/competitor data |
| Dual interface (CLI + Streamlit) | Serves both developers and non-technical users |

### Multi-Turn Conversation

```
You > List clothing stores near Koramangala
      → [table: 25 stores with distance, type, address]

You > What are the peak hours?
      → [footfall table: High/Medium/Low per store, busiest days]

You > Generate a full competitor report
      → [5-section Markdown report saved to data/reports/]

You > Now check Indiranagar
      → [new search, new table]

You > Compare both areas
      → [comparative analysis]
```

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `DEFAULT_RADIUS_KM` | `2` | Search radius in km |
| `DEMO_MODE` | `true` | Use synthetic demo data (no network) |
| `CACHE_TTL_SECONDS` | `3600` | Cache expiry |

### Quick Start

```powershell
cd "L3/Building your First AI Agent with LangGraph"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # Edit .env: set LLM_PROVIDER

# Start Ollama (if using local LLM)
ollama pull llama3.1
ollama serve

# Interactive CLI
python main.py

# Single query
python main.py --query "List clothing competitors near my area"

# Demo mode (offline, no API calls)
python main.py --demo

# Web UI
streamlit run app.py
```

---

## � Project 8 — Logistics Optimization System (CrewAI)

### Purpose

A **production-grade, multi-agent logistics optimization system** that pre-computes all numerical metrics in Python and delegates only interpretation and strategy synthesis to the LLM — eliminating arithmetic hallucination entirely. Built with **CrewAI** and a local **Ollama** LLM.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Agent Framework** | CrewAI ≥ 0.86.0 |
| **LLM** | Ollama — LLaMA 3.1 (local, via OpenAI-compatible `/v1` endpoint) |
| **Process** | Sequential (Analyst → Strategist) |
| **Key Innovation** | Deterministic Metrics Engine — all numbers pre-computed in Python |
| **Python Version** | 3.10+ |

### Architecture — Hallucination-Proof Pipeline

The system's key design principle: **the LLM never calculates — it only interprets**.

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

### Pre-Computed Metrics (No LLM Math)

| Metric | Computation |
|--------|-------------|
| Route speed | `distance_km / delivery_time_hr` |
| Cost per km | `fuel_cost_usd / distance_km` |
| Route overlaps | Same origin→destination served by multiple route IDs |
| Excess stock | `stock_level - (turnover_rate × reorder_point)` |
| Monthly waste | `max(excess, 0) × holding_cost_per_unit_usd` |
| Classification | Turnover < 1.0 → Overstocked · > 6.0 → Stockout Risk · = 6.0 → Borderline |

Prompt guardrail: *"⚠️ CRITICAL: Do NOT recalculate any numbers. Use ONLY the pre-computed values."*

### Agents

| Agent | Role | What It Does |
|-------|------|--------------|
| **Logistics Analyst** | Identify inefficiencies | Exhaustively flags every route violation (speed < 80 km/hr, cost/km > $0.45), overlapping routes, and inventory classifications |
| **Optimization Strategist** | Propose strategies | Prioritized action plans for route consolidation, re-routing, inventory right-sizing; bounded savings estimates; quick wins |

### Sample Output (Demo Dataset)

| Metric | Value |
|--------|-------|
| Routes flagged (speed < 80) | R001, R002, R003, R006 |
| Cost-inefficient route | R002 ($0.49/km) |
| Overlapping routes | R001 + R006 (Chicago → Detroit) |
| Most overstocked product | P003 — 7,400 excess units |
| Monthly holding waste | $10,301 |
| Annual holding waste | $123,612 |
| Strategist savings | ≤ $123,612 (bounded) |

### Quick Start

```powershell
cd "L3/Building your First AI Agent with CrewAI"
pip install -r requirements.txt

# Ensure Ollama is running with llama3.1
ollama pull llama3.1
ollama serve

# Default run
python main.py

# Custom data + output
python main.py --data path/to/data.json --output output/report.md

# Quiet mode
python main.py --quiet
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_TEMPERATURE` | `0.2` | Lower → more deterministic |
| `CREW_VERBOSE` | `true` | Show agent reasoning |
| `MAX_EXECUTION_TIMEOUT` | `120` | Max seconds for pipeline |

---

## 🧠 Project 9 — Health Classification ML Model

### Purpose

An **end-to-end machine learning pipeline** for **Anova Insurance** that predicts whether an individual is **Healthy (0)** or **Unhealthy (1)** based on 22 lifestyle, biometric, and medical features. The model supports premium pricing decisions by quantifying health risk.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Format** | Jupyter Notebook (single self-contained file) |
| **ML Framework** | scikit-learn 1.3+ |
| **Optional Boost** | XGBoost 1.7+ |
| **Visualization** | Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Python Version** | 3.10+ |

### Models Compared

| Model | Type | Key Config |
|-------|------|------------|
| Logistic Regression | Linear | `max_iter=1000` |
| Decision Tree | Tree | Default (gini) |
| Random Forest | Ensemble | `n_estimators=200` |
| Gradient Boosting | Ensemble | `n_estimators=200` |
| K-Nearest Neighbors | Distance | `n_neighbors=7` |
| SVM | Kernel | `probability=True` |
| XGBoost *(optional)* | Boosting | `n_estimators=200` |

### Pipeline Flow

```
CSV Data → EDA (5 visualization types) → Preprocessing (impute + scale)
    → Train 7 models (5-fold Stratified CV) → Select best by ROC-AUC
    → Evaluate (classification report + confusion matrix + ROC curves)
    → Feature importance → Reusable predict_health_status() function
```

### Evaluation Metrics

- **5-fold Stratified Cross-Validation** with ROC-AUC scoring
- **Test set accuracy** and **Test set ROC-AUC**
- **Classification report** (precision, recall, F1 per class)
- **Confusion matrix** and **ROC curves** for all models

### Key Features

| Feature | Description |
|---------|-------------|
| **22 input features** | Age, BMI, Blood Pressure, Cholesterol, Glucose, Heart Rate, Sleep, Exercise, Smoking, Diet, and more |
| **Stratified splitting** | 80/20 split preserving class distribution |
| **Median imputation** | Robust to outliers |
| **Standard scaling** | Required for SVM, KNN, Logistic Regression |
| **Prediction function** | `predict_health_status(input_dict)` returns label + probability |

### Quick Start

```powershell
cd "L4"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib seaborn scikit-learn jupyter xgboost
jupyter notebook "Building your First ML Model.ipynb"
```

> **Note:** Place the dataset CSV (`mDugQt7wQOKNNIAFjVku_Healthcare_Data_Preprocessed_FIXED.csv`) in the `L4/` directory before running.

### Sample Prediction

```python
example = {
    'Age': 45, 'BMI': 28.5, 'Blood_Pressure': 130, 'Cholesterol': 210,
    'Glucose_Level': 105, 'Heart_Rate': 80, 'Sleep_Hours': 6,
    'Exercise_Hours': 0.5, 'Water_Intake': 1.5, 'Stress_Level': 7,
    'Smoking': 2, 'Alcohol': 1, 'Diet': 0, 'MentalHealth': 1,
    'PhysicalActivity': 0, 'MedicalHistory': 1, 'Allergies': 0,
    'Diet_Type_Vegan': 0, 'Diet_Type_Vegetarian': 0,
    'Blood_Group_AB': 0, 'Blood_Group_B': 1, 'Blood_Group_O': 0
}
result = predict_health_status(example)
# {'prediction': 'Unhealthy', 'unhealthy_probability': 0.8723}
```

---

## � Project 10 — NYC Taxi Trip Duration

### Purpose

A **geospatial regression notebook** that predicts NYC taxi trip duration (seconds) from pickup/dropoff coordinates and timestamp features. Covers the complete supervised-learning workflow — EDA, feature engineering, model comparison, and residual analysis.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Format** | Jupyter Notebook (single self-contained file) |
| **ML Framework** | scikit-learn 1.3+ |
| **Visualization** | Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Python Version** | 3.10+ |

### Models Compared

| Model | Type |
|-------|------|
| Linear Regression | Parametric baseline |
| Random Forest Regressor | Ensemble — Bagging |
| **Gradient Boosting Regressor** | **Ensemble — Boosting** ✅ |

### Feature Engineering

| Feature | Source |
|---------|--------|
| `distance_km` | Haversine formula on pickup/dropoff coords |
| `pickup_hour` | Hour extracted from pickup datetime |
| `pickup_day` | Day of week (0=Mon … 6=Sun) |
| `pickup_month` | Month of year |
| `is_weekend` | Saturday or Sunday flag |
| `rush_hour` | Weekday 7–9 AM or 4–7 PM flag |

**Target:** `trip_duration` (seconds) — log-transformed before training; outliers (top/bottom 1%) removed.

### Key Results

- **Best Model:** Gradient Boosting Regressor (highest R², lowest RMSE & MAE)
- **Top Predictors:** `distance_km`, `pickup_hour`, pickup coordinates, `rush_hour`
- **Dataset:** 1,499 cleaned trips (post outlier removal)

### Quick Start

```powershell
cd "L4/Foundational ML Algorithms"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook "nyc_taxi_trip_duration(1).ipynb"
```

---

## 💧 Project 11 — Water Quality Prediction (Deep Learning)

### Purpose

An **end-to-end deep learning notebook** that trains two Multi-Layer Perceptron (MLP) neural networks on India's **Central Pollution Control Board (CPCB)** water quality dataset — one for **WQI regression** and one for **multi-class water quality classification** — demonstrating the full ML workflow from EDA through residual analysis on real environmental data.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Format** | Jupyter Notebook (single self-contained file) |
| **ML Framework** | scikit-learn 1.3+ (`MLPRegressor` / `MLPClassifier`) |
| **Visualization** | Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Dataset** | CPCB Water Quality — 19,029 records, 15 physicochemical indicators |
| **Python Version** | 3.10+ |

### Dual-Task Problem

| Task | Model | Output |
|------|-------|--------|
| **WQI Regression** | `MLPRegressor` | Continuous Water Quality Index score |
| **Quality Classification** | `MLPClassifier` | 5-class water quality category |

### Neural Network Architecture

Both models share the same deep MLP backbone:

```
Input(15) → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU)
    ├── → Dense(1, Linear)     [Regression — MSE loss]
    └── → Dense(5, Softmax)    [Classification — Cross-Entropy loss]
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Early Stopping | Yes (`n_iter_no_change=20`) |
| Max Iterations | 500 |
| Validation Split | 10% |

### Dataset

| Property | Value |
|----------|-------|
| **Source** | Central Pollution Control Board (CPCB), India |
| **Records** | 19,029 samples |
| **Period** | 2019–2022 |
| **Features** | Year, pH, EC, CO3, HCO3, Cl, SO4, NO3, TH, Ca, Mg, Na, K, F, TDS (15 total) |
| **Regression Target** | `WQI` (continuous) |
| **Classification Target** | `Water Quality Classification` (5 classes) |

### Pipeline Flow

```
CSV Load (19,029 rows)
    → EDA (correlation heatmap, distribution plots, WQI box plots)
    → Preprocessing (drop NaN → label-encode → 80/20 split → StandardScaler)
    → Model 1: MLPRegressor (512→256→128→64→1)  ← MSE loss
        → Evaluate: R², RMSE, MAE
        → Plot: loss curve, actual vs. predicted, residuals
    → Model 2: MLPClassifier (512→256→128→64→5) ← Cross-Entropy
        → Evaluate: Accuracy, F1-Macro, F1-Weighted
        → Plot: confusion matrix, per-class P/R/F1 bar chart
    → Final Summary Table
```

### Evaluation Metrics

**Regression (Model 1)**

| Metric | Description |
|--------|-------------|
| R² Train / Test | Coefficient of determination |
| RMSE Test | Root Mean Squared Error |
| MAE Test | Mean Absolute Error |

**Classification (Model 2)**

| Metric | Description |
|--------|-------------|
| Accuracy Train / Test | Proportion of correct predictions |
| F1 Macro | Unweighted average F1 across 5 classes |
| F1 Weighted | Class-size-weighted average F1 |

### Quick Start

```powershell
cd "L4/Introduction to Deep Learning using PyTorch"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook water_quality_prediction.ipynb
```

> **No external API keys or additional downloads required.** `water_quality.csv` is bundled in the project folder. Run all cells top-to-bottom (Kernel → Restart & Run All).

---

## 🐦 Project 12 — Comparative Sentiment Analysis (NLP)

### Purpose

An **end-to-end NLP benchmark notebook** that trains and rigorously compares **four neural architectures** — Vanilla RNN, LSTM, GRU, and fine-tuned BERT — on the **Twitter Sentiment140** dataset (1.6M tweets). Each model is evaluated across seven dimensions: Accuracy, Precision, Recall, F1-Score, ROC-AUC, training time, and peak memory — culminating in deployment recommendations by use case.

### Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Format** | Jupyter Notebook (single self-contained file) |
| **Deep Learning Framework** | PyTorch 2.0+ |
| **Transformer Library** | HuggingFace Transformers (`bert-base-uncased`) |
| **Classical ML / Metrics** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | Sentiment140 — 1,600,000 tweets, balanced 6,000-sample subset |
| **Task** | Binary sentiment classification (Negative / Positive) |
| **Python Version** | 3.10+ |
| **GPU** | Optional (CUDA auto-detected; falls back to CPU) |

### Four-Model Comparison

| Model | Architecture | Params | Pre-trained | Strength |
|-------|-------------|:------:|:-----------:|----------|
| **Vanilla RNN** | 2-layer RNN + Embedding | ~2M | ✗ | Baseline / academic |
| **LSTM** | 2-layer LSTM + Embedding | ~2M | ✗ | Long-range context, CPU-friendly |
| **GRU** | 2-layer GRU + Embedding | ~2M | ✗ | Faster than LSTM, near-equal accuracy |
| **BERT** | Fine-tuned `bert-base-uncased` | ~110M | ✓ | Highest accuracy via contextual embeddings |

### Model Architectures

**RNN / LSTM / GRU**

```
Embedding(vocab=10K, dim=64) → RNN/LSTM/GRU(hidden=128, layers=2, drop=0.3)
    → last hidden state → Dropout(0.3) → Linear(128→2) → CrossEntropyLoss
```

**BERT**

```
bert-base-uncased (110M params) → [CLS] token → Linear(768→2) → CrossEntropyLoss
Optimizer: AdamW (lr=2e-5, weight_decay=0.01) + linear warmup + gradient clip (1.0)
```

### Dataset

| Property | Value |
|----------|-------|
| **Name** | Sentiment140 (Stanford) |
| **Total Records** | 1,600,000 tweets |
| **Training Sample** | 6,000 (balanced — 3,000 Negative / 3,000 Positive) |
| **Split** | 70% Train / 15% Validation / 15% Test (stratified) |
| **Download** | [Kaggle — Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) |
| **File** | `training.1600000.processed.noemoticon.csv` |

> **Dataset is NOT bundled.** Download from Kaggle and place the CSV in the same folder as the notebook before running.

### Pipeline Flow

```
CSV (1.6M tweets) → Balance + Sample (6K) → Clean text (regex)
    → 70/15/15 stratified split
    → Vocab build (RNN) + BertTokenizer (BERT)
    ├── Train RNN / LSTM / GRU  (Adam, 5 epochs, batch=64)
    └── Fine-tune BERT          (AdamW + linear warmup, 5 epochs, batch=32)
         │
         ▼
7-metric comparative table → 5 saved visualisations → deployment guide
```

### Metrics Reported

| Metric | Description |
|--------|-------------|
| Accuracy | Overall fraction correct on test set |
| Precision (weighted) | Class-weighted positive predictive value |
| Recall (weighted) | Class-weighted true positive rate |
| F1-Score (weighted) | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve (positive-class probability) |
| Train Time (s) | Wall-clock training duration per model |
| Peak Mem (MB) | `tracemalloc` peak traced memory |

### Visualizations Generated

| File (saved to `outputs/`) | Content |
|----------------------------|---------| 
| `f1_comparison.png` | Bar chart — weighted F1 per model |
| `metrics_comparison.png` | Grouped bar — all 5 performance metrics |
| `training_curves.png` | 2×4 grid — loss & accuracy curves (train vs. val) |
| `confusion_matrices.png` | Normalised confusion matrices (4 panels) |
| `computational_cost.png` | Training time & peak memory side-by-side |
| `metrics_summary.csv` | Full comparative metrics table export |

### Deployment Recommendations

| Scenario | Best Model | Rationale |
|----------|:----------:|-----------|
| Production (accuracy critical) | **BERT** | Highest F1 via deep contextual embeddings |
| Edge / mobile / low-latency | **GRU** | Best speed-accuracy trade-off |
| Rapid prototyping / CPU-only | **LSTM or GRU** | Fast training, no GPU requirement |
| Academic baseline | **Vanilla RNN** | Simplest architecture, useful lower-bound |

### Quick Start

```powershell
cd "L4/Natural Language Processing using PyTorch"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Place training.1600000.processed.noemoticon.csv in the notebook folder, then:
jupyter notebook "Natural Language Processing using PyTorch/sentiment_analysis_comparative.ipynb"
```

> **GPU tip:** For ~20× faster BERT training, install the CUDA build of PyTorch from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## 🔧 Project 13 — Parameter-Efficient Fine-Tuning with QLoRA

### Purpose

A **memory-optimized fine-tuning implementation** using QLoRA (Quantized Low-Rank Adaptation) to fine-tune BERT for sentiment analysis with 4-bit quantization and LoRA adapters, enabling efficient training on consumer GPUs while maintaining competitive performance.

### Technical Approach

**QLoRA Innovation**: Combines 4-bit NF4 quantization with LoRA adapters to reduce memory usage by ~75% while training only 0.16% of the model parameters. This makes large model fine-tuning accessible on 4GB+ GPUs.

**Key Technical Features:**
- **4-bit Quantization**: Uses NF4 (Normal Float 4-bit) with double quantization
- **LoRA Adapters**: Low-rank matrices (rank=16) inserted into attention layers
- **Parameter Efficiency**: Only Query and Value projection layers are adapted
- **Memory Optimization**: Gradient accumulation and dynamic batch sizing
- **Experiment Tracking**: Full W&B integration for reproducibility

### Architecture Details

| Component | Configuration |
|-----------|---------------|
| Base Model | BERT-base-uncased (110M params) |
| Quantization | 4-bit NF4 + double quantization |
| LoRA Rank | 16 (α=32, dropout=0.1) |
| Target Modules | `["query", "value"]` attention layers |
| Trainable Params | ~177K (0.16% of total) |
| Memory Usage | ~4GB VRAM (vs ~16GB full fine-tuning) |

### Training Pipeline

```python
# 1. Setup quantized model with LoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config
)

# 2. Apply LoRA configuration
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)

# 3. Train with parameter efficiency
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

### Expected Results

- **Accuracy**: 92-95% on IMDB sentiment classification
- **F1-Score**: 0.92-0.95 (binary classification)
- **Training Time**: 10-15 minutes on RTX 3080
- **Memory Usage**: 4GB+ VRAM (adjustable batch size)
- **Parameter Efficiency**: 90%+ reduction in trainable parameters

### Quick Start

```powershell
cd "L4/Finetuning LLMs"
jupyter notebook qlora_bert_local_gpu.ipynb
# Follow notebook sections 1-8 sequentially
# Setup W&B API key when prompted
```

**Prerequisites**: CUDA-compatible GPU, W&B account, 4GB+ VRAM

---

## 🔤 Project 14 — Building Custom Tokenizers from Scratch

### Purpose

A **foundational tokenization system** that teaches BPE (Byte Pair Encoding) implementation from scratch using the WikiText-2 dataset, providing the essential building blocks for training language models with proper vocabulary construction and text preprocessing.

### Technical Approach

**BPE Algorithm**: Iteratively merges the most frequent character pairs to build optimal subword vocabularies that balance compression efficiency with semantic coherence. This approach handles out-of-vocabulary words naturally and captures morphological patterns.

**Production-Grade Implementation:**
- **Normalization Pipeline**: NFD → Lowercase → Strip Accents
- **Pre-tokenization**: Whitespace splitting with configurable patterns
- **Vocabulary Management**: 30K tokens with frequency-based filtering
- **Special Token Handling**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- **Post-processing**: Template-based special token insertion

### Tokenizer Architecture

```python
# Complete tokenization pipeline
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)]
)
```

### Dataset Processing

| Stage | Operations |
|-------|-----------|
| **Loading** | WikiText-2-v1 via HuggingFace datasets |
| **Cleaning** | Remove `<unk>`, normalize headers, deduplicate |
| **Statistics** | ~36K articles, ~11.8M characters, ~145 chars/article avg |
| **Training** | BPE with min_frequency=2, vocab_size=30K |
| **Validation** | Coverage analysis, compression ratio measurement |

### Core Learning Objectives

- **Subword Tokenization**: Understand why BPE outperforms word-level approaches
- **Vocabulary Construction**: Learn optimal vocab sizes and frequency thresholds  
- **Text Normalization**: Master preprocessing for multilingual robustness
- **Tokenizer Evaluation**: Measure coverage, compression, and OOV rates
- **Integration Patterns**: Save/load tokenizers for downstream model training

### Key Metrics

| Metric | Expected Value | Significance |
|--------|----------------|--------------|
| **Vocabulary Size** | 30,000 | Standard for most NLP models |
| **Coverage Rate** | >99.5% | Minimal out-of-vocabulary tokens |
| **Compression Ratio** | 3.2:1 | Characters to tokens efficiency |
| **Avg Subword Length** | 3.2-4.1 chars | Optimal granularity |

### Implementation Pipeline

```python
# 1. Load and explore WikiText-2
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1") 

# 2. Clean and preprocess corpus
cleaned_data = preprocess_texts(dataset['train']['text'])

# 3. Train BPE tokenizer
trainer = BpeTrainer(vocab_size=30000, special_tokens=special_tokens)
tokenizer.train_from_iterator(cleaned_data, trainer=trainer)

# 4. Evaluate and export
analyze_tokenization_quality(tokenizer)
tokenizer.save("custom_bpe_tokenizer.json")
```

### Quick Start

```powershell
cd "L4/Training LLMs from Scratch" 
jupyter notebook bpe_tokenizer_wikitext2.ipynb
# Execute all sections sequentially
# Examine vocabulary and compression analysis
# Export tokenizer for use in other projects
```

**Integration**: The trained tokenizer can be directly used with HuggingFace Transformers via `PreTrainedTokenizerFast(tokenizer_object=tokenizer)`.

---

## 🤖 Project 15 — Advanced AI Agents with AutoGen

### Purpose

A **comprehensive collection of four production-grade multi-agent systems** demonstrating Microsoft AutoGen's enterprise patterns for autonomous agent coordination, including Group Chat, Stateful FSM, Reflection, and Sequential patterns with function calling.

### Technical Architecture

**Multi-Agent Coordination Patterns**:
- **Group Chat**: Dynamic speaker selection for complex workflows (Bill Management)
- **Stateful FSM**: Explicit state transitions for deterministic progression (Portfolio Manager)
- **Reflection Pattern**: Creator-critic loops for iterative improvement (Content Creation)
- **Sequential with Tools**: Linear workflows with function calling (Health Assistant)

**Technology Stack:**
- **Framework**: AutoGen 0.11+ (Microsoft's multi-agent framework)
- **LLM Provider**: Groq (llama-3.1-70b-versatile, llama-3.2-90b-vision-preview)
- **Vision Processing**: Groq Vision API with base64 encoding
- **Agent Types**: ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat

### Four Included Projects

#### 1. **Bill Managing Agent** (Group Chat Pattern)
- **Purpose**: Automated bill processing and expense categorization
- **Agents**: User Proxy → Group Manager → Bill Processor (Vision) → Summarizer
- **Key Feature**: Vision API extracts text from receipts, categorizes into 8+ expense types
- **Output**: Structured expense summaries with category breakdowns

#### 2. **Financial Portfolio Manager** (Stateful FSM Pattern)
- **Purpose**: Personalized investment advisory for Indian markets
- **States**: INIT → PORTFOLIO_ANALYSIS → RECOMMENDATION → REPORT → COMPLETE
- **Agents**: User Proxy → Portfolio Analyst → Financial Advisor → Report Generator
- **Key Feature**: Risk-based asset allocation across equity, debt, gold, emergency funds

#### 3. **Smart Content Creation** (Reflection Pattern)
- **Purpose**: Iterative content improvement through creator-critic dialogue
- **Agents**: Content Creator ⇄ Content Critic (bidirectional)
- **Key Feature**: Numeric scoring (Language Clarity + Technical Accuracy), threshold-based approval
- **Termination**: Auto-approval at 8.5/10 score or max 5 iterations

#### 4. **Smart Health Assistant** (Sequential with Function Calling)
- **Purpose**: Personalized health assessment and meal planning
- **Agents**: User Proxy → BMI Agent (Tool) → Meal Planner → Summary Agent
- **Key Feature**: AutoGen function calling with `calculate_bmi()` tool
- **Output**: BMI classification, health recommendations, customized meal plans

### Agent Pattern Comparison

| Pattern | Best For | Complexity | Termination | Example Use |
|---------|----------|------------|-------------|-------------|
| **Group Chat** | Complex, non-linear workflows | High | Group decision | Multi-step document processing |
| **Stateful FSM** | Sequential stages with validation | Medium | State: COMPLETE | Financial planning, workflows |
| **Reflection** | Iterative improvement | Low | Quality threshold | Content review, code review |
| **Sequential + Tools** | Linear with external tools | Medium | Keyword signal | Health assessment, calculations |

### Quick Start

```powershell
cd "L4/Building Advanced AI Agents with AutoGen"
pip install autogen groq pillow requests

# Set up API key
$env:GROQ_API_KEY="your_groq_api_key_here"

# Navigate to any project
cd "Bill Managing Agent"
jupyter notebook bill_management_agent_autogen.ipynb
```

### Learning Outcomes

- **Multi-Agent Design**: Understand when to use different coordination patterns
- **State Management**: Implement FSMs for deterministic workflows
- **Function Calling**: Register and use tools with AutoGen agents
- **Vision APIs**: Process images with vision-enabled LLMs
- **Quality Control**: Build reflection loops for automated QA
- **Termination Strategies**: Implement robust exit conditions

### Integration Patterns

**Reusable Components**:
```python
# LLM configuration (all projects)
llm_config = {
    "model": "llama-3.1-70b-versatile",
    "api_key": os.environ.get("GROQ_API_KEY"),
    "temperature": 0.3,
    "max_tokens": 2000,
}

# Function registration (Health Assistant)
@user_proxy.register_for_execution()
@bmi_agent.register_for_llm(description="Calculate BMI")
def calculate_bmi(weight: float, height: float) -> dict:
    # Implementation
    pass

# Group chat (Bill Management)
group_chat = autogen.GroupChat(
    agents=[user_proxy, bill_processor, summarizer],
    messages=[],
    max_round=10
)
manager = autogen.GroupChatManager(groupchat=group_chat)
```

**Deployment**: Each project is self-contained and can be deployed as:
- Jupyter notebooks (development/demos)
- Python scripts (automation)
- REST API endpoints (production)

---

## 🔄 Project 16 — LangGraph Multi-Agent Research System

### Purpose

A **production-grade graph-based multi-agent system** demonstrating LangGraph's conditional routing, RAG integration, web search coordination, and intelligent summarization with persistent memory.

### Technical Architecture

**Graph-Based Workflow**:
```
__start__ → Router Agent (classify intent)
              ↓
    ┌─────────┼─────────┐
    │         │         │
   LLM      RAG    Web Research
    │         │         │
    └─────────┼─────────┘
              ↓
      Summarization Agent
              ↓
           __end__
```

**Technology Stack:**
- **Orchestration**: LangGraph 0.2+ (graph-based agent coordination)
- **LLM**: Groq (llama-3.3-70b-versatile) for ultra-fast inference
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2) - free, local
- **Vector Store**: ChromaDB (persistent document storage)
- **Web Search**: Tavily API (advanced search with AI summaries)
- **Memory**: MemorySaver (conversation persistence)

### Agent Responsibilities

| Agent | Trigger Condition | Data Sources | Output |
|-------|-------------------|--------------|--------|
| **Router** | Always (entry point) | Query analysis | Route: 'llm' \| 'rag' \| 'web_research' |
| **Direct LLM** | General knowledge questions | Model parameters | Direct answer from LLM |
| **RAG Agent** | Domain-specific queries | ChromaDB (AI/ML corpus) | Context + LLM-generated answer |
| **Web Research** | Current events, recent data | Tavily search results | Real-time web information |
| **Summarization** | Always (exit point) | All previous sources | Consolidated final answer |

### Intelligent Routing Logic

**Router Classification**:
- **'llm'**: Definitions, explanations, general knowledge (e.g., "Explain quantum computing")
- **'rag'**: Questions about stored documents (e.g., "What are components of Agentic AI?")
- **'web_research'**: Current events, breaking news (e.g., "Latest GPT-5 developments in 2026")

**Implementation**:
```python
def router_node(state: AgentState) -> AgentState:
    system_prompt = """Analyze query and route to:
    - 'llm': General knowledge
    - 'rag': AI/ML from knowledge base
    - 'web_research': Current events
    
    Respond with ONE word only.
    """
    route = llm.invoke([SystemMessage(system_prompt), 
                       HumanMessage(state['query'])])
    return {"route": route.content.strip().lower()}
```

### State Management

**AgentState Schema**:
```python
class AgentState(TypedDict):
    query:             str              # User question
    route:             str              # Selected agent path
    retrieved_context: str              # RAG/Web results
    llm_response:      str              # Direct LLM answer
    final_summary:     str              # Consolidated output
    messages:          List[BaseMessage] # Conversation history
    metadata:          dict             # Timestamps, sources
```

### RAG Configuration

**Vector Store Setup**:
- **Collection**: "ai_ml_knowledge" (sample AI/ML documents included)
- **Retrieval**: Top-K similarity search (k=5)
- **Persistence**: `./chroma_db` directory
- **Embeddings**: 384-dimensional vectors (MiniLM-L6-v2)

**Web Search Setup**:
- **Provider**: Tavily (advanced search depth)
- **Results**: Top 5 with AI-generated summaries
- **Content**: Excludes raw HTML for efficiency

### Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| **Router Classification** | 0.5-1s | Single LLM call |
| **RAG Retrieval** | 0.2-0.5s | ChromaDB similarity search |
| **Web Search** | 1-2s | Tavily API latency |
| **LLM Generation** | 1-3s | Groq (fastest available) |
| **Total (RAG path)** | 2-5s | End-to-end |
| **Total (Web path)** | 3-6s | End-to-end |

### Quick Start

```powershell
cd "L4/Building Advanced AI Agents with LangGraph"
pip install langgraph langchain langchain-groq langchain-community \
    langchain-chroma chromadb sentence-transformers tavily-python

# Set up API keys
$env:GROQ_API_KEY="your_groq_api_key"
$env:TAVILY_API_KEY="your_tavily_api_key"

# Run notebook
jupyter notebook langgraph_multi_agent_groq.ipynb
```

### Example Queries

**RAG Query**:
```python
result = graph.invoke({"query": "What are key components of Agentic AI?"})
# Routes to: RAG → retrieves from ChromaDB → summarizes
```

**Web Research Query**:
```python
result = graph.invoke({"query": "Latest OpenAI GPT developments in 2026"})
# Routes to: Web Research → Tavily search → summarizes with sources
```

**Direct LLM Query**:
```python
result = graph.invoke({"query": "Explain gradient descent vs SGD"})
# Routes to: LLM → direct answer from model
```

### Learning Outcomes

- **Graph-Based Orchestration**: Build complex agent workflows with LangGraph
- **Conditional Routing**: Implement dynamic agent selection based on query intent
- **RAG Implementation**: Set up vector stores, embeddings, and retrieval
- **Multi-Source Synthesis**: Combine local knowledge + web search
- **State Management**: Pass context through graph nodes
- **Memory Persistence**: ChromaDB for vectors, MemorySaver for conversations
- **API Integration**: Coordinate multiple services (Groq, Tavily, HuggingFace)

### Extension Ideas

- **Multi-Turn Conversations**: Add conversation history to state
- **Advanced RAG**: Hybrid search (dense + sparse), re-ranking
- **Tool Integration**: Calculator, code execution, SQL queries
- **Streaming Output**: Real-time token streaming
- **Human-in-the-Loop**: Add approval nodes for critical decisions
- **Evaluation**: Track routing accuracy, retrieval quality (RAGAS framework)

**Deployment**: Package as:
- Streamlit UI for interactive demos
- FastAPI REST endpoint for production
- Docker container with all dependencies

---

## 🔗 Shared Technical Concepts

### Multi-LLM Provider Pattern

All AI projects (Projects 2, 3, 4, 5, 6, 7, 8) implement LLM provider abstractions:

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
| Factory Function | Projects 4, 7 | `get_llm_provider()` / `get_llm()` returns configured instance |
| Dataclass Models | Projects 3, 4 | `@dataclass` for structured data (RAGResponse, SearchResult) |
| Pydantic Settings | Projects 2, 7 | Environment variable configuration with validation |
| LangGraph ReAct Agent | Project 7 | `StateGraph` with tool-calling loop + post-processing |
| CrewAI Sequential Crew | Project 8 | Multi-agent pipeline with deterministic pre-compute + prompt guardrails |
| Deterministic Pre-compute | Project 8 | All numerical metrics computed in Python, LLM interprets only |
| No-Code Workflow | Project 6 | n8n JSON workflow with scheduled triggers |

---

## ⚙️ Global Prerequisites

| Requirement | Version | Required By | Notes |
|-------------|---------|-------------|-------|
| **Python** | 3.10+ | All projects | 3.9+ for RAG only |
| **pip** | Latest | All projects | Package manager |
| **Node.js** | 18+ | Project 6 | n8n runtime |
| **n8n** | v1.0+ | Project 6 | Workflow platform |
| **Ollama** | Latest | Projects 3, 4, 6, 7, 8 | Local LLM runtime |
| **Tesseract OCR** | Latest | Project 2 | System-level install |
| **Internet** | — | All (except Ollama/demo) | API access |
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
#  TAVILY WEB SEARCH (Projects 4, 6)
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
| Ollama not responding | 2, 3, 4, 6, 7, 8 | Run `ollama serve` in a separate terminal; pull model with `ollama pull llama3.1` |
| n8n workflow not triggering | 6 | Ensure n8n is running (`n8n start`); check Google Sheets OAuth2 credentials in n8n UI |
| Overpass API timeout | 7 | Reduce `DEFAULT_RADIUS_KM`; enable `DEMO_MODE=true` for offline testing |
| LangGraph tool output garbled | 7 | Expected with very small models; post-processing guard auto-corrects; try a larger model |
| Streamlit not loading | 7 | Run `streamlit run app.py` from the project directory; ensure port 8501 is free |
| `Connection error` (CrewAI) | 8 | Ollama must be running: `ollama serve`. Verify with `ollama list` |
| LLM omits route violations | 8 | Non-deterministic LLM behavior; re-run — prompt guardrails catch most cases |

---

## 📄 License & Author

**License:** MIT — Use freely for educational and portfolio purposes.

**Author:** AHILL S

---

<div align="center">

*Built with dedication to production-grade software engineering, clean architecture, and responsible AI design.*

</div>