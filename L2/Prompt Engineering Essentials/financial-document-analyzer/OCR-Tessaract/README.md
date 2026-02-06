# Financial Document Analyzer 📊

An AI-powered system that converts financial document images (scanned PDFs, images) into concise, context-aware financial insights. This MVP delivers **structured, role-aware summaries** with **actionable insights** using a **hallucination-resistant architecture**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)
![LLM](https://img.shields.io/badge/LLM-Multi--Provider-purple.svg)

## 🎯 What This System Does

```
Upload Financial Document (PDF/Image/JSON)
         ↓
Image Preprocessing & Enhancement
         ↓
pdfplumber + OCR (Tesseract) Extraction
         ↓
Deterministic Data Validation (Code decides, not LLM!)
         ↓
Period-Keyed Structured JSON (Source of Truth)
         ↓
Role-Aware LLM Report Generation (Single Call)
         ↓
Structured Financial Summary with Confidence Score
```

### Key Differentiators

✅ **Multi-Provider LLM Support** - Choose Ollama (local), Groq (fast), Gemini, or OpenAI per request  
✅ **Hallucination-Resistant Architecture** - Deterministic validation before LLM, structured JSON as source of truth  
✅ **Chart & Graph Interpretation** - Extracts insights from visual charts (trend direction, approximate values)  
✅ **Role-Aware Summarization** - Adapts output for Investor, Analyst, Auditor, or Executive  
✅ **Structured Output** - Consistent format: Executive Summary → Metrics → Analysis → Risks → Actions  
✅ **Financial Entity Extraction** - Revenue, Profit, EPS, Ratios with period comparisons (YoY/QoQ)  
✅ **Period-Accurate Data** - Always uses LATEST period values, not first-seen data

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  • File Upload • Role Selection • LLM Provider Selection    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  • /analyze endpoint • /providers endpoint • Validation     │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   OCR Service   │  │  pdfplumber     │  │ Chart Analyzer  │
│   (Tesseract)   │  │  (Tables)       │  │  (Vision API)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Deterministic Financial Entity Extractor           │
│  • Metric extraction • Value normalization • Period detect   │
│  • DATA VALIDATION (code decides, NOT LLM)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Period-Keyed Structured JSON                    │
│  • Source of Truth for all metrics                           │
│  • latest_period, earliest_period, changes                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Report Generator (Single Call)              │
│  • Pre-templated values • No hallucination risk              │
│  • Ollama / Groq / Gemini / OpenAI                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Structured Summary + Confidence Score          │
│  • JSON output • Export ready (JSON/Markdown)                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** installed on your system:
   - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`
3. **LLM Provider** - Choose at least ONE:
   - **Ollama** (local, free) - Recommended for getting started
   - **Groq** (fast, free tier) - Best for speed
   - **Google Gemini** - Good quality
   - **OpenAI** (best quality, paid)

### Installation

```bash
# Clone/navigate to the project
cd financial-document-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and configure your LLM provider (see below)
```

### LLM Provider Configuration

Edit `.env` and set your preferred provider:

#### Option 1: Ollama (Local, Free) - Recommended to Start
```bash
# First, install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.2
# Start Ollama server:
ollama serve
```
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# For chart analysis (optional - needs API key)
VISION_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
VISION_MODEL=gemini-1.5-flash
```

#### Option 2: Groq (Fast & Free Tier)
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-groq-key
GROQ_MODEL=llama-3.3-70b-versatile

VISION_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
VISION_MODEL=gemini-1.5-flash
```

#### Option 3: Google Gemini
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-1.5-pro
VISION_PROVIDER=gemini
VISION_MODEL=gemini-1.5-flash
```

#### Option 4: OpenAI (Best Quality)
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o
VISION_PROVIDER=openai
VISION_MODEL=gpt-4o
```

### Running the Application

**Terminal 1 - Start Backend:**
```bash
cd backend
python run.py
```
Backend will be available at: http://localhost:8000

**Terminal 2 - Start Frontend:**
```bash
cd frontend
streamlit run app.py --server.port 8501
```
Frontend will be available at: http://localhost:8501

### Quick Test (Command Line)

```bash
# Test with the included sample document
cd docs
curl -X POST -F "file=@NovaBank_Financial_Report.pdf" -F "role=investor" -F "llm_provider=ollama" http://localhost:8000/analyze
```

### API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Providers Status: http://localhost:8000/providers

## 📖 Usage Guide

### 1. Select LLM Provider

Choose your preferred LLM before analysis:
- **🏠 Ollama** - Run locally, no API costs, good privacy
- **⚡ Groq** - Very fast inference, free tier available
- **🌟 Gemini** - Good quality, free tier available
- **🔷 OpenAI** - Best quality, paid

### 2. Upload Document

- Click "Browse files" or drag-and-drop
- Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, JSON
- Max file size: 50 MB

### 3. Select Your Role

| Role | Focus Areas | Key Metrics |
|------|-------------|-------------|
| 💰 **Investor** | Growth, profitability, risks | Revenue Growth, NPM, EPS, ROE, Dividend Yield |
| 📈 **Analyst** | Ratios, trends, comparisons | ROE, ROA, D/E, Current Ratio, Operating Margin |
| 🔍 **Auditor** | Compliance, anomalies, red flags | Asset Quality, CET1, NPL Ratio, Provision Coverage |
| 👔 **Executive** | Strategic performance | Revenue, EBITDA, Market Share, Cost Efficiency |

### 3. Configure Options (Optional)

- **LLM Provider**: Select which AI to use (Ollama/Groq/Gemini/OpenAI)
- **Document Type**: Auto-detect or specify (Annual Report, Quarterly, etc.)
- **Company Name**: For context in analysis
- **Fiscal Period**: e.g., "Q3 2024", "FY2023"
- **Focus Areas**: Comma-separated areas to emphasize

### 4. Get Results

The structured output includes:

```
📌 Executive Summary
   └── Overview + Key Highlights

📊 Key Financial Metrics
   └── Revenue, Profit, EPS, Ratios with YoY/QoQ changes

📈 Performance Analysis
   └── Revenue, Profitability, Operations analysis

📉 Chart & Graph Insights
   └── Trend direction (↑↓→), key values, textual insights

⚠️ Risks & Outlook
   └── Categorized risks with severity and mitigation

💡 Actionable Insights
   └── Prioritized recommendations with specific actions

📋 Extracted Tables
   └── Structured data from financial tables
```

## 🔌 API Reference

### POST /analyze

Analyze a financial document with selected LLM provider.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@quarterly_report.pdf" \
  -F "role=investor" \
  -F "llm_provider=ollama" \
  -F "company_name=NovaBank" \
  -F "fiscal_period=Q2 2025"
```

**Response:**
```json
{
  "success": true,
  "message": "Document analyzed successfully",
  "data": {
    "executive_summary": {
      "overview": "NovaBank demonstrated solid financial performance in Q2 2025 with revenue of USD 5.3 billion...",
      "key_highlights": [
        "Revenue reached USD 5.3 billion (latest quarter)",
        "Net Profit of USD 655 million shows strong profitability",
        "EPS of USD 1.52 indicates healthy earnings",
        "ROE at 11.8% reflects efficient capital utilization",
        "CET1 Ratio of 14.9% demonstrates robust capital position"
      ],
      "period_covered": "Q2 2025"
    },
    "key_financial_metrics": {
      "revenue": {"value": 5.3, "unit": "billion USD", "period": "Q2 2025"},
      "net_profit": {"value": 655.0, "unit": "million USD"},
      "eps": {"value": 1.52, "unit": "USD"},
      "roe": {"value": 11.8, "unit": "%"}
    },
    "performance_analysis": {...},
    "risks_and_outlook": [...],
    "actionable_insights": [...],
    "confidence_score": 0.87,
    "processing_time_seconds": 15.2
  }
}
```

### GET /providers

Get available LLM providers and their status.

**Response:**
```json
{
  "current_provider": "ollama",
  "providers": {
    "ollama": {"available": true, "name": "Ollama (Local)", "model": "llama3.2"},
    "groq": {"available": true, "name": "Groq", "model": "llama-3.3-70b-versatile"},
    "gemini": {"available": true, "name": "Google Gemini", "model": "gemini-1.5-pro"},
    "openai": {"available": false, "name": "OpenAI", "model": "gpt-4o"}
  }
}
```

### GET /roles

Get available user roles and their configurations.

### GET /document-types

Get supported document types.

## 📁 Project Structure

```
financial-document-analyzer/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app + provider selection
│   │   ├── config.py            # Configuration settings
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py       # Pydantic models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── ocr_service.py   # Tesseract OCR
│   │   │   ├── table_extractor.py
│   │   │   ├── chart_analyzer.py # Vision API integration
│   │   │   ├── entity_extractor.py # Deterministic validation
│   │   │   ├── llm_service.py   # Multi-provider LLM
│   │   │   └── summarizer.py    # Main orchestrator
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── image_processor.py
│   │       ├── pdf_handler.py   # pdfplumber integration
│   │       └── validators.py
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── app.py                   # Streamlit UI with LLM selector
│   └── requirements.txt
├── docs/
│   ├── NovaBank_Financial_Report.pdf  # Sample document
│   └── sample_output.md
├── .env.example
├── requirements.txt             # Combined requirements
└── README.md
```

## 🧪 Testing

### Test with Sample Document

```bash
# Using curl with Ollama
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@docs/NovaBank_Financial_Report.pdf" \
  -F "role=investor" \
  -F "llm_provider=ollama"

# Using curl with Groq
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@docs/NovaBank_Financial_Report.pdf" \
  -F "role=analyst" \
  -F "llm_provider=groq"
```

### Expected Results (NovaBank Q2 2025)

| Metric | Expected Value |
|--------|---------------|
| Revenue | USD 5.3 billion |
| Net Profit | USD 655 million |
| EPS | USD 1.52 |
| ROE | 11.8% |
| Equity | USD 15.5 billion |
| CET1 Ratio | 14.9% |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Primary LLM provider (ollama/groq/gemini/openai) | ollama |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_MODEL` | Ollama model name | llama3.2 |
| `GROQ_API_KEY` | Groq API key | - |
| `GROQ_MODEL` | Groq model name | llama-3.3-70b-versatile |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `GEMINI_MODEL` | Gemini model name | gemini-1.5-pro |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | OpenAI model name | gpt-4o |
| `VISION_PROVIDER` | Vision API provider | gemini |
| `VISION_MODEL` | Vision model for charts | gemini-1.5-flash |
| `OCR_LANGUAGE` | Tesseract language | eng |
| `OCR_DPI` | OCR resolution | 300 |
| `MAX_PAGES` | Max pages to process | 50 |
| `MAX_FILE_SIZE_MB` | Max upload size | 50 |

## 🔒 Hallucination Prevention

This system uses a **deterministic validation architecture** to prevent LLM hallucinations:

1. **pdfplumber + OCR Extraction** - Direct table extraction, not OCR interpretation
2. **Deterministic Validation** - Code validates financial patterns, not LLM
3. **Period-Keyed JSON** - Structured data with explicit period→metric mapping
4. **Pre-Templated Values** - LLM receives actual values already inserted in the prompt
5. **Single LLM Call** - Only ONE LLM call for report generation, no chained analysis
6. **Placeholder Detection** - Post-processing removes any remaining template markers

## 🚧 Limitations (MVP)

- ❌ No real-time stock prices
- ❌ No forecasting/predictions
- ❌ No multi-company comparison
- ❌ No user authentication
- ⚠️ Chart value extraction requires Vision API
- ⚠️ Image quality affects OCR accuracy

## 🔮 Future Enhancements (Post-MVP)

- [ ] Multi-period trend visualization
- [ ] Financial forecasting with ML models
- [ ] Risk scoring models
- [ ] Export to PDF/Excel with charts
- [ ] Enterprise audit trails
- [ ] Batch processing
- [ ] Multi-language OCR support
- [ ] Custom metric extraction rules

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Built with ❤️ for financial analysis automation**

**Supports:** Ollama • Groq • Google Gemini • OpenAI
