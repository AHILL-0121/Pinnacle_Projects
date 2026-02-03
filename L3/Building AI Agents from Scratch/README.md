# 🔬 AI Web Research Agent

<div align="center">

**An Autonomous Research Agent powered by the ReAct (Reason + Act) Pattern**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)]()

[Getting Started](#-quick-start) • [Documentation](#-documentation) • [API Reference](#-api-reference) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Configuration Guide](#-configuration-guide)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [LLM Providers](#-llm-providers)
- [Output Formats](#-output-formats)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **AI Web Research Agent** is an intelligent system that automates the research process using the ReAct (Reason + Act) design pattern. It combines Large Language Model reasoning with real-time web search capabilities to produce comprehensive, well-sourced research reports.

### What It Does

```
Input: "Artificial Intelligence in Healthcare"
         ↓
   [30-60 seconds]
         ↓
Output: Complete research report with:
        ✓ Executive summary
        ✓ 4-6 researched sections
        ✓ Cited sources
        ✓ HTML + Markdown formats
```

### Why ReAct?

The **ReAct Pattern** (Reason + Act) mirrors human research behavior:

| Human Researcher | AI Agent |
|------------------|----------|
| Think about what to search | **PLAN** - LLM generates research questions |
| Search the web | **ACT** - Tavily API searches the web |
| Read and understand | **SYNTHESIZE** - LLM processes results |
| Write the report | **REPORT** - Generate structured output |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🧠 Intelligent Planning** | LLM generates targeted research questions covering multiple angles |
| **🔍 Real-Time Web Search** | Tavily API fetches current information from the web |
| **📊 Smart Synthesis** | AI-powered summarization with fact extraction |
| **📝 Dual Output** | Generates both Markdown (`.md`) and HTML (`.html`) reports |
| **🌐 Auto Browser Open** | HTML report opens automatically for immediate viewing |
| **🤖 Multi-LLM Support** | Choose from Gemini, Groq, Ollama, or OpenRouter |
| **💰 Free Tier Optimized** | Token-efficient prompts for free API usage |
| **🎨 Beautiful CLI** | Rich terminal output with progress indicators |

---

## 🏗️ System Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                       │
│                        "Tamil Nadu Elections 2026"                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PLANNING (Reason)                                                  │
│  ─────────────────────────                                                   │
│  LLM generates 4 focused research questions:                                 │
│    • What are the key issues?                                                │
│    • Who are the major candidates?                                           │
│    • What are the current polls showing?                                     │
│    • What policies are being proposed?                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: ACTING (Act)                                                       │
│  ─────────────────────                                                       │
│  For each question → Tavily Web Search API                                   │
│    • Returns 3 results per question                                          │
│    • Extracts: Title, URL, Content snippet                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: SYNTHESIS (Reason)                                                 │
│  ──────────────────────────                                                  │
│  LLM processes each question's results:                                      │
│    • Extracts key facts                                                      │
│    • Formats as bullet points                                                │
│    • No hallucination - only sourced facts                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: REPORT GENERATION                                                  │
│  ──────────────────────────                                                  │
│  Output Files:                                                               │
│    📄 reports/topic_name.md   (Markdown)                                     │
│    🌐 reports/topic_name.html (Auto-opens in browser)                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              agent.py                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  LLM Providers  │    │  WebSearchTool  │    │ ResearchAgent   │         │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤         │
│  │ • GeminiProvider│    │ • Tavily API    │    │ • plan()        │         │
│  │ • GroqProvider  │    │ • Result parsing│    │ • act()         │         │
│  │ • OllamaProvider│    │ • Rate limiting │    │ • synthesize()  │         │
│  │ • OpenRouter    │    └─────────────────┘    │ • research()    │         │
│  └─────────────────┘                           └─────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime environment |
| pip | Latest | Package manager |
| Internet | - | API calls & web search |

### Installation (3 minutes)

```powershell
# 1. Navigate to project directory
cd "Pinnacle L3"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# source venv/bin/activate     # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure API keys
copy .env.example .env
# Edit .env with your API keys (see Configuration Guide)
```

### First Run

```powershell
# Basic research
python agent.py "Artificial Intelligence in Healthcare"

# With custom output name
python agent.py "Climate Change" -o climate_report.md

# Using specific provider
python agent.py "Quantum Computing" --provider groq
```

---

## ⚙️ Configuration Guide

### Environment Variables

Create a `.env` file in the project root:

```env
# ============================================
# LLM PROVIDER CONFIGURATION
# ============================================

# Primary provider: gemini | groq | ollama | openrouter
LLM_PROVIDER=gemini

# Google Gemini (Recommended for free tier)
# Get key: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

# Groq (Fast inference)
# Get key: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama (Local - no API key needed)
# Download: https://ollama.ai
OLLAMA_MODEL=llama3.2

# OpenRouter (Access to multiple models)
# Get key: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemma-2-9b-it:free

# ============================================
# WEB SEARCH CONFIGURATION
# ============================================

# Tavily API (Required)
# Get key: https://tavily.com
TAVILY_API_KEY=your_tavily_api_key

# ============================================
# AGENT SETTINGS
# ============================================

# Number of search results per question (default: 3)
MAX_SEARCH_RESULTS=3

# Number of research questions to generate (default: 4)
MAX_QUESTIONS=4
```

### API Key Sources

| Provider | Free Tier | Rate Limit | Get Key |
|----------|-----------|------------|---------|
| **Tavily** | 1000 searches/month | 100/day | [tavily.com](https://tavily.com/) |
| **Gemini** | Free (with limits) | 15 RPM | [makersuite.google.com](https://makersuite.google.com/app/apikey) |
| **Groq** | Free (generous) | Varies | [console.groq.com](https://console.groq.com/keys) |
| **Ollama** | Unlimited (local) | None | [ollama.ai](https://ollama.ai/) |
| **OpenRouter** | Pay-per-use + free models | Varies | [openrouter.ai](https://openrouter.ai/keys) |

---

## 📖 Usage Guide

### Command Line Interface

```
usage: agent.py [-h] [-o OUTPUT] [--provider {gemini,groq,ollama,openrouter}] topic

AI Web Research Agent - ReAct Pattern

positional arguments:
  topic                          Research topic to investigate

optional arguments:
  -h, --help                     Show this help message and exit
  -o OUTPUT, --output OUTPUT     Output filename (auto-generated if not specified)
  --provider {gemini,groq,ollama,openrouter}
                                 Override LLM provider from .env
```

### Usage Examples

```powershell
# Basic - Auto-generated filename, opens in browser
python agent.py "Electric Vehicles Market 2026"

# Custom filename - Saved to reports/my_report.md
python agent.py "Renewable Energy Trends" -o my_report.md

# Use Groq for faster inference
python agent.py "SpaceX Starship Progress" --provider groq

# Use local Ollama (no internet for LLM)
python agent.py "Machine Learning Basics" --provider ollama

# Use OpenRouter free models
python agent.py "Cryptocurrency Regulations" --provider openrouter
```

### Output Location

All reports are saved to the `reports/` directory:

```
reports/
├── electric_vehicles_market.md      # Markdown report
├── electric_vehicles_market.html    # HTML report (auto-opens)
├── renewable_energy_trends.md
├── renewable_energy_trends.html
└── ...
```

---

## 📚 API Reference

### ResearchAgent Class

```python
class ResearchAgent:
    """Main orchestrator for the ReAct research workflow."""
    
    def __init__(self):
        """Initialize with LLM provider and search tool from environment."""
        
    def plan(self, topic: str) -> List[str]:
        """Generate research questions using LLM.
        
        Args:
            topic: Research topic
            
        Returns:
            List of 4 research questions
        """
        
    def act(self, questions: List[str]) -> List[SearchResult]:
        """Execute web searches for each question.
        
        Args:
            questions: List of research questions
            
        Returns:
            List of SearchResult objects with titles, URLs, content
        """
        
    def synthesize(self, question: str, results: List[SearchResult]) -> str:
        """Synthesize search results into coherent summary.
        
        Args:
            question: The research question
            results: Search results for this question
            
        Returns:
            Markdown-formatted synthesis with bullet points
        """
        
    def research(self, topic: str, output_file: Optional[str] = None) -> ResearchReport:
        """Execute full research workflow.
        
        Args:
            topic: Research topic
            output_file: Optional custom filename (auto-generated if None)
            
        Returns:
            ResearchReport dataclass with all findings
        """
```

### Data Classes

```python
@dataclass
class SearchResult:
    title: str      # Page title
    url: str        # Source URL
    content: str    # Content snippet (truncated to 300 chars)

@dataclass
class SynthesizedSection:
    question: str           # Research question
    synthesis: str          # LLM-generated summary
    results: List[SearchResult]  # Source results

@dataclass
class ResearchReport:
    topic: str                          # Original topic
    timestamp: str                      # Generation time
    questions: List[str]                # Generated questions
    sections: List[SynthesizedSection]  # Research sections
    introduction: str                   # Executive summary
    conclusion: str                     # Key findings
```

---

## 🤖 LLM Providers

### Provider Comparison

| Provider | Speed | Quality | Cost | Best For |
|----------|-------|---------|------|----------|
| **Gemini** | Medium | High | Free tier | Default choice |
| **Groq** | ⚡ Fast | High | Free tier | Quick research |
| **Ollama** | Slow | Varies | Free (local) | Privacy, offline |
| **OpenRouter** | Varies | Varies | Pay/Free | Model variety |

### Recommended Models

| Provider | Model | Notes |
|----------|-------|-------|
| Gemini | `gemini-2.0-flash` | Good balance of speed/quality |
| Groq | `llama-3.3-70b-versatile` | Very fast, high quality |
| Ollama | `llama3.2` | Good local option |
| OpenRouter | `google/gemma-2-9b-it:free` | Free, decent quality |

### Switching Providers

```powershell
# Via command line (temporary)
python agent.py "Topic" --provider groq

# Via .env file (permanent)
# Edit .env: LLM_PROVIDER=groq
```

---

## 📄 Output Formats

### Markdown Report Structure

```markdown
# Research Report: [Topic]

Generated: [Timestamp]
Research Method: AI-Powered Web Research (ReAct Pattern)

---

## Executive Summary

[Brief introduction to the topic and what the report covers]

---

## Table of Contents

1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. Conclusion
6. Sources

---

## Section 1: [Question 1]

- Key finding 1
- Key finding 2
- Key finding 3

[Repeat for each section...]

---

## Conclusion

[Summary of key findings and insights]

---

## Sources

1. [Source Title](URL)
2. [Source Title](URL)
...
```

### HTML Report Features

- **Light theme** - Clean, professional appearance
- **Responsive design** - Works on all screen sizes
- **Styled headings** - Blue accent with left border
- **Proper bullet points** - Clear hierarchy
- **Clickable sources** - Links open in browser
- **Auto-opens** - Immediately viewable after generation

---

## 🔧 Troubleshooting

### Common Issues

#### 1. API Rate Limit Exceeded

**Error:** `429 RESOURCE_EXHAUSTED` or `Rate limit exceeded`

**Solutions:**
```powershell
# Option 1: Wait and retry (automatic with Gemini)
# Agent will retry with exponential backoff

# Option 2: Switch provider
python agent.py "Topic" --provider groq

# Option 3: Use local Ollama
python agent.py "Topic" --provider ollama
```

#### 2. Model Decommissioned

**Error:** `model has been decommissioned`

**Solution:** Update model name in `.env`:
```env
# Old (deprecated)
GROQ_MODEL=llama-3.1-70b-versatile

# New (current)
GROQ_MODEL=llama-3.3-70b-versatile
```

#### 3. Import Errors in VS Code

**Error:** `Import "tavily" could not be resolved`

**Solution:** Select the correct Python interpreter:
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.\venv\Scripts\python.exe`

#### 4. Tavily Search Returns Empty

**Possible causes:**
- Invalid API key
- Rate limit reached
- Network issues

**Solution:**
```powershell
# Verify API key
echo $env:TAVILY_API_KEY

# Test with fewer questions
# Edit .env: MAX_QUESTIONS=2
```

#### 5. HTML Not Opening in Browser

**Possible cause:** Browser not set as default

**Solution:** Manually open the HTML file from `reports/` folder

---

## ⚡ Performance Optimization

### Token Efficiency Settings

The agent is optimized for free tier API usage:

| Setting | Value | Purpose |
|---------|-------|---------|
| `MAX_SEARCH_RESULTS` | 3 | Fewer results = fewer tokens |
| `MAX_QUESTIONS` | 4 | Focused research |
| `max_output_tokens` | 800 | Limits response size |
| Content truncation | 300 chars | Reduces context size |

### Estimated Token Usage Per Research

| Phase | Tokens (approx) |
|-------|-----------------|
| Planning | ~200 |
| Synthesis (×4) | ~800 |
| Introduction | ~150 |
| Conclusion | ~150 |
| **Total** | **~1,300** |

### Speed Optimization

| Provider | Avg. Time | Notes |
|----------|-----------|-------|
| Groq | 30-45s | Fastest |
| Gemini | 45-60s | Good balance |
| OpenRouter | 45-90s | Varies by model |
| Ollama | 60-120s | Depends on hardware |

---

## 📁 Project Structure

```
Pinnacle L3/
├── agent.py              # Main application (850+ lines)
│   ├── LLM Providers     # Gemini, Groq, Ollama, OpenRouter
│   ├── WebSearchTool     # Tavily API integration
│   ├── ResearchAgent     # ReAct orchestration
│   └── CLI Interface     # Argument parsing
│
├── reports/              # Generated reports directory
│   ├── *.md              # Markdown reports
│   └── *.html            # HTML reports
│
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── .env                  # Your API keys (gitignored)
├── .gitignore            # Git ignore rules
└── README.md             # This documentation
```

### Dependencies

```
python-dotenv    # Environment variable management
requests         # HTTP client
google-genai     # Gemini API
groq             # Groq API
ollama           # Ollama client
tavily-python    # Web search API
rich             # Beautiful CLI output
markdown         # MD to HTML conversion
```

---

## 🗺️ Roadmap

### Current Version: v1.0

✅ Multi-LLM support (Gemini, Groq, Ollama, OpenRouter)  
✅ ReAct pattern implementation  
✅ Markdown + HTML output  
✅ Auto browser open  
✅ LLM-generated filenames  
✅ Free tier optimization  

### Planned Features

| Version | Feature | Status |
|---------|---------|--------|
| v1.1 | PDF export | 📋 Planned |
| v1.2 | Source credibility scoring | 📋 Planned |
| v1.3 | Research history/sessions | 📋 Planned |
| v2.0 | Web UI (Streamlit/Next.js) | 📋 Planned |
| v2.1 | RAG with vector database | 📋 Planned |
| v2.2 | Multi-agent debate pattern | 📋 Planned |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Update README for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[ReAct Paper](https://arxiv.org/abs/2210.03629)** - Yao et al., 2022
- **[Tavily](https://tavily.com/)** - AI-optimized web search
- **[Google Gemini](https://ai.google.dev/)** - LLM capabilities
- **[Groq](https://groq.com/)** - Fast LLM inference
- **[Ollama](https://ollama.ai/)** - Local LLM runtime
- **[Rich](https://rich.readthedocs.io/)** - Beautiful terminal formatting

---

<div align="center">

**Built with ❤️ using the ReAct Pattern**

[Report Bug](../../issues) • [Request Feature](../../issues) • [Documentation](#-table-of-contents)

</div>
