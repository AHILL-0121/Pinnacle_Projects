# RAG Systems Essentials

A **portfolio-grade Retrieval-Augmented Generation (RAG) system** for question answering over AI research papers. Built with confidence gating, hallucination prevention, and multi-LLM support.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Confidence Gating** | Refuses to answer when retrieval confidence is low (prevents hallucination) |
| **Section-Aware Retrieval** | Boosts chunks with explicit section headers for structural queries |
| **Multi-LLM Support** | Gemini, Groq, and Ollama (NO OpenAI dependency) |
| **Citation Discipline** | Max 2 citations per answer, filtered to relevant papers |
| **Query-Type Detection** | Factual, table, section, and cross-paper query handling |
| **MMR Diversity** | Maximal Marginal Relevance prevents redundant context |

## ✅ Test Results

All 5 edge-case tests pass:

| Test | Description | Result |
|------|-------------|--------|
| Table Exactness | Refuses to hallucinate missing table data | ✅ PASS |
| Negative Refusal | Clean refusal with proper citations | ✅ PASS |
| Cross-Paper Reasoning | Concise synthesis without extra metrics | ✅ PASS |
| Section Precision | Identifies Section 3.2 for multi-head attention | ✅ PASS |
| Knowledge Boundaries | No data leakage across papers | ✅ PASS |

**Grade: A** (5/5 PASS)

## 📚 Theoretical Foundation

Built on three foundational AI papers:

| Paper | Key Concepts |
|-------|--------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer architecture, multi-head attention |
| [RAG for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) | Retrieval-augmented generation, DPR-style retrieval |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | In-context learning, few-shot prompting |

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│  Query Type Detection               │
│  • Factual / Table / Section        │
│  • Cross-paper synthesis            │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Query Encoder (MiniLM-L6-v2)       │
│  • 384-dim embeddings               │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Vector Store (FAISS)               │
│  • MMR for diversity                │
│  • Section-aware boosting           │
│  • Confidence scoring               │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Confidence Gate                    │
│  • 0.75 threshold for factual       │
│  • 0.50 threshold for general       │
│  • Refuse if below threshold        │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  LLM Generator                      │
│  • Query-specific prompts           │
│  • Token limits by query type       │
│  • Strict no-hallucination rules    │
└─────────────────────────────────────┘
     │
     ▼
Answer + Source Citations (max 2)
```

## 🚀 Quick Start

### 1. Installation

```bash
cd "L2/RAG Systems Essentials"
pip install -r requirements.txt
```

### 2. Add Research Papers

Place PDF files in the `data/papers/` directory:
```
data/papers/
├── attention_is_all_you_need.pdf
├── rag_paper.pdf
└── gpt3_paper.pdf
```

### 3. Configure LLM Provider

**Option A: Ollama (Recommended - Local, Free)**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1:latest
ollama serve
```

**Option B: Groq (Cloud, Fast)**
```bash
export GROQ_API_KEY="your-api-key"
```

**Option C: Gemini (Cloud)**
```bash
export GEMINI_API_KEY="your-api-key"
```

### 4. Run the System

```bash
# Interactive mode with Ollama
python main.py --provider ollama

# Single question
python main.py -q "What is multi-head attention?"

# Run edge case tests
python test_edge_cases.py
```

## 💻 CLI Commands

| Command | Description |
|---------|-------------|
| `/ingest` | Ingest PDFs from papers directory |
| `/stats` | Show index statistics |
| `/provider <name>` | Change LLM provider |
| `/help` | Show available commands |
| `/quit` | Exit application |

## 📂 Project Structure

```
RAG Systems Essentials/
├── main.py                 # Entry point
├── cli.py                  # Interactive CLI
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── test_edge_cases.py      # Edge case test suite
├── src/
│   ├── document_processor.py   # PDF extraction & semantic chunking
│   ├── embeddings.py           # MiniLM-L6-v2 embeddings
│   ├── vector_store.py         # FAISS with MMR
│   ├── retriever.py            # Dual-strategy retrieval
│   ├── llm_providers.py        # Gemini/Groq/Ollama
│   └── rag_pipeline.py         # Main orchestration
├── data/
│   ├── papers/             # Place PDFs here
│   └── index/              # Persisted vector index
└── tests/                  # Unit tests
```

## 🔧 Configuration

Key settings in `config.py`:

```python
# Chunking
chunk_size = 400       # tokens per chunk
chunk_overlap = 75     # overlap tokens

# Retrieval
top_k = 6              # chunks to retrieve
use_mmr = True         # diversity via MMR
similarity_threshold = 0.3

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.50         # general queries
FACTUAL_CONFIDENCE_THRESHOLD = 0.75 # factual/table queries

# Token Limits by Query Type
cross_paper_max_tokens = 256   # concise synthesis
factual_max_tokens = 150       # short factual
general_max_tokens = 1024      # detailed explanations
```

## 🧪 Example Usage

```python
from src.rag_pipeline import RAGPipeline
from src.llm_providers import LLMManager
from pathlib import Path

# Setup LLM
llm = LLMManager()
llm.setup_ollama(model='llama3.1:latest')

# Initialize pipeline
rag = RAGPipeline(llm_manager=llm)
rag.initialize(index_path=Path('data/index'))

# Query
response = rag.query("What is multi-head attention?")
print(response.answer)
print(response.sources)
```

## 📝 Sample Q&A

**Question**: Which section introduces multi-head attention, and why is it necessary?

**Answer**: 
> Section 3.2 of the Transformer paper introduces multi-head attention. It is necessary because multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this capability.

**Sources**:
- Attention Is All You Need, Section 3.2.2: Multi-Head Attention, Page 2

---

**Question**: What is the BLEU score of GPT-3 on WMT-14 English-German?

**Answer**:
> This information is not present in the indexed documents.

**Sources**:
- Language Models are Few-Shot Learners, Tables, Page 2

*(Correctly refuses - GPT-3 paper doesn't report this, and doesn't leak Transformer's BLEU scores)*

## 🎓 Why This Project Matters

This is **not** a toy demo. It demonstrates:

| Skill | Implementation |
|-------|----------------|
| **Retrieval Theory** | DPR-style dense retrieval, MMR diversity |
| **Hallucination Prevention** | Confidence gating, constrained generation |
| **Query Understanding** | Type detection, query-specific prompts |
| **System Design** | Modular architecture, clean abstractions |
| **Testing** | Edge case coverage, regression prevention |

**Perfect for**: Technical interviews, portfolio showcases, academic projects.

## 🔒 Hallucination Prevention

Multiple layers of protection:

1. **Confidence Gating**: Refuses when retrieval confidence < threshold
2. **Strict Prompts**: "Answer ONLY from context" rules
3. **Citation Discipline**: Max 2 citations, filtered to query-relevant papers
4. **Table Query Detection**: Extra-strict threshold (0.75) for numeric data
5. **Cross-Paper Isolation**: Prevents data leakage between papers

## 📄 License

MIT License - Use freely for educational and portfolio purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python test_edge_cases.py`
4. Submit a pull request

---

*Built with ❤️ for learning RAG systems*
