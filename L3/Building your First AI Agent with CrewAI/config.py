"""
Configuration for the CrewAI Logistics Optimization System.

Technical Stack:
- Agent Framework: CrewAI
- LLM: Ollama (LLaMA 3.1 local)
- Execution: CLI / Script-based
"""

import os

# ── LLM Configuration ────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Temperature controls creativity vs determinism (SRS: deterministic output)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Dummy API key — Ollama ignores it, but the OpenAI SDK requires one
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# ── Performance Constraints (SRS NFR) ────────────────────────────────
# Analysis should complete within 30 seconds for MVP data size
MAX_EXECUTION_TIMEOUT = int(os.getenv("MAX_EXECUTION_TIMEOUT", "120"))  # seconds

# ── Agent Verbosity ──────────────────────────────────────────────────
VERBOSE = os.getenv("CREW_VERBOSE", "true").lower() == "true"

# ── Output Configuration ─────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
