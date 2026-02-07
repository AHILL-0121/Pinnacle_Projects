"""
config.py – Centralised configuration for the Travel Assistant.
Loads environment variables and exposes settings used by tools and agents.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Supported values: "openai", "gemini", "ollama"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower().strip()
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Google Gemini ─────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Ollama (local) ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

# ── Backwards-compat alias ────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "")  # overrides per-provider model if set

# ── Weather API ──────────────────────────────────────────────────────────────
WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_BASE: str = "http://api.weatherapi.com/v1/current.json"

# ── Search ───────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
