"""
Configuration module for the Competitor Intelligence System.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── LLM Provider ────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "openai"

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Google Maps ─────────────────────────────────────────────
GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ── Defaults ────────────────────────────────────────────────
DEFAULT_LOCATION: str = os.getenv("DEFAULT_LOCATION", "Koramangala, Bangalore")
DEFAULT_RADIUS_KM: float = float(os.getenv("DEFAULT_RADIUS_KM", "2"))
DEFAULT_BUSINESS_TYPE: str = os.getenv("DEFAULT_BUSINESS_TYPE", "clothing_store")

# ── Application ─────────────────────────────────────────────
DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# ── Paths ───────────────────────────────────────────────────
DATA_DIR: Path = _PROJECT_ROOT / "data"
DEMO_DATA_DIR: Path = DATA_DIR / "demo"
CACHE_DIR: Path = DATA_DIR / "cache"
REPORTS_DIR: Path = DATA_DIR / "reports"

# Ensure directories exist
for _dir in (DATA_DIR, DEMO_DATA_DIR, CACHE_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
