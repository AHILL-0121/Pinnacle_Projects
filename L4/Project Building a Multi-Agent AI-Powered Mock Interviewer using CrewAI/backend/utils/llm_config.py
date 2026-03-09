"""
LLM Configuration Module
========================
Supports Ollama (llama3.1) as primary and Groq as secondary.
NO OpenAI API is used anywhere.

Uses direct API calls — no CrewAI LLM wrapper, no LiteLLM, no OpenAI SDK:
  • Ollama  → HTTP POST to /api/chat  (local, free)
  • Groq    → Official `groq` Python SDK  (cloud, free tier)
"""

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"


# --- Ollama config ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- Groq config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")


def call_llm(
    messages: list[dict],
    provider: LLMProvider,
    temperature: float = 0.3,
) -> str:
    """
    Call the configured LLM and return the response text.

    Args:
        messages:    OpenAI-style list of {"role": ..., "content": ...} dicts.
        provider:    LLMProvider.OLLAMA or LLMProvider.GROQ
        temperature: Sampling temperature.

    Returns:
        The assistant's reply as a plain string.
    """
    if provider == LLMProvider.OLLAMA:
        return _call_ollama(messages, temperature)
    elif provider == LLMProvider.GROQ:
        return _call_groq(messages, temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ── Ollama (direct HTTP) ──────────────────────────────────────────────────────

def _call_ollama(messages: list[dict], temperature: float) -> str:
    import requests  # stdlib-bundled in any Python environment
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )


# ── Groq (official SDK) ───────────────────────────────────────────────────────

def _call_groq(messages: list[dict], temperature: float) -> str:
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com/"
        )
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


# ── Health checks ─────────────────────────────────────────────────────────────

def check_ollama_available() -> bool:
    """Ping Ollama server to check if it is running."""
    try:
        import requests
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def check_groq_available() -> bool:
    """Check whether GROQ_API_KEY is configured and non-placeholder."""
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "your_groq_api_key_here"


# ── Kept for backward-compat — returns None (agents no longer use it) ─────────
def get_llm(provider: LLMProvider):  # noqa: F811
    """Deprecated shim — direct call_llm() instead."""
    return None
