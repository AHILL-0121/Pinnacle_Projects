"""
LLM service – provides a single `get_llm()` factory that returns a
LangChain-compatible chat model based on the project configuration.
"""

import config


def get_llm(temperature: float = 0.3):
    """Return the configured LLM instance.

    Supports:
      - ``ollama`` → ChatOllama (local)
      - ``openai`` → ChatOpenAI

    The provider is selected via the ``LLM_PROVIDER`` env var / config.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}. Use 'ollama' or 'openai'.")
