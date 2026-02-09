"""
Agent definitions for the Logistics Optimization System.

SRS Section 5 — Two agents:
1. Logistics Analyst  — identifies inefficiencies
2. Optimization Strategist — proposes actionable strategies
"""

from crewai import Agent, LLM
from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    VERBOSE,
)


def _build_llm() -> LLM:
    """Construct the Ollama-backed LLM used by all agents.

    Uses Ollama's OpenAI-compatible endpoint (http://localhost:11434/v1)
    so crewai routes through its native OpenAI provider — no litellm needed.
    """
    return LLM(
        model=OLLAMA_MODEL,
        provider="openai",
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",          # dummy key; Ollama ignores it
        temperature=LLM_TEMPERATURE,
    )


def create_logistics_analyst() -> Agent:
    """
    Agent 1 — Logistics Analyst

    Role : Logistics Analyst
    Goal : Identify inefficiencies in logistics operations
    Backstory: Senior logistics data analyst with deep experience in
               supply chain optimization, route planning, and inventory control.
    """
    return Agent(
        role="Logistics Analyst",
        goal=(
            "Thoroughly analyze logistics data to identify inefficiencies in "
            "delivery routes and inventory turnover. Produce structured, "
            "data-backed insights with concrete metrics."
        ),
        backstory=(
            "You are a senior logistics data analyst with 15+ years of experience "
            "in supply chain optimization, route planning, and inventory control. "
            "You have worked with Fortune 500 manufacturers and 3PL providers. "
            "You are meticulous about numbers and always back your findings with "
            "quantitative evidence. You communicate in structured reports."
        ),
        llm=_build_llm(),
        verbose=VERBOSE,
        allow_delegation=False,
    )


def create_optimization_strategist() -> Agent:
    """
    Agent 2 — Optimization Strategist

    Role : Optimization Strategist
    Goal : Propose actionable optimization strategies
    Backstory: Supply chain optimization consultant specializing in cost
               reduction, route planning, and inventory optimization.
    """
    return Agent(
        role="Optimization Strategist",
        goal=(
            "Develop clear, actionable optimization strategies based on the "
            "Logistics Analyst's insights. Each recommendation must include "
            "expected impact, implementation steps, and priority ranking."
        ),
        backstory=(
            "You are a supply chain optimization consultant with a track record "
            "of reducing logistics costs by 15-30% for mid-to-large enterprises. "
            "You specialize in route consolidation, inventory right-sizing, and "
            "warehouse network design. You always present recommendations in a "
            "prioritized, implementation-ready format with estimated ROI."
        ),
        llm=_build_llm(),
        verbose=VERBOSE,
        allow_delegation=False,
    )
