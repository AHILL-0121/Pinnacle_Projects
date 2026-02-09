"""
Crew assembly for the Logistics Optimization System.

SRS Section 7 — Sequential crew with two agents and two tasks.

Usage:
    from crew import build_crew
    result = build_crew(logistics_data).kickoff()
"""

from crewai import Crew, Process

from agents import create_logistics_analyst, create_optimization_strategist
from tasks import create_analysis_task, create_optimization_task
from models.schemas import LogisticsData
from config import VERBOSE


def build_crew(logistics_data: LogisticsData) -> Crew:
    """
    Assemble and return the logistics optimization crew.

    Pipeline (sequential):
        1. Logistics Analyst   → analysis_task
        2. Optimization Strategist → optimization_task (consumes #1 output)
    """
    # ── Create agents ────────────────────────────────────────────────
    analyst = create_logistics_analyst()
    strategist = create_optimization_strategist()

    # ── Create tasks (parameterized with logistics data) ─────────────
    analysis_task = create_analysis_task(
        agent=analyst,
        logistics_data=logistics_data,
    )
    optimization_task = create_optimization_task(
        agent=strategist,
        analysis_task=analysis_task,
        logistics_data=logistics_data,
    )

    # ── Assemble crew (SRS Section 7) ────────────────────────────────
    crew = Crew(
        agents=[analyst, strategist],
        tasks=[analysis_task, optimization_task],
        process=Process.sequential,
        verbose=VERBOSE,
    )

    return crew
