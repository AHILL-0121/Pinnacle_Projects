"""
CLI entry point for the Competitor Intelligence System.

Usage:
    python main.py                          # Interactive chat mode
    python main.py --query "List clothing competitors near Koramangala"
    python main.py --demo                   # Quick demo run
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage

console = Console()

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("competitor-intel")


def _print_response(text: str) -> None:
    """Render a Markdown response in the terminal."""
    console.print()
    console.print(Markdown(text))
    console.print()


def run_query(agent, query: str) -> str:
    """Send a single query to the agent and return the final text response."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 25},
    )
    # The last message from the agent is the final answer
    final = result["messages"][-1]
    return final.content


def interactive_mode(agent) -> None:
    """Run a REPL-style interactive chat loop."""
    console.print(
        "[bold cyan]╔══════════════════════════════════════════════════════════╗[/]"
    )
    console.print(
        "[bold cyan]║   CompetitorIQ – Clothing Retail Intelligence Agent      ║[/]"
    )
    console.print(
        "[bold cyan]╚══════════════════════════════════════════════════════════╝[/]"
    )
    console.print(
        "[dim]Type your question below. Type 'quit' or 'exit' to leave.[/]\n"
    )

    if config.DEMO_MODE:
        console.print("[yellow]⚡ Running in DEMO mode (synthetic data).[/]\n")

    messages = []
    while True:
        try:
            user_input = console.input("[bold green]You > [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        messages.append(HumanMessage(content=user_input))

        with console.status("[bold blue]Thinking…[/]", spinner="dots"):
            try:
                result = agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": 25},
                )
                messages = list(result["messages"])
                answer = messages[-1].content
            except Exception as e:
                logger.exception("Agent error")
                answer = f"⚠️ Error: {e}"

        _print_response(answer)


def demo_run(agent) -> None:
    """Run a predefined demo flow."""
    queries = [
        "List the top clothing store competitors near Koramangala, Bangalore within 2 km.",
        "What are the busiest hours for these competitors?",
        "Generate a full competitor analysis report.",
    ]
    console.print("[bold magenta]── Demo Mode ─────────────────────────────[/]\n")

    messages = []
    for q in queries:
        console.print(f"[bold green]You > [/]{q}")
        messages.append(HumanMessage(content=q))
        with console.status("[bold blue]Thinking…[/]", spinner="dots"):
            try:
                result = agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": 25},
                )
                messages = list(result["messages"])
                answer = messages[-1].content
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        _print_response(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Competitor Intelligence AI Agent (CLI)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a pre-defined demo flow.",
    )
    args = parser.parse_args()

    # Build the agent
    from agent import get_agent

    console.print("[dim]Building agent graph…[/]")
    agent = get_agent()
    console.print("[dim]Agent ready.[/]\n")

    if args.query:
        answer = run_query(agent, args.query)
        _print_response(answer)
    elif args.demo:
        demo_run(agent)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
