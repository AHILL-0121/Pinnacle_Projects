"""
main.py – Entry point for the Intelligent Travel Assistant AI.

Run:
    python main.py

The assistant enters an interactive loop where the user types a
destination city and receives weather + attraction information
powered by a LangChain tool-calling agent.
"""

import sys
from agent import build_agent


BANNER = r"""
╔══════════════════════════════════════════════════════╗
║      🌍  Intelligent Travel Assistant AI  🌍        ║
║  Powered by LangChain Tool-Calling Agent            ║
╠══════════════════════════════════════════════════════╣
║  Type a city name to get:                           ║
║    • 🌦️  Current weather                            ║
║    • 📍 Top tourist attractions                     ║
║                                                      ║
║  Commands:  quit / exit / q  →  leave                ║
╚══════════════════════════════════════════════════════╝
"""


def main() -> None:
    """Interactive CLI loop."""
    print(BANNER)

    try:
        executor = build_agent(verbose=True)
    except EnvironmentError as exc:
        print(f"\n❌ Setup error: {exc}")
        sys.exit(1)

    print("✅ Agent ready. Enter a destination to begin.\n")

    while True:
        try:
            user_input = input("🗺️  Enter destination (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye! Happy travels!")
            break

        # Wrap bare city names with a natural prompt for the agent
        query = (
            f"I'm planning a trip to {user_input}. "
            "Please give me the current weather and the top tourist attractions."
        )

        print("\n⏳ Agent is thinking…\n")
        try:
            result = executor.invoke({"input": query})

            # Extract answer – classic returns dict, LangGraph returns dict with 'messages'
            if isinstance(result, dict):
                if "output" in result:
                    answer = result["output"]
                elif "messages" in result:
                    answer = result["messages"][-1].content
                else:
                    answer = str(result)
            else:
                answer = str(result)

            print("\n" + "─" * 56)
            print(answer)
            print("─" * 56 + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\n❌ Agent error: {exc}\n")


if __name__ == "__main__":
    main()
