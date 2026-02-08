"""
LangGraph agent definition.

Builds a **ReAct-style tool-calling agent** with the following graph:

    ┌──────────┐
    │  agent   │ ← LLM decides whether to call a tool or respond
    └────┬─────┘
         │
    ┌────▼─────┐
    │  tools   │ ← execute the chosen tool
    └────┬─────┘
         │ (loops back to agent until done)
         ▼
       [END]
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)


def _build_agent_node():
    """Create the LLM agent node with tools bound."""
    from services.llm_service import get_llm
    llm = get_llm(temperature=0.2)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        messages = state["messages"]

        # Inject system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = llm_with_tools.invoke(messages)

        # --- Post-processing: ensure tool output is relayed, not described ---
        # Small LLMs often describe tool output ("This is a list of…") instead
        # of relaying the formatted Markdown.  Detect this and inject the real
        # tool output into the response.
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            # This is a final answer (no more tool calls).
            # Collect the most recent tool results from the conversation.
            tool_outputs: list[str] = []
            for msg in reversed(state["messages"]):
                if hasattr(msg, "type"):
                    if msg.type == "tool" and msg.content:
                        tool_outputs.append(msg.content)
                    elif msg.type == "ai":
                        break  # stop at the AI message that triggered tools

            if tool_outputs:
                resp_text = response.content or ""
                # Our tools always produce Markdown tables ("|") or headers ("###").
                # If the LLM's response lacks these, it described instead of relayed.
                has_table = "|" in resp_text and "---" in resp_text
                has_header = "###" in resp_text or "## " in resp_text
                if not has_table and not has_header:
                    combined = "\n\n".join(reversed(tool_outputs))
                    # Append LLM's brief commentary if it adds anything useful
                    commentary = resp_text.strip()
                    if commentary:
                        combined += "\n\n" + commentary
                    response.content = combined
                    logger.debug("Post-processing: injected tool output into response.")

        return {"messages": [response]}

    return agent_node


def _should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Edge function: route to tools if the last message has tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph agent graph."""
    agent_node = _build_agent_node()
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Define edges
    graph.add_conditional_edges("agent", _should_continue)
    graph.add_edge("tools", "agent")  # After tool execution, go back to agent

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully.")
    return compiled


# Module-level convenience
def get_agent():
    """Return a ready-to-use compiled agent graph."""
    return build_graph()
