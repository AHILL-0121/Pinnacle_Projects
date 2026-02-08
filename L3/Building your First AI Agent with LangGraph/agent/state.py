"""
Agent state schema for the LangGraph conversational agent.
"""

from __future__ import annotations

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state that flows through every node in the agent graph.

    Attributes:
        messages: Conversation history (accumulated via the ``add_messages`` reducer).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
