"""
agent.py – LangChain Tool-Calling Agent for the Travel Assistant.

Constructs the agent, binds tools, and returns an AgentExecutor
ready to process user queries.

Supports both:
  - langchain_classic  (AgentExecutor + create_tool_calling_agent)
  - langgraph.prebuilt (create_react_agent)  – modern LangChain ≥ 1.x
The code auto-detects which is available at import time.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)
from tools import ALL_TOOLS

# ── Compatibility: detect available agent API ────────────────────────────────
try:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
    _USE_CLASSIC = True
except ImportError:
    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        _USE_CLASSIC = True
    except ImportError:
        from langgraph.prebuilt import create_react_agent  # noqa: F401
        _USE_CLASSIC = False


# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an Intelligent Travel Assistant AI.

Your job is to help travellers plan trips by providing:
1. **Current weather** at the destination
2. **Top tourist attractions** worth visiting

When the user gives you a destination city:
- Always call the weather tool to get live weather data.
- Always call the attractions search tool to find popular places.
- Combine the results into a single, well-formatted, friendly response.
- If a tool fails, acknowledge the issue and still present what you have.

Keep your tone friendly, concise, and helpful. Use emojis sparingly
for readability. Structure the response with clear sections.\
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


SUPPORTED_PROVIDERS = ("openai", "gemini", "ollama")


def _build_llm():
    """Initialise and return the LLM based on LLM_PROVIDER setting.

    Supported providers:
      - openai  → ChatOpenAI  (requires OPENAI_API_KEY)
      - gemini  → ChatGoogleGenerativeAI (requires GEMINI_API_KEY)
      - ollama  → ChatOllama  (requires Ollama running locally)
    """
    provider = LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Please add it to your .env file or export it."
            )
        model = LLM_MODEL or OPENAI_MODEL
        print(f"  LLM provider : OpenAI  ({model})")
        return ChatOpenAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        model = LLM_MODEL or GEMINI_MODEL
        print(f"  LLM provider : Gemini  ({model})")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            google_api_key=GEMINI_API_KEY,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = LLM_MODEL or OLLAMA_MODEL
        print(f"  LLM provider : Ollama  ({model} @ {OLLAMA_BASE_URL})")
        return ChatOllama(
            model=model,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        f"Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def build_agent(*, verbose: bool = True):
    """Create and return a ready-to-run agent executor.

    Automatically picks the best available LangChain agent backend:
      • langchain_classic  → AgentExecutor + create_tool_calling_agent
      • langgraph.prebuilt → create_react_agent  (modern path)

    Parameters
    ----------
    verbose : bool
        If True the agent prints its reasoning steps to stdout.

    Returns
    -------
    AgentExecutor or CompiledGraph
        Configured executor with weather + attractions tools bound.
    """
    llm = _build_llm()

    if _USE_CLASSIC:
        # Classic path: create_tool_calling_agent → AgentExecutor
        agent = create_tool_calling_agent(
            llm=llm,
            tools=ALL_TOOLS,
            prompt=PROMPT,
        )
        executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        return executor

    # Modern LangGraph path: create_react_agent
    from langgraph.prebuilt import create_react_agent as _create_react_agent

    graph = _create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )
    return graph
