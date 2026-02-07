# 🌍 Intelligent Travel Assistant AI

> **LangChain Tool-Calling Agent** -- MVP Implementation

An AI-powered travel assistant that accepts a destination city and returns
**current weather** and **top tourist attractions** using a LangChain
tool-calling agent for reasoning and orchestration.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How the LLM Reasoning Works](#how-the-llm-reasoning-works)
4. [Code Explanation and Program Flow](#code-explanation-and-program-flow)
5. [Tool Design](#tool-design)
6. [Setup and Run](#setup-and-run)
7. [Sample Output](#sample-output)
8. [Project Structure](#project-structure)
9. [Technology Stack](#technology-stack)
10. [MVP Success Criteria](#mvp-success-criteria)
11. [Extending the Agent](#extending-the-agent)

---

## Overview

| Feature           | Description                                              |
|-------------------|----------------------------------------------------------|
| **User Input**    | City / destination name via CLI                          |
| **Weather Tool**  | Real-time weather from WeatherAPI.com                    |
| **Attractions**   | Top tourist spots via DuckDuckGo Search (ddgs)           |
| **Agent**         | LangChain `create_tool_calling_agent` + `AgentExecutor`  |
| **LLM Providers** | OpenAI GPT-4o-mini / Google Gemini / Ollama (llama3.1)   |
| **Output**        | Unified natural-language response                        |

---

## Architecture

```
User (CLI input)
       |
       v
  main.py  --->  agent.py (build_agent)
                      |
              LangChain Tool-Calling Agent
                 /                \
        Weather Tool         Attractions Tool
        (weather.py)         (attractions.py)
             |                      |
       WeatherAPI.com        DuckDuckGo Search
             |                      |
             v                      v
          LLM synthesises final response
                      |
                      v
       Formatted answer displayed to user
```

### Key Components

| Component                       | Role                                                         |
|---------------------------------|--------------------------------------------------------------|
| **LLM**                        | Reasoning engine -- understands intent, picks tools, merges output |
| **@tool functions**             | Encapsulate external API calls as callable tools             |
| **create_tool_calling_agent()** | Binds LLM + tools + system prompt into an agent              |
| **AgentExecutor**               | Runs the agent plan, manages tool invocation and error handling |
| **config.py**                   | Central config -- provider selection, API keys from `.env`   |

---

## How the LLM Reasoning Works

### What is "Reasoning" in This Context?

The LLM (Large Language Model) does **not** follow hard-coded `if/else` logic.
Instead, it acts as an **autonomous planner** that dynamically decides:

- **What** information is needed to answer the user
- **Which tools** to call (and in what order)
- **How to combine** the tool outputs into a coherent answer

This is called **tool-based reasoning** -- the intelligence comes from the LLM's
ability to read a query, break it down into sub-tasks, and delegate each
sub-task to the right tool.

### Step-by-Step Reasoning Process

When a user types a city name (e.g. "Paris"), here is exactly what happens
inside the agent:

#### Step 1 -- Intent Understanding

The LLM receives the user's message via the **system prompt** and the
**human message template** defined in `agent.py`:

- **System prompt** tells the LLM: "You are an Intelligent Travel Assistant AI.
  Your job is to help travellers by providing current weather and top tourist
  attractions. Always call the weather tool and the attractions search tool..."
- **Human message** is the wrapped user input: "I'm planning a trip to Paris.
  Please give me the current weather and the top tourist attractions."

The LLM reads both messages and **understands the intent**: the user wants
weather data and attraction recommendations for Paris.

#### Step 2 -- Tool Selection (Planning)

The LLM knows about the available tools because they are **bound to the agent**
at creation time via `create_tool_calling_agent(llm, tools, prompt)`. Each tool
has a name, a description, and a typed input schema (provided by the `@tool`
decorator):

| Tool Name          | Description (from docstring)                     | Input         |
|--------------------|--------------------------------------------------|---------------|
| `get_weather`      | Fetch current weather for a city                 | `city: str`   |
| `get_attractions`  | Search for top tourist attractions in a city     | `city: str`   |

The LLM **reasons about the query** and decides it needs to invoke **both**
tools. It generates a structured **tool call request**:

```json
[
  { "tool": "get_weather",      "args": { "city": "Paris" } },
  { "tool": "get_attractions",  "args": { "city": "Paris" } }
]
```

**This is not hard-coded.** If the user asked "What's the weather in Tokyo?"
the LLM would reason that only the weather tool is needed. If they asked
"What should I see in Rome?" only the attractions tool would be called. The
LLM dynamically chooses which tools are relevant based on the query.

#### Step 3 -- Tool Execution

The **AgentExecutor** receives the LLM's tool call request and actually
runs each tool:

1. `get_weather("Paris")` -- calls the WeatherAPI.com HTTP endpoint and
   returns structured weather data (temperature, humidity, wind, etc.)
2. `get_attractions("Paris")` -- runs a DuckDuckGo search for
   "top tourist attractions in Paris" and returns a list of results.

The tool outputs are appended to the **agent scratchpad** (a message history
that the LLM can see in subsequent reasoning steps).

#### Step 4 -- Synthesis (Final Answer)

The LLM now sees:
- The original user query
- The weather tool output (e.g. "8.2C, Partly Cloudy, 87% humidity...")
- The attractions tool output (e.g. "1. Eiffel Tower, 2. Louvre Museum...")

It **synthesises** all of this into a single, well-formatted, conversational
response. This is the "merge" step -- the LLM writes a friendly travel
briefing that combines both data sources into one cohesive answer.

#### Step 5 -- Error Handling

If a tool fails (e.g. no internet, bad API key), the LLM **acknowledges the
gap** and still presents whatever data it successfully retrieved. This
graceful degradation is specified in the system prompt:
*"If a tool fails, acknowledge the issue and still present what you have."*

### Reasoning Flow Diagram

```
+------------------------------------+
|  User: "Tell me about              |
|  visiting Tokyo"                   |
+----------------+-------------------+
                 |
                 v
+------------------------------------+
|  LLM reads system prompt + query   |
|  Understands: needs weather + spots|
+----------------+-------------------+
                 |
                 v
+------------------------------------+
|  LLM generates tool call requests: |
|   - get_weather("Tokyo")           |
|   - get_attractions("Tokyo")       |
+--------+--------------+-----------+
         |              |
         v              v
  +-----------+   +-------------+
  | Weather   |   | Attractions |
  | API Call  |   | Search Call |
  +-----------+   +-------------+
         |              |
         v              v
   Weather JSON    Search Results
         |              |
         +------+-------+
                |
                v
+------------------------------------+
|  LLM merges both outputs into a    |
|  single coherent travel briefing   |
+------------------------------------+
                |
                v
+------------------------------------+
|  Final response shown to user      |
+------------------------------------+
```

### Why This is "Reasoning" and Not Hard-Coded Logic

In a traditional program you might write:

```python
# HARD-CODED approach (NOT what we do)
weather = get_weather(city)
attractions = get_attractions(city)
print(f"Weather: {weather}\nAttractions: {attractions}")
```

In our agent-based approach, the **LLM decides at runtime**:
- Whether to call one tool, both tools, or neither
- What arguments to pass to each tool
- What order to call them in
- How to format and merge the results

The developer only defines the tools and the system prompt. The reasoning
logic is entirely within the LLM. This makes the system **extensible** --
adding a new tool (e.g. hotel search) requires zero changes to the agent
logic; the LLM will discover and use the new tool automatically.

---

## Code Explanation and Program Flow

### File-by-File Walkthrough

#### 1. `config.py` -- Centralised Configuration

```python
load_dotenv()  # loads .env file

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # openai | gemini | ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
```

**Purpose:** Single source of truth for all configuration. Uses `python-dotenv`
to load API keys from a `.env` file so no secrets are hard-coded.

#### 2. `tools/weather.py` -- Weather Tool

```python
@tool
def get_weather(city: str) -> str:
    # Calls WeatherAPI.com, returns formatted weather string
    response = requests.get(WEATHER_API_BASE, params={"key": KEY, "q": city})
    data = response.json()
    return f"Temperature: {data['current']['temp_c']}C ..."
```

**How it works:**
- `@tool` decorator registers it as a LangChain tool with name, description, and input schema
- Makes an HTTP GET request to WeatherAPI.com with the city name
- Parses the JSON response and returns a formatted string
- If the API fails, returns a graceful error message instead of crashing

#### 3. `tools/attractions.py` -- Attractions Search Tool

```python
@tool
def get_attractions(city: str) -> str:
    # Uses DuckDuckGo to search for attractions (no API key needed)
    with DDGS() as ddgs:
        results = list(ddgs.text(f"top tourist attractions in {city}", max_results=5))
    return formatted_results
```

**How it works:**
- Also decorated with `@tool` so the LLM can discover and call it
- Uses DuckDuckGo Search (`ddgs` package) -- no API key required
- Formats each result with title, snippet, and URL

#### 4. `tools/__init__.py` -- Tool Registry

```python
from tools.weather import get_weather
from tools.attractions import get_attractions
ALL_TOOLS = [get_weather, get_attractions]
```

Single list of all tools. When you add a new tool, just import and append here.

#### 5. `agent.py` -- Agent Factory (Core Reasoning Setup)

This is the heart of the application. It wires together the LLM, tools, and
prompt into a reasoning agent.

**System Prompt:** Instructs the LLM on its role and behaviour. Contains the
"personality" and tool-usage instructions.

**`agent_scratchpad`:** A `MessagesPlaceholder` in the prompt template that
stores intermediate tool results. This is how the LLM "remembers" what tools
returned between reasoning steps.

**`_build_llm()` -- Multi-provider factory:**

```python
def _build_llm():
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY)
    if provider == "ollama":
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL)
```

Uses the factory pattern -- switch providers by changing one env variable.

**`build_agent()` -- Agent construction:**

```python
def build_agent(verbose=True):
    llm = _build_llm()

    # Bind LLM + tools + prompt into a reasoning agent
    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=PROMPT)

    # Wrap in executor for managed execution
    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,            # Print reasoning steps to stdout
        handle_parsing_errors=True, # Recover from bad LLM output
        max_iterations=5,          # Safety limit against infinite loops
    )
    return executor
```

- `create_tool_calling_agent` teaches the LLM about available tools
- `AgentExecutor` manages the think-act-observe loop

#### 6. `main.py` -- Entry Point (User Interface)

```python
def main():
    executor = build_agent(verbose=True)          # Build the agent

    while True:
        user_input = input("Enter destination: ")  # Get city name

        # Wrap into natural language for better LLM comprehension
        query = f"I'm planning a trip to {user_input}. ..."

        result = executor.invoke({"input": query}) # Run reasoning chain

        print(result["output"])                    # Show final answer
```

**Why we wrap the city name:** The user types just "Paris", but we wrap it
into a full sentence because the LLM reasons better with natural language
context. The wrapping is done in `main.py`, keeping the agent generic.

### End-to-End Program Flow

```
 1. User runs: python main.py

 2. main.py calls build_agent():
      - config.py loads .env (API keys, provider choice)
      - _build_llm() creates the LLM (OpenAI / Gemini / Ollama)
      - create_tool_calling_agent() binds LLM + tools + prompt
      - AgentExecutor wraps the agent with execution management

 3. User types: "Paris"

 4. main.py wraps it: "I'm planning a trip to Paris.
    Please give me the current weather and the top tourist attractions."

 5. executor.invoke({"input": query}) starts the reasoning chain:

      a. LLM receives system prompt + user message
      b. LLM reasons: "I need weather AND attractions for Paris"
      c. LLM outputs tool calls:
           get_weather("Paris")
           get_attractions("Paris")
      d. AgentExecutor runs both tools:
           - weather.py calls WeatherAPI.com -> returns weather data
           - attractions.py calls DuckDuckGo -> returns search results
      e. Tool outputs are added to agent_scratchpad
      f. LLM sees scratchpad with both results
      g. LLM generates final natural-language answer
      h. AgentExecutor detects "final answer" and returns it

 6. main.py prints the formatted response

 7. Loop continues until user types "quit"
```

---

## Tool Design

### Weather Tool (`tools/weather.py`)

| Property  | Detail                                   |
|-----------|------------------------------------------|
| API       | [WeatherAPI.com](https://weatherapi.com) |
| Input     | `city: str`                              |
| Output    | Temperature, condition, humidity, wind   |
| Error     | Graceful messages for bad city / no key  |

### Attractions Tool (`tools/attractions.py`)

| Property  | Detail                                   |
|-----------|------------------------------------------|
| Engine    | DuckDuckGo Search (free, no key needed)  |
| Input     | `city: str`                              |
| Output    | Numbered list of attractions + links     |
| Fallback  | Error message with connectivity tip      |

---

## Setup and Run

### Prerequisites

- Python 3.10+
- One of: **OpenAI API key**, **Gemini API key**, or **Ollama** running locally
- A **WeatherAPI key** ([weatherapi.com](https://www.weatherapi.com/) -- free tier)

### Installation

```bash
cd "L3/Building AI Agents with LangChain"

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt

copy .env.example .env       # Windows
# Then edit .env and fill in your keys + set LLM_PROVIDER
```

### LLM Provider Configuration

| Provider   | Setting                  | Model Default      | API Key Required?         |
|------------|--------------------------|--------------------|-----------------------------|
| **OpenAI** | `LLM_PROVIDER=openai`   | `gpt-4o-mini`     | Yes (`OPENAI_API_KEY`)     |
| **Gemini** | `LLM_PROVIDER=gemini`   | `gemini-2.0-flash` | Yes (`GEMINI_API_KEY`)     |
| **Ollama** | `LLM_PROVIDER=ollama`   | `llama3.1`        | None (run `ollama pull llama3.1`) |

### Run

```bash
python main.py
```

---

## Sample Output

```
  Intelligent Travel Assistant AI
  Powered by LangChain Tool-Calling Agent

  LLM provider : Ollama  (llama3.1 @ http://localhost:11434)
  Agent ready. Enter a destination to begin.

  Enter destination (or 'quit'): Paris

  Agent is thinking...

> Entering new AgentExecutor chain...

Invoking: get_weather with {'city': 'Paris'}

  Weather for Paris, France:
    Temperature : 8.2C / 46.8F
    Condition   : Partly Cloudy
    Humidity    : 87%
    Wind        : 15.1 km/h S

Invoking: get_attractions with {'city': 'Paris'}

  Top attractions in Paris:
    1. Eiffel Tower
    2. Louvre Museum
    3. Notre-Dame Cathedral
    4. Arc de Triomphe
    5. Montmartre

> Finished chain.

  Paris Travel Briefing

  Current Weather:
  - Temperature: 8.2C, Partly Cloudy
  - Humidity: 87%, Wind: 15.1 km/h

  Top Tourist Attractions:
  1. Eiffel Tower - Iconic landmark
  2. Louvre Museum - World's largest art museum
  3. Notre-Dame Cathedral - Gothic architecture
  4. Arc de Triomphe - Monumental arch
  5. Montmartre - Historic artistic neighborhood
```

*(Actual output depends on live API data at the time of execution.)*

---

## Project Structure

```
Building AI Agents with LangChain/
|-- .env.example        # API key template (copy to .env)
|-- config.py           # Centralised settings (loads .env)
|-- agent.py            # LangChain agent + executor factory
|-- main.py             # CLI entry point (interactive loop)
|-- requirements.txt    # Python dependencies
|-- README.md           # This report
+-- tools/
    |-- __init__.py     # Exports ALL_TOOLS list
    |-- weather.py      # @tool -- WeatherAPI integration
    +-- attractions.py  # @tool -- DuckDuckGo search
```

---

## Technology Stack

| Layer       | Technology                             |
|-------------|----------------------------------------|
| Language    | Python 3.10+                           |
| Framework   | LangChain >= 1.2                       |
| LLM         | OpenAI GPT-4o-mini / Gemini / Ollama   |
| Weather API | WeatherAPI.com                         |
| Search      | DuckDuckGo (via ddgs package)          |
| Execution   | AgentExecutor                          |
| Config      | python-dotenv                          |

---

## MVP Success Criteria

| # | Criterion                                | Status |
|---|------------------------------------------|--------|
| 1 | User gets correct weather                | Done   |
| 2 | User gets relevant attractions           | Done   |
| 3 | Agent correctly calls tools              | Done   |
| 4 | Clean, readable response                 | Done   |
| 5 | No hard-coded logic                      | Done   |
| 6 | Extensible -- add tools without refactor | Done   |
| 7 | API keys via environment variables       | Done   |
| 8 | Graceful error handling                  | Done   |
| 9 | Multi-LLM provider support               | Done   |

---

## Extending the Agent

To add a new tool (e.g., hotel search, flight prices):

1. Create `tools/hotels.py` with a `@tool` function
2. Import it in `tools/__init__.py` and add to `ALL_TOOLS`
3. Done -- the agent will automatically discover and use the new tool

No changes to `agent.py` or `main.py` required. The LLM will see the new
tool's name and description, and will call it whenever the user's query
is relevant. This is the power of the **tool-calling agent architecture**.

---

---

**Author:** AHILL S

*Built as part of the Pinnacle Projects -- Level 3: Building AI Agents with LangChain.*
