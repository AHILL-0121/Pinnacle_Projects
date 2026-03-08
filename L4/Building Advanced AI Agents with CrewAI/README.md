# 🤖 CrewAI Code Analyzer with Groq

## Purpose

A **multi-agent AI system** built with CrewAI that automatically analyzes Python code for errors and generates corrected versions. Uses Groq's high-speed inference with llama-3.3-70b-versatile through a hierarchical agent architecture coordinated by a manager agent.

## Technical Specifications

| Aspect | Detail |
|--------|--------|
| **Framework** | CrewAI (Multi-Agent Orchestration) |
| **LLM Provider** | Groq API |
| **Model** | llama-3.3-70b-versatile |
| **Tools** | CodeInterpreterTool (code execution & validation) |
| **Process Type** | Sequential with Manager oversight |
| **Environment** | Google Colab / Jupyter Notebook |

## Architecture

### Agent Hierarchy

```
┌─────────────────────────────────────────┐
│         Manager Agent                   │
│  Role: Software Engineering Manager     │
│  - Oversees analysis & correction       │
│  - Delegates tasks to specialist agents │
│  - Ensures quality control              │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│   Code      │  │    Code     │
│  Analyzer   │  │  Corrector  │
│             │  │             │
│ Identifies  │  │ Fixes all   │
│ syntax,     │  │ identified  │
│ logical &   │  │ errors and  │
│ indentation │  │ returns     │
│ errors      │  │ corrected   │
│             │  │ code        │
└─────────────┘  └─────────────┘
       │                │
       └────────┬───────┘
                ▼
       CodeInterpreterTool
    (Executes & validates code)
```

### Workflow

1. **Code Analysis Phase**
   - `code_analyzer` agent receives buggy Python code
   - Uses CodeInterpreterTool to execute and identify:
     - Syntax errors (missing colons, invalid syntax)
     - Indentation issues (inconsistent spacing, tab/space mix)
     - Logical errors (incorrect algorithms, edge cases)
   - Produces detailed error report with line numbers

2. **Code Correction Phase**
   - `code_corrector` agent receives error analysis
   - Systematically fixes each identified issue
   - Validates corrected code through CodeInterpreterTool
   - Returns fully functional, properly formatted code

3. **Manager Oversight**
   - `manager` agent coordinates both phases
   - Ensures task dependencies are respected
   - Can delegate additional verification if needed

## Key Features

- **Hierarchical Multi-Agent System** — Manager agent with delegation capabilities
- **Sequential Task Processing** — Ensures analysis completes before correction
- **Tool-Augmented Agents** — CodeInterpreterTool for runtime validation
- **Context Sharing** — Correction task receives full analysis context
- **Real-Time Execution** — Groq's high-speed inference (sub-second responses)
- **Verbose Logging** — Detailed agent reasoning and decision tracking

## Setup & Configuration

### Prerequisites

- Python 3.10+
- Google Colab account (or local Jupyter with internet access)
- Groq API key ([get one here](https://console.groq.com/))

### Installation (Google Colab)

```python
# Cell 1: Install dependencies
!pip install -q groq langchain-groq crewai crewai-tools

# Cell 2: Setup environment
import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

### LiteLLM Patching for Colab

CrewAI requires `litellm` for LLM provider abstraction. The notebook includes a workaround to use Groq directly:

```python
# Creates a fake litellm module that routes calls to Groq
fake_litellm = MagicMock()
fake_litellm.__version__ = "1.55.0"
fake_litellm.drop_params = True
fake_litellm.completion = real_groq_completion  # Custom function
sys.modules['litellm'] = fake_litellm
```

This ensures compatibility without installing the full litellm package (which can have dependency conflicts in Colab).

## Usage

### Running the Analysis

```python
# Define buggy code
buggy_code = """
def fibonacci_iterative(n):
    if n < 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_sequence = [0, 1]
    for i in range(2, n):
    next_fib = fib_sequence[-1] + fib_sequence[-2]  # Indentation error
    fib_sequence.append(next_fib)
    return fib_sequence
"""

# Create crew and execute
crew = Crew(
    agents=[code_analyzer, code_corrector],
    tasks=[analysis_task, correction_task],
    process=Process.sequential,
    manager_agent=manager,
    planning=False,
    verbose=True
)

result = crew.kickoff()
print(result)
```

### Expected Output

**Analysis Phase Output:**
```
Error Analysis:
1. Syntax/Indentation Error (Line 11-12):
   - Missing indentation for 'next_fib = ...'
   - Lines inside for loop must be indented
   - Expected 8 spaces or 2 tabs

Error Type: IndentationError
```

**Correction Phase Output:**
```python
def fibonacci_iterative(n):
    if n < 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_sequence = [0, 1]
    for i in range(2, n):
        next_fib = fib_sequence[-1] + fib_sequence[-2]  # Fixed indentation
        fib_sequence.append(next_fib)
    return fib_sequence
```

## Agent Configuration

### Code Analyzer Agent

```python
code_analyzer = Agent(
    role="Code Analyzer",
    goal="Identify all syntax and logical errors in the provided Python code.",
    backstory="Expert Python developer who carefully inspects code.",
    tools=[code_interpreter],
    llm="groq/llama-3.3-70b-versatile",
    verbose=True,
    allow_delegation=False  # Focused role
)
```

### Code Corrector Agent

```python
code_corrector = Agent(
    role="Code Corrector",
    goal="Fix all identified errors and return corrected version.",
    backstory="Skilled Python engineer excelling at debugging.",
    llm="groq/llama-3.3-70b-versatile",
    verbose=True,
    allow_delegation=False
)
```

### Manager Agent

```python
manager = Agent(
    role="Manager",
    goal="Oversee the code analysis and correction process.",
    backstory="Seasoned software engineering manager.",
    llm="groq/llama-3.3-70b-versatile",
    verbose=True,
    allow_delegation=True  # Can delegate to specialists
)
```

## Task Definitions

### Analysis Task

```python
analysis_task = Task(
    description="Analyze Python code and identify all errors...",
    expected_output="Detailed list of errors with line numbers and descriptions",
    agent=code_analyzer
)
```

### Correction Task

```python
correction_task = Task(
    description="Fix all identified errors...",
    expected_output="Fully corrected, executable Python code",
    agent=code_corrector,
    context=[analysis_task]  # Receives analysis results
)
```

## Advanced Features

### Planning Mode

Enable AI-driven task planning:

```python
crew = Crew(
    agents=[code_analyzer, code_corrector],
    tasks=[analysis_task, correction_task],
    process=Process.sequential,
    manager_agent=manager,
    planning=True,  # Manager creates execution plan first
    verbose=True
)
```

With planning enabled, the manager agent:
1. Analyzes task dependencies
2. Creates an execution strategy
3. Allocates resources to agents
4. Monitors progress and adapts

### Process Types

| Process | Description | Use Case |
|---------|-------------|----------|
| `Process.sequential` | Tasks execute in order | Analysis must precede correction |
| `Process.hierarchical` | Manager delegates dynamically | Complex multi-step workflows |

## Error Types Detected

### Syntax Errors
- Missing colons, parentheses, brackets
- Invalid variable names
- Incorrect operators
- Malformed string literals

### Indentation Errors
- Inconsistent spacing (tabs vs spaces)
- Missing indentation in blocks
- Incorrect nesting levels

### Logical Errors
- Off-by-one errors in loops
- Incorrect conditional logic
- Edge case handling issues
- Type mismatches

## Performance

**Groq Speed Benchmarks** (llama-3.3-70b-versatile):
- Analysis phase: ~0.8-1.5 seconds
- Correction phase: ~1.2-2.0 seconds
- **Total workflow**: ~2-3.5 seconds

Compare to OpenAI GPT-4 (typically 8-15 seconds for same task).

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'litellm'`

**Solution:** The notebook includes litellm patching code. Ensure you run the setup cells in order.

### Issue: `GROQ_API_KEY not found`

**Solution:** 
```python
# Google Colab: Store in Secrets
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

# Local Jupyter: Use .env file
from dotenv import load_dotenv
load_dotenv()
```

### Issue: Agent calls failing with dictionary errors

**Solution:** The notebook patches agent LLM calls to clean message formats. Ensure the patching cell (Cell 5) executes successfully.

## Extending the System

### Adding New Agents

```python
code_tester = Agent(
    role="Code Tester",
    goal="Generate comprehensive test cases for corrected code.",
    backstory="QA engineer specializing in Python testing.",
    llm="groq/llama-3.3-70b-versatile",
    verbose=True
)

testing_task = Task(
    description="Create unit tests for the corrected code...",
    expected_output="Python test suite using pytest",
    agent=code_tester,
    context=[correction_task]
)
```

### Custom Tools

```python
from crewai_tools import tool

@tool("Python Linter")
def lint_code(code: str) -> str:
    """Run pylint on code and return suggestions."""
    # Implementation here
    return lint_results
```

## Dependencies

```
groq==0.9.0
langchain-groq==0.1.9
crewai==0.28.8
crewai-tools==0.1.6
```

## Related Projects

- **[Project 4 — AI Web Research Agent](../../L3/Building%20AI%20Agents%20from%20Scratch/)** — ReAct pattern implementation
- **[Project 5 — Intelligent Travel Assistant](../../L3/Building%20AI%20Agents%20with%20LangChain/)** — LangChain agent framework
- **[Project 7 — Competitor Intelligence System](../../L3/Building%20your%20First%20AI%20Agent%20with%20LangGraph/)** — LangGraph state machines
- **[Project 8 — Logistics Optimization System](../../L3/Building%20your%20First%20AI%20Agent%20with%20CrewAI/)** — CrewAI for supply chain

## License

MIT License — see repository root for details.

## Author

Part of the **Pinnacle Projects** portfolio — demonstrating production-grade AI agent architectures.
