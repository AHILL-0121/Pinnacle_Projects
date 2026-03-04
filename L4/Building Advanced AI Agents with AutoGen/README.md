# 🤖 Building Advanced AI Agents with AutoGen

A comprehensive collection of **four production-grade multi-agent systems** built with **Microsoft AutoGen**, demonstrating industry patterns for autonomous agent coordination.

## 🎯 Collection Overview

This folder contains **four independently deployable projects** showcasing different AutoGen agent patterns:

| Project | Pattern | Domain | Key Technologies |
|---------|---------|--------|------------------|
| [Bill Managing Agent](Bill%20Managing%20Agent/) | Group Chat | Financial Automation | Vision API, Multi-Agent Coordination |
| [Financial Portfolio Manager](Financial%20Portfolio%20Manager/) | Stateful FSM | Investment Advisory | State Machines, Risk Analysis |
| [Smart Content Creation](Smart%20Content%20Creation/) | Reflection | Content Generation | Creator-Critic Loop, Quality Scoring |
| [Smart Health Assistant](Smart%20Health%20Assistant/) | Sequential | Healthcare | Function Calling, BMI Tools |

## 🏗️ AutoGen Agent Patterns

### 1. Group Chat Pattern
**Project**: Bill Managing Agent

**Architecture**:
```
Group Manager (orchestrator)
    ├── User Proxy
    ├── Bill Processing Agent (Vision)
    └── Expense Summarization Agent (Text)
```

**When to Use**:
- Multiple specialized agents need dynamic coordination
- Tasks require non-linear workflows
- Complex decision-making across agent teams

**Key Features**:
- Dynamic speaker selection
- Shared conversation context
- Centralized orchestration

---

### 2. Stateful Finite State Machine (FSM)
**Project**: Financial Portfolio Manager

**Architecture**:
```
INIT → PORTFOLIO_ANALYSIS → RECOMMENDATION → REPORT → COMPLETE
  ↓          ↓                    ↓             ↓          ↓
Agent1    Agent2              Agent3        Agent4    Terminate
```

**When to Use**:
- Workflows have clear sequential stages
- Each stage requires validation before proceeding
- Deterministic progression is required

**Key Features**:
- Explicit state transitions
- Validation at each stage
- Rollback capability

---

### 3. Reflection Pattern (Creator-Critic)
**Project**: Smart Content Creation

**Architecture**:
```
Content Creator  ⇄  Content Critic
    (Generate)      (Evaluate & Feedback)
         ↓               ↓
     Revise         Re-evaluate
         ↓               ↓
     APPROVED ✅
```

**When to Use**:
- Iterative improvement needed
- Quality assurance is critical
- Multiple revision cycles beneficial

**Key Features**:
- Bidirectional communication
- Structured feedback loops
- Quality threshold termination

---

### 4. Sequential Coordination with Function Calling
**Project**: Smart Health Assistant

**Architecture**:
```
User Proxy → BMI Agent (Tool) → Meal Planning Agent → Summary Agent
                 ↓
         calculate_bmi()
```

**When to Use**:
- Linear workflows with dependencies
- Tool/function calling required
- Clear handoff points between agents

**Key Features**:
- Function registration
- Sequential context passing
- Tool-augmented reasoning

---

## 🛠️ Shared Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | AutoGen | 0.11+ |
| **LLM Provider** | Groq | API-based |
| **Models** | llama-3.1-70b-versatile, llama-3.2-90b-vision-preview | Latest |
| **Vision** | Groq Vision API | Base64 encoding |
| **Python** | 3.10+ | Required |

## 📋 Prerequisites

All projects in this collection require:

- **Python 3.10+**
- **Groq API Key** ([Sign up here](https://console.groq.com))
- **AutoGen Framework**: `pip install autogen`

## 🚀 Quick Start (All Projects)

### 1. Install AutoGen

```bash
pip install autogen groq pillow requests
```

### 2. Set Up Groq API Key

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"
```

Or use Google Colab secrets:

```python
from google.colab import userdata
import os
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

### 3. Navigate to Project

```bash
cd "Bill Managing Agent"  # or any other project folder
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Or upload to **Google Colab** for cloud execution.

## 📊 Project Comparison

| Feature | Bill Management | Portfolio Manager | Content Creation | Health Assistant |
|---------|----------------|-------------------|------------------|------------------|
| **Agent Count** | 4 (Group Chat) | 4 (Sequential FSM) | 2 (Reflection) | 4 (Sequential) |
| **Coordination** | Dynamic | State-driven | Iterative | Linear |
| **Function Calling** | ❌ | ❌ | ❌ | ✅ |
| **Vision API** | ✅ | ❌ | ❌ | ❌ |
| **Termination** | Group decision | State: COMPLETE | Approval threshold | Keyword signal |
| **Complexity** | High | Medium | Low | Medium |
| **Use Case** | Expense tracking | Investment advisory | Content review | Health assessment |

## 🎓 Learning Path

We recommend exploring projects in this order:

### 1. **Smart Content Creation** (Beginner)
- Simplest: 2 agents, reflection pattern
- Learn: Bidirectional communication, termination conditions
- Time: 30 minutes

### 2. **Smart Health Assistant** (Beginner-Intermediate)
- Learn: Function calling, tool registration, sequential coordination
- Time: 45 minutes

### 3. **Financial Portfolio Manager** (Intermediate)
- Learn: State machines, FSM design, complex workflows
- Time: 1 hour

### 4. **Bill Managing Agent** (Advanced)
- Learn: Group chat, vision APIs, dynamic coordination
- Time: 1.5 hours

## 🔑 Key Concepts

### Agent Types

| Type | Description | Example Use |
|------|-------------|-------------|
| **ConversableAgent** | Bidirectional communication | Creator-Critic dialogue |
| **AssistantAgent** | LLM-powered reasoning | Analysis, summarization |
| **UserProxyAgent** | Human interaction or initiator | User input, workflow trigger |
| **GroupChat** | Multi-agent orchestration | Team coordination |

### Termination Strategies

| Strategy | Implementation | Project |
|----------|----------------|---------|
| **Keyword-based** | `is_termination_msg=lambda x: "TERMINATE" in x` | All projects |
| **Threshold-based** | `score >= 8.5` → APPROVED | Content Creation |
| **State-based** | `state == "COMPLETE"` | Portfolio Manager |
| **Max turns** | `max_consecutive_auto_reply=10` | All projects (fallback) |

### Best Practices

1. **Clear System Prompts**: Define roles, responsibilities, and output formats
2. **Explicit Termination**: Always include termination conditions
3. **Context Passing**: Use state/messages to share information between agents
4. **Error Handling**: Validate inputs and handle API failures gracefully
5. **Token Management**: Monitor token usage with `max_tokens` configuration
6. **Temperature Tuning**: 
   - `0.0-0.3` for factual tasks (health advice, analysis)
   - `0.7-0.9` for creative tasks (content generation)

## 🐛 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'autogen'`
```bash
pip install pyautogen  # Note: package name is 'pyautogen'
```

### Issue: `Invalid API Key`
- Verify key is set: `echo $env:GROQ_API_KEY` (PowerShell) or `echo $GROQ_API_KEY` (Bash)
- Check key validity on [Groq Console](https://console.groq.com)

### Issue: `Rate limit exceeded`
- Groq free tier: 30 requests/minute
- Solution: Add retry logic or upgrade plan

### Issue: `Vision API not extracting text`
- Ensure image is clear and high-resolution (300+ DPI)
- Check image format (JPG, PNG supported)
- Verify base64 encoding is correct

### Issue: `Agents not terminating`
- Check termination condition in `is_termination_msg`
- Set `max_consecutive_auto_reply` as fallback
- Ensure termination keyword is included in final message

## 📚 Additional Resources

### AutoGen Documentation
- [Official Docs](https://microsoft.github.io/autogen/)
- [API Reference](https://microsoft.github.io/autogen/docs/reference)
- [Conversation Patterns](https://microsoft.github.io/autogen/docs/tutorial/conversation-patterns)
- [Function Calling Guide](https://microsoft.github.io/autogen/docs/tutorial/tool-use)

### Groq Platform
- [API Documentation](https://console.groq.com/docs)
- [Model Benchmarks](https://wow.groq.com/groq-lpu-inference-engine/)
- [Pricing](https://console.groq.com/settings/limits)

### Multi-Agent Systems
- [AutoGen Research Paper](https://arxiv.org/abs/2308.08155)
- [Multi-Agent Coordination Patterns](https://www.microsoft.com/en-us/research/project/autogen/)

## 🔄 Extending the Projects

### Enhancement Ideas

**All Projects**:
- Add conversation history persistence (SQLite, JSON)
- Implement human-in-the-loop approval
- Add logging and monitoring
- Create REST API wrappers

**Bill Management**:
- Multi-currency support
- Budget tracking over time
- Receipt database with search

**Portfolio Manager**:
- Real-time stock price integration
- Tax optimization (Section 80C, LTCG)
- Portfolio rebalancing automation

**Content Creation**:
- Multi-critic system (style, SEO, fact-checking)
- Version control with diffs
- Export to multiple formats (MD, HTML, PDF)

**Health Assistant**:
- Exercise routine generator
- Progress tracking dashboard
- Integration with fitness APIs (Google Fit, Apple Health)

## 📊 Performance Benchmarks

| Project | Avg Execution Time | Token Usage | API Calls |
|---------|-------------------|-------------|-----------|
| Bill Management | 15-30s | 2,000-3,000 | 3-5 |
| Portfolio Manager | 20-40s | 3,000-5,000 | 4-6 |
| Content Creation | 30-60s (3-5 iterations) | 4,000-8,000 | 6-10 |
| Health Assistant | 10-20s | 1,500-2,500 | 3-4 |

*Note: Times vary based on LLM response speed and network latency*

## 📄 License

This collection is part of the Pinnacle Projects portfolio.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add new agent patterns (Planning, Tree-of-Thought)
- Implement multi-modal agents (audio, video)
- Add evaluation benchmarks
- Create visualization tools for agent communication
- Develop deployment guides (Docker, Cloud)

---

## 🎯 Next Steps

After mastering these AutoGen patterns, explore:
- **LangGraph**: Graph-based agent orchestration with cycles
- **CrewAI**: Role-playing multi-agent teams
- **Custom Frameworks**: Build your own agent coordination systems

---

**Part of the Pinnacle Projects - L4: Building Advanced AI Agents with AutoGen**
