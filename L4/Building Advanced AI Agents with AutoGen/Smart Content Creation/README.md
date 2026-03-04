# ✍️ Smart Content Creation with AutoGen

A reflection-based multi-agent system demonstrating the **Reflection Pattern** using **AutoGen** for iterative content improvement through creator-critic dialogue.

## 🎯 Overview

This project implements an intelligent content generation and refinement workflow where:
- A **Content Creator** agent drafts articles on technical topics
- A **Content Critic** agent evaluates and provides structured feedback
- Agents engage in iterative dialogue until content meets quality standards
- Termination occurs when the critic approves or max iterations reached

## 🏗️ Architecture

### Reflection Pattern Design

```
User Request
    ↓
Content Creator Agent
    ↓ (Draft v1)
Content Critic Agent
    ↓ (Feedback)
Content Creator Agent
    ↓ (Draft v2)
Content Critic Agent
    ↓ (Approval / More Feedback)
    ...
    ↓
Final Approved Content
```

### Agent Roles

| Agent | Responsibility | Evaluation Criteria |
|-------|----------------|---------------------|
| **Content Creator** | Drafts and revises technical content on Agentic AI | Creates structured articles with examples and use cases |
| **Content Critic** | Evaluates language clarity, technical accuracy, structure | Checks for jargon, factual correctness, flow, completeness |

### Feedback Dimensions

The critic evaluates content across multiple dimensions:

1. **Language Clarity** (1-10)
   - Simplicity, readability, grammar, structure
   
2. **Technical Accuracy** (1-10)
   - Correctness of concepts, up-to-date information, proper terminology
   
3. **Overall Score** (Average of above)
   - Threshold for approval: **8.5+**

## 🔑 Key Features

- **Iterative Refinement**: Content improves through multiple revision cycles
- **Structured Feedback**: Numeric scores + qualitative suggestions
- **Termination Condition**: Automatic approval when quality threshold met
- **Topic Focus**: Specialized in Agentic AI content generation
- **Conversation History**: Full dialogue preserved for analysis
- **Configurable Max Turns**: Prevents infinite loops (default: 5 turns)

## 🛠️ Technical Specifications

| Component | Technology |
|-----------|-----------|
| **Framework** | AutoGen 0.11+ |
| **LLM Provider** | Groq (llama-3.1-70b-versatile) |
| **Agent Type** | ConversableAgent (bidirectional communication) |
| **Pattern** | Reflection Pattern (Creator-Critic Loop) |
| **Termination Strategy** | Keyword-based (`APPROVED`) + max turns fallback |

## 📋 Prerequisites

- Python 3.10+
- Groq API Key ([Get it here](https://console.groq.com))
- Basic understanding of LLM agents

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install autogen groq
```

### 2. Set Up API Key

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"
```

Or use Google Colab secrets:

```python
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

### 3. Run the Notebook

Open `agentic_ai_autogen_reflection.ipynb` in Jupyter or Google Colab and run all cells.

### 4. Provide Content Topic

The notebook defaults to **"Agentic AI"**, but you can customize:

```python
initial_message = "Write a comprehensive article about [YOUR_TOPIC]"
```

## 📊 Example Workflow

### Iteration 1

**Creator (Draft v1)**:
```
# What is Agentic AI?

Agentic AI refers to autonomous systems that can plan, reason, 
and take actions to achieve specific goals...
```

**Critic (Feedback)**:
```
Language Clarity: 7/10
- Too much jargon in the introduction
- Add concrete examples for non-technical readers

Technical Accuracy: 8/10
- Missing mention of multi-agent systems
- Consider adding recent developments (2024)

Overall: 7.5/10 - REVISION NEEDED
```

### Iteration 2

**Creator (Draft v2)**:
```
# What is Agentic AI? (Simple Explanation)

Imagine a smart assistant that doesn't just answer questions, 
but can actually DO things for you. That's Agentic AI...

Example: A travel planning agent that:
- Searches for flights
- Compares prices
- Books tickets
- Sends you confirmations

Multi-Agent Systems:
Modern agentic AI often involves multiple specialized agents 
working together (like AutoGen, CrewAI, LangGraph)...
```

**Critic (Approval)**:
```
Language Clarity: 9/10
- Clear, accessible language
- Excellent use of concrete examples

Technical Accuracy: 9/10
- Accurate representation of current state
- Good coverage of multi-agent frameworks

Overall: 9/10 - APPROVED ✅
```

## 🔧 Configuration

### LLM Configuration

```python
llm_config = {
    "model": "llama-3.1-70b-versatile",
    "api_key": os.environ.get("GROQ_API_KEY"),
    "temperature": 0.7,  # Higher for creative content
    "max_tokens": 2000,
}
```

### Agent System Prompts

**Content Creator**:
```python
system_message = """You are an expert content creator specializing in Agentic AI.

Your responsibilities:
1. Draft well-structured articles with clear sections
2. Use accessible language for technical concepts
3. Provide concrete examples and use cases
4. Incorporate feedback from the critic iteratively
5. Revise until the critic is satisfied

When you receive feedback, revise the entire article addressing all points."""
```

**Content Critic**:
```python
system_message = """You are a meticulous content critic evaluating technical articles.

Evaluation Criteria:
1. Language Clarity (1-10): Simplicity, readability, grammar, structure
2. Technical Accuracy (1-10): Correctness, currency, proper terminology

Provide:
- Numeric scores for each dimension
- Specific actionable feedback
- Overall score (average)

If overall score >= 8.5, respond with: "APPROVED ✅"
Otherwise, request revisions."""
```

### Termination Conditions

```python
is_termination_msg = lambda x: "APPROVED" in x.get("content", "") or \
                                "TERMINATE" in x.get("content", "")
max_consecutive_auto_reply = 5  # Maximum iteration cycles
```

## 🎓 Reflection Pattern Benefits

### Why Reflection?

1. **Quality Assurance**: Systematic evaluation prevents low-quality output
2. **Iterative Improvement**: Content gets progressively better
3. **Multi-Perspective**: Creator focuses on generation, critic on evaluation
4. **Automated Refinement**: No manual intervention needed
5. **Scalable**: Works for any content type (articles, code, reports)

### Use Cases

- 📝 **Content Writing**: Blog posts, documentation, tutorials
- 💻 **Code Review**: AI-generated code reviewed by a critic agent
- 📊 **Report Generation**: Business reports with quality checks
- 🎓 **Educational Materials**: Lesson plans, study guides
- 📧 **Email Composition**: Professional emails with tone checking

## 🎓 Learning Outcomes

This project demonstrates:

1. **Reflection Pattern**: Implementation of the creator-critic paradigm
2. **Bidirectional Communication**: ConversableAgent dialogue management
3. **Termination Strategies**: Multiple exit conditions for robustness
4. **Structured Feedback**: Numeric scoring + qualitative analysis
5. **Iterative Refinement**: Progressive improvement through dialogue
6. **Agent Specialization**: Role-based system prompt engineering

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'autogen'`
- **Solution**: Install AutoGen: `pip install autogen`

**Issue**: `Invalid API Key`
- **Solution**: Verify your Groq API key is correctly set in environment variables

**Issue**: `Agents not converging (infinite loop)`
- **Solution**: Adjust `max_consecutive_auto_reply` or lower critic's approval threshold

**Issue**: `Content quality plateaus`
- **Solution**: Increase temperature for creator (0.8-0.9) or refine critic's feedback prompts

**Issue**: `Critic too lenient/strict`
- **Solution**: Adjust approval threshold (currently 8.5/10)

## 📚 Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Reflection Pattern in AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [AutoGen Conversation Patterns](https://microsoft.github.io/autogen/docs/tutorial/conversation-patterns)
- [Prompt Engineering for Critics](https://www.promptingguide.ai/)

## 🔄 Extending the Project

### Advanced Enhancements

1. **Multi-Critic System**: Add specialized critics for different aspects
   - Style Critic (tone, voice)
   - SEO Critic (keywords, readability)
   - Fact Checker (verifies claims)

2. **Weighted Scoring**: Different weights for different criteria
   ```python
   overall_score = (clarity * 0.4) + (accuracy * 0.4) + (engagement * 0.2)
   ```

3. **Versioning**: Track all draft versions with diffs

4. **Human-in-the-Loop**: Add manual approval step for high-stakes content

5. **Multi-Modal Feedback**: Add image generation for diagrams/charts

## 📄 License

This project is part of the Pinnacle Projects portfolio.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more evaluation dimensions (engagement, SEO, accessibility)
- Implement multi-critic systems
- Add support for different content types (code, poetry, legal)
- Create visualization of the refinement process

---

**Part of the Pinnacle Projects - L4: Building Advanced AI Agents with AutoGen**
