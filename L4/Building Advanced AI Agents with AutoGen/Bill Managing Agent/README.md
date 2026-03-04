# 📝 Bill Managing Agent

A multi-agent system using **AutoGen Group Chat** for automated bill processing and expense summarization using vision-enabled LLMs.

## 🎯 Overview

This project demonstrates a sophisticated multi-agent workflow where specialized agents collaborate to:
- Extract text from bill/receipt images using vision-enabled LLMs
- Categorize expenses into predefined categories
- Generate comprehensive expense summaries
- Coordinate workflow through a Group Manager

## 🏗️ Architecture

### Multi-Agent System Design

```
User Proxy Agent
    ↓
Group Manager (orchestrator)
    ↓
Bill Processing Agent → Expense Summarization Agent
    ↓                        ↓
Vision API               Text Analysis
(Extract Text)           (Categorize & Summarize)
```

### Agent Roles

| Agent | Responsibility | Technology |
|-------|---------------|------------|
| **User Proxy** | Initiates workflow, provides bill images | AutoGen ConversableAgent |
| **Group Manager** | Orchestrates agent coordination | AutoGen GroupChat |
| **Bill Processing Agent** | Extracts line items from bill images using GROQ Vision | Groq Vision API |
| **Expense Summarization Agent** | Categorizes expenses and generates summary | GROQ LLM |

### Expense Categories

- 🍕 **Food & Dining**: Restaurant meals, groceries, beverages
- 🚗 **Transportation**: Fuel, public transit, ride-sharing
- 🏠 **Utilities**: Electricity, water, internet, phone
- 🎉 **Entertainment**: Movies, events, subscriptions
- 🛒 **Shopping**: Clothing, electronics, household items
- 🏥 **Healthcare**: Medical expenses, pharmacy, insurance
- 📚 **Education**: Books, courses, tuition
- ✈️ **Travel**: Hotels, flights, vacation expenses

## 🔑 Key Features

- **Vision-Enabled Processing**: Uses Groq Vision API to extract text from bill images
- **Intelligent Categorization**: Automatically categorizes expenses into 8+ predefined categories
- **Multi-Agent Coordination**: Group chat pattern enables agents to collaborate autonomously
- **Base64 Image Encoding**: Handles image processing and API transmission
- **Structured Output**: Generates organized expense summaries with totals and categories

## 🛠️ Technical Specifications

| Component | Technology |
|-----------|-----------|
| **Framework** | AutoGen 0.11+ |
| **LLM Provider** | Groq (llama-3.2-90b-vision-preview, llama-3.1-70b-versatile) |
| **Image Processing** | Pillow (PIL) |
| **Vision API** | Groq Vision API with base64 image encoding |
| **Agent Pattern** | Group Chat with Sequential Coordination |

## 📋 Prerequisites

- Python 3.10+
- Groq API Key ([Get it here](https://console.groq.com))
- Bill/receipt images (JPG, PNG)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install autogen groq pillow requests
```

### 2. Set Up API Key

Set your Groq API key as an environment variable:

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

Open `bill_management_agent_autogen.ipynb` in Jupyter or Google Colab and run all cells.

### 4. Provide Bill Image

Update the image path in the notebook:

```python
bill_image_path = "/path/to/your/bill.jpg"
```

## 📊 Example Usage

### Input

A bill/receipt image containing:
- Restaurant meals
- Transportation costs
- Utility bills
- Entertainment expenses

### Output

```
📊 EXPENSE SUMMARY 📊

Total Amount: ₹3,450.00

Category Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🍕 Food & Dining:        ₹1,200.00 (34.8%)
🚗 Transportation:       ₹800.00   (23.2%)
🏠 Utilities:            ₹650.00   (18.8%)
🎉 Entertainment:        ₹500.00   (14.5%)
🛒 Shopping:             ₹300.00   (8.7%)

Top Categories:
1. Food & Dining
2. Transportation
3. Utilities
```

## 🔧 Configuration

### LLM Configuration

```python
llm_config = {
    "model": "llama-3.2-90b-vision-preview",  # Vision model for image processing
    "api_key": os.environ.get("GROQ_API_KEY"),
    "temperature": 0.0,
    "max_tokens": 2000,
}

text_llm_config = {
    "model": "llama-3.1-70b-versatile",  # Text model for summarization
    "api_key": os.environ.get("GROQ_API_KEY"),
    "temperature": 0.3,
}
```

### Termination Conditions

The workflow terminates when:
- Expense summary is generated
- Message contains "TERMINATE"
- Maximum turns reached (10)

## 🎓 Learning Outcomes

This project demonstrates:

1. **Multi-Agent Coordination**: How to orchestrate multiple specialized agents
2. **Vision-Language Models**: Using vision APIs for image understanding
3. **Group Chat Pattern**: AutoGen's group chat for agent collaboration
4. **Role-Based Design**: Separating concerns across specialized agents
5. **API Integration**: Working with Groq's vision and text APIs
6. **Base64 Encoding**: Handling image data for API transmission

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'autogen'`
- **Solution**: Install AutoGen: `pip install autogen`

**Issue**: `Invalid API Key`
- **Solution**: Verify your Groq API key is correctly set in environment variables

**Issue**: `Vision model not extracting text correctly`
- **Solution**: Ensure bill image is clear and high-resolution (recommended: 300 DPI)

**Issue**: `Group chat not terminating`
- **Solution**: Check termination condition in agent configuration

## 📚 Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Groq API Documentation](https://console.groq.com/docs)
- [AutoGen Group Chat Tutorial](https://microsoft.github.io/autogen/docs/tutorial/conversation-patterns)

## 📄 License

This project is part of the Pinnacle Projects portfolio.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add support for more expense categories
- Improve categorization accuracy
- Add multi-language support
- Implement expense tracking over time

---

**Part of the Pinnacle Projects - L4: Building Advanced AI Agents with AutoGen**
