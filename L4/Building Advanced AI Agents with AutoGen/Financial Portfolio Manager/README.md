# 💼 Financial Portfolio Manager

An intelligent multi-agent system using **AutoGen** for comprehensive financial portfolio analysis and personalized investment recommendations.

## 🎯 Overview

This project implements a sophisticated stateful multi-agent workflow that:
- Analyzes user financial profiles and current portfolios
- Evaluates risk tolerance and investment goals
- Generates personalized asset allocation recommendations
- Provides actionable financial insights based on Indian investment landscape

## 🏗️ Architecture

### State Machine Design

The system uses a **finite-state machine** to orchestrate the workflow:

```
INIT → PORTFOLIO_ANALYSIS → RECOMMENDATION → REPORT → COMPLETE
   ↓            ↓                  ↓           ↓          ↓
Collect      Analyze         Asset        Generate    Finalize
User Data    Portfolio    Allocation    Recommendations Output
```

### Agent Ecosystem

| Agent | Responsibility | State Transition |
|-------|---------------|------------------|
| **User Proxy** | Collects user data, initiates workflow | INIT → PORTFOLIO_ANALYSIS |
| **Portfolio Analyst** | Analyzes current holdings, calculates ratios | PORTFOLIO_ANALYSIS → RECOMMENDATION |
| **Financial Advisor** | Generates asset allocation strategy | RECOMMENDATION → REPORT |
| **Report Generator** | Creates comprehensive financial report | REPORT → COMPLETE |

### State Definitions

| State | Description | Output |
|-------|-------------|--------|
| `INIT` | Initialize workflow and collect user data | User profile with financial goals |
| `PORTFOLIO_ANALYSIS` | Evaluate current portfolio composition | Risk metrics, diversification score |
| `RECOMMENDATION` | Generate asset allocation strategy | Target allocation percentages |
| `REPORT` | Create comprehensive report | Markdown-formatted recommendations |
| `COMPLETE` | Workflow termination | Final report ready |

## 🔑 Key Features

- **Risk Profiling**: Automatic risk assessment based on age, income, and goals
- **Asset Allocation**: Personalized recommendations across 10+ asset classes
- **Indian Market Focus**: INR-denominated, India-specific investment vehicles
- **Goal-Based Planning**: Retirement, education, wealth creation strategies
- **Comprehensive Reporting**: Markdown reports with actionable insights
- **State-Driven Coordination**: Deterministic workflow progression

## 🛠️ Technical Specifications

| Component | Technology |
|-----------|-----------|
| **Framework** | AutoGen 0.11+ |
| **LLM Provider** | Groq (llama-3.1-70b-versatile) |
| **Agent Pattern** | Stateful Multi-Agent System with Sequential Coordination |
| **State Management** | Custom FSM with explicit state transitions |
| **Financial Domain** | Indian investment landscape (INR, NSE, BSE, Mutual Funds) |

## 📋 Prerequisites

- Python 3.10+
- Groq API Key ([Get it here](https://console.groq.com))
- Basic understanding of financial planning concepts

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

Open `financial_portfolio_manager.ipynb` in Jupyter or Google Colab and run all cells.

### 4. Configure User Profile

Update the user profile in the notebook:

```python
user_portfolio = {
    "name": "Your Name",
    "age": 30,
    "annual_salary": 1200000,  # INR
    "risk_tolerance": "moderate",  # conservative | moderate | aggressive
    "investment_horizon": "7-10 years",
    "current_portfolio": {
        "Emergency Fund": 300000,
        "Fixed Deposits": 500000,
        "Stocks (Direct Equity)": 200000,
        "Mutual Funds": 400000,
        "PPF": 150000,
        "Gold": 100000,
    },
    "financial_goals": ["Retirement", "Child Education", "Wealth Creation"]
}
```

## 📊 Example Output

### Input Portfolio

```
User: Rahul Sharma
Age: 30 years
Annual Salary: ₹12,00,000
Total Portfolio Value: ₹16,50,000
Risk Tolerance: Moderate
Investment Horizon: 7-10 years
```

### Analysis & Recommendations

```
📊 FINANCIAL PORTFOLIO ANALYSIS 📊

Current Allocation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixed Deposits:      ₹5,00,000 (30.3%)
Mutual Funds:        ₹4,00,000 (24.2%)
Emergency Fund:      ₹3,00,000 (18.2%)
Stocks:              ₹2,00,000 (12.1%)
PPF:                 ₹1,50,000 (9.1%)
Gold:                ₹1,00,000 (6.1%)

Risk Profile: MODERATE
Diversification Score: 7.5/10

📈 RECOMMENDED ALLOCATION

Target Asset Mix:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equity (Stocks + MF):    50% (₹8,25,000)
Debt (FD + PPF):         30% (₹4,95,000)
Gold:                    10% (₹1,65,000)
Emergency Fund:          10% (₹1,65,000)

Action Items:
1. Increase equity exposure by ₹2,25,000
2. Reduce FD allocation by ₹1,55,000
3. Maintain gold at current level
4. Top up emergency fund by ₹65,000

Expected Returns: 10-12% CAGR
Time to Goal: 8 years
```

## 🎓 Asset Classes Covered

### Equity Instruments
- 📈 **Direct Stocks**: NSE/BSE listed companies
- 📊 **Equity Mutual Funds**: Large-cap, mid-cap, small-cap, ELSS
- 🌍 **Index Funds**: Nifty 50, Sensex, Nifty Next 50

### Debt Instruments
- 🏦 **Fixed Deposits**: Bank FDs, Corporate FDs
- 📑 **PPF** (Public Provident Fund)
- 💰 **Debt Mutual Funds**: Liquid, short-term, long-term
- 🏛️ **Government Securities**: G-Secs, T-Bills

### Alternative Investments
- 🥇 **Gold**: Physical gold, Gold ETFs, Sovereign Gold Bonds
- 🏢 **REITs**: Real Estate Investment Trusts
- 💵 **Emergency Fund**: Liquid savings

## 🔧 Configuration

### LLM Configuration

```python
llm_config = {
    "model": "llama-3.1-70b-versatile",
    "api_key": os.environ.get("GROQ_API_KEY"),
    "temperature": 0.3,
    "max_tokens": 3000,
}
```

### Risk Tolerance Levels

| Level | Age Range | Equity Allocation | Investment Horizon |
|-------|-----------|-------------------|-------------------|
| **Conservative** | 50+ | 20-30% | 3-5 years |
| **Moderate** | 30-50 | 40-60% | 5-10 years |
| **Aggressive** | 18-30 | 60-80% | 10+ years |

### State Transition Logic

```python
state_descriptions = {
    "INIT": "Initializing portfolio analysis",
    "PORTFOLIO_ANALYSIS": "Analyzing user portfolio",
    "RECOMMENDATION": "Generating asset allocation recommendations",
    "REPORT": "Creating comprehensive financial report",
    "COMPLETE": "Analysis complete"
}
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Stateful Agent Systems**: Implementing FSM-based agent coordination
2. **Financial Domain Modeling**: Structuring investment analysis workflows
3. **Risk Assessment**: Algorithmic risk profiling and asset allocation
4. **Multi-Agent Collaboration**: Specialized agents working toward a common goal
5. **Deterministic Workflows**: State-based progression with validation
6. **Indian Financial Markets**: INR-denominated investment strategies

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'autogen'`
- **Solution**: Install AutoGen: `pip install autogen`

**Issue**: `Invalid API Key`
- **Solution**: Verify your Groq API key is correctly set in environment variables

**Issue**: `Recommendations not India-specific`
- **Solution**: Update agent system prompts to emphasize INR and Indian markets

**Issue**: `State not progressing`
- **Solution**: Check state transition logic and termination conditions

## 📚 Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Groq API Documentation](https://console.groq.com/docs)
- [SEBI Investor Education](https://www.investor.sebi.gov.in/)
- [Asset Allocation Strategies](https://www.investopedia.com/terms/a/assetallocation.asp)

## ⚠️ Disclaimer

This tool provides educational insights and should NOT be considered professional financial advice. Always consult a SEBI-registered financial advisor before making investment decisions.

## 📄 License

This project is part of the Pinnacle Projects portfolio.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more Indian investment instruments (NPS, SCSS, etc.)
- Implement tax optimization strategies (Section 80C, LTCG)
- Add historical performance analysis
- Create visualization dashboards

---

**Part of the Pinnacle Projects - L4: Building Advanced AI Agents with AutoGen**
