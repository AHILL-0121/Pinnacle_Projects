# Sample Output Format

This document shows the expected output format when analyzing a financial document.

## Example: Quarterly Report Analysis (Investor Role)

### Input
- **Document**: Q3 2024 Quarterly Report
- **Role**: Investor
- **Company**: Sample Financial Corp

### Output Structure

```json
{
  "executive_summary": {
    "overview": "Sample Financial Corp delivered strong Q3 2024 results with revenue growing 15% YoY to $2.5B, driven by robust performance in the retail banking segment. Net profit increased 22% to $450M, with improved operating efficiency contributing to margin expansion. The bank maintains strong capital adequacy with CET1 ratio at 14.2%, well above regulatory requirements.",
    "key_highlights": [
      "Revenue grew 15% YoY to $2.5B, beating market expectations",
      "Net profit margin expanded 150bps to 18%",
      "EPS of $2.45 represents 20% growth YoY",
      "ROE improved to 18.5%, indicating efficient capital utilization",
      "Dividend of $0.75 per share declared, yield of 2.8%"
    ],
    "period_covered": "Q3 2024",
    "company_name": "Sample Financial Corp"
  },
  
  "key_financial_metrics": {
    "revenue": {
      "name": "Revenue",
      "value": 2500000000,
      "unit": "USD",
      "period": "Q3 2024",
      "change": 0.15,
      "change_type": "YoY",
      "confidence": 0.95
    },
    "net_profit": {
      "name": "Net Profit",
      "value": 450000000,
      "unit": "USD",
      "change": 0.22,
      "change_type": "YoY"
    },
    "eps": {
      "name": "EPS",
      "value": 2.45,
      "unit": "USD",
      "change": 0.20,
      "change_type": "YoY"
    },
    "roe": {
      "name": "ROE",
      "value": 18.5,
      "unit": "%"
    },
    "current_ratio": {
      "name": "Current Ratio",
      "value": 1.25,
      "unit": "ratio"
    }
  },
  
  "performance_analysis": {
    "revenue_analysis": "Revenue growth of 15% YoY was driven primarily by net interest income (+18%) benefiting from the higher rate environment, while fee income grew 8%. The retail banking segment contributed 60% of total revenue, showing particular strength in mortgage originations.",
    "profitability_analysis": "Net profit margin expanded to 18% from 16.5% in Q3 2023, driven by operating leverage and improved cost efficiency. The cost-to-income ratio improved to 52% from 55%, reflecting successful digital transformation initiatives.",
    "operational_efficiency": "Operating expenses grew only 8% YoY despite revenue growth of 15%, demonstrating strong cost discipline. Digital channels now account for 75% of transactions, up from 65% a year ago.",
    "yoy_comparison": "Strong improvement across all key metrics vs Q3 2023: Revenue +15%, Net Profit +22%, ROE +200bps",
    "qoq_comparison": "Sequential improvement with Revenue +3% QoQ and continued margin expansion"
  },
  
  "risks_and_outlook": [
    {
      "category": "Credit Risk",
      "description": "NPL ratio increased slightly to 1.8% from 1.5%, primarily in the commercial real estate segment. Provision coverage remains adequate at 125%.",
      "severity": "medium",
      "mitigation": "Management has tightened underwriting standards and increased provisioning for at-risk segments"
    },
    {
      "category": "Interest Rate Risk",
      "description": "Net interest margin may compress if rate cuts materialize faster than expected in 2025",
      "severity": "medium",
      "mitigation": "Diversifying revenue through fee-based services"
    },
    {
      "category": "Regulatory",
      "description": "New capital requirements may increase capital needs by $500M",
      "severity": "low",
      "mitigation": "Strong capital position provides buffer; CET1 at 14.2% vs 10.5% requirement"
    }
  ],
  
  "actionable_insights": [
    {
      "category": "Investment Thesis",
      "insight": "Strong earnings momentum and attractive valuation (P/E of 12x vs sector 15x) suggest upside potential",
      "action": "Consider accumulating on any pullback; current price offers good entry point",
      "priority": "high",
      "supporting_data": "20% EPS growth, 18.5% ROE, 2.8% dividend yield"
    },
    {
      "category": "Dividend",
      "insight": "Dividend payout ratio of 30% leaves room for future increases given strong earnings growth",
      "action": "Attractive for income-focused portfolios; expect 10-15% dividend growth",
      "priority": "medium"
    },
    {
      "category": "Risk Monitor",
      "insight": "Watch commercial real estate exposure given macroeconomic uncertainty",
      "action": "Monitor NPL trends in Q4 earnings; management guidance on provisions",
      "priority": "medium"
    }
  ],
  
  "chart_insights": [
    {
      "chart_type": "bar",
      "title": "Quarterly Revenue Trend",
      "trend": "up",
      "key_values": {
        "Q1 2024": 2300000000,
        "Q2 2024": 2430000000,
        "Q3 2024": 2500000000
      },
      "insight": "Revenue shows consistent upward trajectory with acceleration in Q3, indicating strong business momentum"
    },
    {
      "chart_type": "line",
      "title": "Net Interest Margin",
      "trend": "up",
      "key_values": {
        "Q3 2023": "3.2%",
        "Q3 2024": "3.6%"
      },
      "insight": "NIM expanded 40bps YoY, benefiting from higher interest rate environment"
    }
  ],
  
  "confidence_score": 0.87,
  "processing_time_seconds": 12.5,
  "analysis_role": "investor",
  "document_type": "quarterly_report"
}
```

## Output Sections Explained

### 1. Executive Summary
- **Overview**: 2-3 sentence high-level summary
- **Key Highlights**: 5 bullet points of most important findings
- **Period/Company**: Context information

### 2. Key Financial Metrics
- Extracted numerical values with units
- YoY/QoQ changes where detected
- Confidence scores

### 3. Performance Analysis
- Revenue analysis with segment breakdown
- Profitability trends and drivers
- Operational efficiency metrics
- Period comparisons

### 4. Risks & Outlook
- Categorized risks (Credit, Market, Regulatory, etc.)
- Severity levels (critical/high/medium/low)
- Mitigation factors

### 5. Actionable Insights
- Investment-relevant conclusions
- Specific recommended actions
- Priority ranking
- Supporting data

### 6. Chart Insights
- Type of chart detected
- Trend direction (↑ up, ↓ down, → stable)
- Key data points extracted
- Textual interpretation
