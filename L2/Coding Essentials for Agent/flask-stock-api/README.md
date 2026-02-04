<div align="center">

# 📈 Flask Stock Intelligence API

**Enterprise-Grade RESTful API for Real-Time Stock Market Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![API](https://img.shields.io/badge/API-REST-blue?style=for-the-badge)](https://restfulapi.net)

*A production-ready, stateless REST API system for stock market data retrieval, processing, and quantitative analysis. Built with Flask, powered by Yahoo Finance, and designed for seamless integration with AI agents, trading systems, and financial dashboards.*

[Getting Started](#-quick-start) • [API Reference](#-api-specification) • [Architecture](#-system-architecture) • [Deployment](#-production-deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Specification](#-api-specification)
- [Error Handling](#-error-handling)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Production Deployment](#-production-deployment)
- [Performance Considerations](#-performance-considerations)
- [Use Cases](#-use-cases)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

The **Flask Stock Intelligence API** is a backend-only MVP designed to provide comprehensive stock market data services through a clean, well-documented RESTful interface. The system retrieves, processes, and analyzes financial data for any publicly traded company symbol, exposing four core endpoints that cover:

- **Company Metadata** — Corporate information, key officers, and business summaries
- **Real-Time Market Data** — Current prices, volume, and market state indicators  
- **Historical OHLCV Data** — Customizable date ranges with multiple interval options
- **Quantitative Analysis** — Volatility metrics, trend detection, drawdown calculations, and AI-generated insights

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Stateless Architecture** | No server-side session storage; each request is independent |
| **RESTful Conventions** | Proper HTTP methods, status codes, and resource-based URLs |
| **Input Validation** | Comprehensive sanitization and validation at all entry points |
| **Graceful Degradation** | Upstream failures return structured error responses |
| **Separation of Concerns** | Distinct layers for routing, business logic, and data access |

---

## ✨ Key Features

### Core Capabilities

```
✅ Company Information Retrieval      ✅ Real-Time Stock Quotes
✅ Historical OHLCV Data (POST)       ✅ Quantitative Analysis Engine
✅ Input Validation & Sanitization    ✅ Structured JSON Error Responses
✅ Market State Detection             ✅ Trend Classification Algorithm
✅ Volatility Calculations            ✅ Maximum Drawdown Analysis
```

### Out of Scope (Future Enhancements)

```
⏳ Authentication & Authorization     ⏳ User Account Management
⏳ Machine Learning Predictions       ⏳ Frontend Dashboard
⏳ Rate Limiting                      ⏳ Response Caching Layer
```

---

## 🏗 System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  (AI Agents, Trading Bots, Dashboards, Mobile Apps, CLI Tools)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/JSON
┌─────────────────────────────────────────────────────────────────┐
│                       FLASK API LAYER                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │  /company   │ │   /stock    │ │  /history   │ │ /analyze  │  │
│  │    (GET)    │ │    (GET)    │ │   (POST)    │ │  (POST)   │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                              │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   YahooFinanceService │    │      AnalysisService         │   │
│  │   ─────────────────── │    │   ────────────────────────   │   │
│  │   • get_company_info  │    │   • calculate_volatility     │   │
│  │   • get_stock_data    │    │   • detect_trend             │   │
│  │   • get_history       │    │   • calculate_max_drawdown   │   │
│  └──────────────────────┘    │   • generate_insights         │   │
│                               └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCE                         │
│                  Yahoo Finance (via yfinance)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Request Processing Pipeline

```
Request → Validation → Service Layer → Data Processing → JSON Response
    │          │             │               │               │
    │          │             │               │               └─ Serialization
    │          │             │               └─ NumPy/Pandas computations
    │          │             └─ Yahoo Finance API calls
    │          └─ Symbol, date, interval validation
    └─ HTTP method & content-type verification
```

---

## 🛠 Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Runtime** | Python | 3.10+ | Core programming language |
| **Framework** | Flask | 3.0+ | Lightweight WSGI web framework |
| **Data Source** | yfinance | 0.2.36+ | Yahoo Finance API wrapper |
| **Data Processing** | Pandas | 2.0+ | DataFrame operations & time series |
| **Numerical Computing** | NumPy | 1.24+ | Statistical calculations |
| **Production Server** | Gunicorn | 21.0+ | WSGI HTTP server |
| **Date Handling** | python-dateutil | 2.8+ | Flexible date parsing |

### Dependency Graph

```
flask-stock-api
├── flask (Web Framework)
│   ├── werkzeug (WSGI utilities)
│   ├── jinja2 (Templating - unused)
│   └── click (CLI utilities)
├── yfinance (Market Data)
│   ├── requests (HTTP client)
│   └── pandas (Data structures)
├── pandas (Data Analysis)
│   └── numpy (Numerical arrays)
└── gunicorn (Production Server)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection (for Yahoo Finance API)

### Installation

```bash
# Clone or navigate to the project directory
cd flask-stock-api

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (CMD)
.\venv\Scripts\activate.bat
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development mode
python run.py

# The API will be available at:
# http://localhost:5000
```

### Verify Installation

```bash
# Health check
curl http://localhost:5000/health

# Expected response:
# {"status": "healthy", "service": "Flask Stock Intelligence API"}
```

### Quick Test Commands

```powershell
# PowerShell Examples

# Health Check
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get

# Get Company Info
Invoke-RestMethod -Uri "http://localhost:5000/api/company/AAPL" -Method Get

# Get Real-Time Stock Data
Invoke-RestMethod -Uri "http://localhost:5000/api/stock/MSFT" -Method Get

# Get Historical Data
$body = @{symbol="AAPL"; start_date="2024-01-01"; end_date="2024-06-01"; interval="1d"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/history" -Method Post -Body $body -ContentType "application/json"

# Analyze Stock
Invoke-RestMethod -Uri "http://localhost:5000/api/analyze" -Method Post -Body $body -ContentType "application/json"
```

```bash
# cURL Examples

# Health Check
curl -X GET http://localhost:5000/health

# Get Company Info
curl -X GET http://localhost:5000/api/company/AAPL

# Get Real-Time Stock Data
curl -X GET http://localhost:5000/api/stock/GOOGL

# Get Historical Data
curl -X POST http://localhost:5000/api/history \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-06-01", "interval": "1d"}'

# Analyze Stock
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-06-01", "interval": "1d"}'
```

---

## 📖 API Specification

### Base URL

```
http://localhost:5000
```

### Endpoints Overview

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/health` | GET | Service health check | None |
| `/api/company/<symbol>` | GET | Company metadata | None |
| `/api/stock/<symbol>` | GET | Real-time market data | None |
| `/api/history` | POST | Historical OHLCV data | None |
| `/api/analyze` | POST | Quantitative analysis | None |

---

### 1. Health Check

Verify API availability and service status.

```http
GET /health
```

#### Response

```json
{
  "status": "healthy",
  "service": "Flask Stock Intelligence API"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Service operational |
| 500 | Internal server error |

---

### 2. Company Information

Retrieve comprehensive company metadata including business description, sector classification, and executive officers.

```http
GET /api/company/<symbol>
```

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol (e.g., AAPL, MSFT, GOOGL) |

#### Response Schema

```json
{
  "symbol": "string",
  "name": "string",
  "industry": "string",
  "sector": "string",
  "business_summary": "string",
  "key_officers": [
    {
      "name": "string",
      "title": "string"
    }
  ],
  "website": "string",
  "country": "string",
  "employees": "integer",
  "market_cap": "integer"
}
```

#### Example Response

```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "industry": "Consumer Electronics",
  "sector": "Technology",
  "business_summary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide...",
  "key_officers": [
    {"name": "Mr. Timothy D. Cook", "title": "CEO & Director"},
    {"name": "Mr. Kevan Parekh", "title": "Senior VP & CFO"}
  ],
  "website": "https://www.apple.com",
  "country": "United States",
  "employees": 150000,
  "market_cap": 3960797134848
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid symbol format |
| 404 | Symbol not found |
| 503 | Upstream service unavailable |

---

### 3. Real-Time Stock Data

Fetch current market data including price, volume, and market state.

```http
GET /api/stock/<symbol>
```

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |

#### Response Schema

```json
{
  "symbol": "string",
  "market_state": "string",
  "current_price": "number",
  "previous_close": "number",
  "change": "number",
  "change_percent": "number",
  "day_high": "number",
  "day_low": "number",
  "open": "number",
  "volume": "integer",
  "avg_volume": "integer",
  "fifty_two_week_high": "number",
  "fifty_two_week_low": "number",
  "timestamp": "string (ISO 8601)"
}
```

#### Market State Values

| State | Description |
|-------|-------------|
| `OPEN` | Regular trading hours |
| `CLOSED` | Market closed |
| `PRE_MARKET` | Pre-market trading session |
| `AFTER_HOURS` | After-hours trading session |
| `UNKNOWN` | State cannot be determined |

#### Example Response

```json
{
  "symbol": "AAPL",
  "market_state": "PRE_MARKET",
  "current_price": 269.48,
  "previous_close": 269.95,
  "change": -0.47,
  "change_percent": -0.17,
  "day_high": 271.88,
  "day_low": 267.61,
  "open": 269.13,
  "volume": 63714867,
  "avg_volume": 46942805,
  "fifty_two_week_high": 288.62,
  "fifty_two_week_low": 169.21,
  "timestamp": "2026-02-04T11:39:05.593599"
}
```

---

### 4. Historical Market Data

Retrieve historical OHLCV (Open, High, Low, Close, Volume) data for specified date ranges.

```http
POST /api/history
Content-Type: application/json
```

#### Request Body

```json
{
  "symbol": "string (required)",
  "start_date": "string (required, YYYY-MM-DD)",
  "end_date": "string (required, YYYY-MM-DD)",
  "interval": "string (optional, default: 1d)"
}
```

#### Request Parameters

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `symbol` | string | Yes | 1-10 alphanumeric chars | Stock ticker symbol |
| `start_date` | string | Yes | YYYY-MM-DD format | Range start date |
| `end_date` | string | Yes | YYYY-MM-DD format | Range end date |
| `interval` | string | No | `1d`, `1wk`, `1mo` | Data granularity |

#### Validation Rules

- `start_date` must be before `end_date`
- `start_date` cannot be in the future
- Maximum date range: 3,650 days (~10 years)

#### Response Schema

```json
{
  "symbol": "string",
  "interval": "string",
  "start_date": "string",
  "end_date": "string",
  "records_count": "integer",
  "data": [
    {
      "date": "string (YYYY-MM-DD)",
      "open": "number",
      "high": "number",
      "low": "number",
      "close": "number",
      "volume": "integer"
    }
  ]
}
```

#### Example Request

```json
{
  "symbol": "AAPL",
  "start_date": "2025-01-01",
  "end_date": "2025-06-01",
  "interval": "1d"
}
```

#### Example Response (Truncated)

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "start_date": "2025-01-01",
  "end_date": "2025-06-01",
  "records_count": 102,
  "data": [
    {
      "date": "2025-01-02",
      "open": 247.81,
      "high": 247.98,
      "low": 240.73,
      "close": 242.75,
      "volume": 55740700
    }
  ]
}
```

---

### 5. Analytical Insights

Perform quantitative analysis on historical data including volatility calculations, trend detection, and risk metrics.

```http
POST /api/analyze
Content-Type: application/json
```

#### Request Body

Same as `/api/history` endpoint.

#### Response Schema

```json
{
  "symbol": "string",
  "period": "string",
  "data_points": "integer",
  "average_price": "number",
  "volatility": "number",
  "trend": "string",
  "max_drawdown": "number",
  "return_percent": "number",
  "statistics": {
    "min_price": "number",
    "max_price": "number",
    "price_range": "number",
    "avg_volume": "integer",
    "start_price": "number",
    "end_price": "number"
  },
  "insight": "string"
}
```

#### Analysis Metrics Explained

| Metric | Formula | Description |
|--------|---------|-------------|
| **Average Price** | `Σ(close) / n` | Mean closing price over the period |
| **Volatility** | `σ(returns) × √252 × 100` | Annualized standard deviation of daily returns (%) |
| **Trend** | Linear regression + price comparison | Directional classification of price movement |
| **Max Drawdown** | `min((price - peak) / peak) × 100` | Largest peak-to-trough decline (%) |
| **Return Percent** | `((end - start) / start) × 100` | Total return over the period (%) |

#### Trend Classification

| Trend | Criteria |
|-------|----------|
| `bullish` | Return > +5% AND positive slope |
| `bearish` | Return < -5% AND negative slope |
| `mildly_bullish` | Return > 0% but ≤ +5% |
| `mildly_bearish` | Return < 0% but ≥ -5% |
| `sideways` | Return within ±5% |

#### Example Response

```json
{
  "symbol": "AAPL",
  "period": "2025-01-01 to 2025-06-01",
  "data_points": 102,
  "average_price": 218.91,
  "volatility": 43.94,
  "trend": "bearish",
  "max_drawdown": -30.22,
  "return_percent": -17.43,
  "statistics": {
    "min_price": 171.83,
    "max_price": 246.26,
    "price_range": 80.52,
    "avg_volume": 58634656,
    "start_price": 242.75,
    "end_price": 200.43
  },
  "insight": "The stock shows a significant downward trend with high volatility, indicating significant price swings. Note: Maximum drawdown of 30.2% suggests notable risk."
}
```

---

## ⚠️ Error Handling

### Standard Error Response Format

All errors return a consistent JSON structure:

```json
{
  "error": true,
  "message": "Human-readable error description",
  "status_code": 400
}
```

### HTTP Status Codes

| Code | Name | Description | Common Causes |
|------|------|-------------|---------------|
| 200 | OK | Request successful | - |
| 400 | Bad Request | Invalid input | Missing fields, invalid format, validation failure |
| 404 | Not Found | Resource not found | Invalid stock symbol, no data for date range |
| 500 | Internal Server Error | Server error | Unexpected exception |
| 503 | Service Unavailable | Upstream failure | Yahoo Finance API unreachable |

### Error Examples

#### Invalid Symbol Format (400)

```json
{
  "error": true,
  "message": "Invalid stock symbol 'INVALID!!!'. Symbol must be 1-10 alphanumeric characters.",
  "status_code": 400
}
```

#### Symbol Not Found (404)

```json
{
  "error": true,
  "message": "Stock symbol 'XXXX' not found",
  "status_code": 404
}
```

#### Invalid Date Range (400)

```json
{
  "error": true,
  "message": "start_date must be before end_date",
  "status_code": 400
}
```

#### Upstream Service Failure (503)

```json
{
  "error": true,
  "message": "Upstream data service unavailable",
  "status_code": 503
}
```

---

## 📁 Project Structure

```
flask-stock-api/
│
├── app/                          # Application package
│   ├── __init__.py               # Application factory & blueprint registration
│   │
│   ├── routes/                   # API endpoint definitions
│   │   ├── __init__.py           # Routes package exports
│   │   ├── company.py            # GET /api/company/<symbol>
│   │   ├── stock.py              # GET /api/stock/<symbol>
│   │   ├── history.py            # POST /api/history
│   │   └── analysis.py           # POST /api/analyze
│   │
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py           # Services package exports
│   │   ├── yahoo_service.py      # Yahoo Finance API integration
│   │   └── analysis_service.py   # Quantitative analysis computations
│   │
│   └── utils/                    # Shared utilities
│       ├── __init__.py           # Utils package exports
│       ├── validators.py         # Input validation functions
│       └── errors.py             # Custom exception classes & handlers
│
├── config.py                     # Configuration classes (Dev/Prod/Test)
├── run.py                        # Application entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

### Layer Responsibilities

| Layer | Responsibility |
|-------|----------------|
| **Routes** | HTTP request handling, response formatting |
| **Services** | Business logic, external API integration |
| **Utils** | Cross-cutting concerns (validation, errors) |
| **Config** | Environment-specific settings |

---

## ⚙️ Configuration

### Configuration Classes

```python
# config.py

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    VALID_INTERVALS = ['1d', '1wk', '1mo']
    MAX_DATE_RANGE_DAYS = 3650

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Environment mode (`development`, `production`, `testing`) |

### Setting Environment

```bash
# Linux/macOS
export FLASK_ENV=production

# Windows (PowerShell)
$env:FLASK_ENV = "production"

# Windows (CMD)
set FLASK_ENV=production
```

---

## 🚢 Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn (included in requirements.txt)
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:5000 run:app

# With access logging
gunicorn -w 4 -b 0.0.0.0:5000 --access-logfile - run:app

# With timeout configuration
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 run:app
```

### Gunicorn Configuration File

Create `gunicorn.conf.py`:

```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
errorlog = "-"
accesslog = "-"
loglevel = "info"
```

Run with config:

```bash
gunicorn -c gunicorn.conf.py run:app
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run:app"]
```

Build and run:

```bash
docker build -t flask-stock-api .
docker run -p 5000:5000 flask-stock-api
```

### Reverse Proxy (Nginx)

```nginx
# /etc/nginx/sites-available/flask-stock-api
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## ⚡ Performance Considerations

### Response Time Targets

| Endpoint | Target | Notes |
|----------|--------|-------|
| `/health` | < 10ms | No external calls |
| `/api/company` | < 2s | Yahoo Finance latency |
| `/api/stock` | < 1s | Yahoo Finance latency |
| `/api/history` | < 3s | Depends on date range |
| `/api/analyze` | < 4s | Includes computation time |

### Optimization Strategies

1. **Connection Pooling**: yfinance handles connection reuse internally
2. **Worker Scaling**: Increase Gunicorn workers for concurrent requests
3. **Caching Layer** (Future): Redis for frequently requested symbols
4. **Async Processing** (Future): Celery for heavy analysis tasks

### Concurrency

With Gunicorn's sync workers:

```
Concurrent Requests ≈ Workers × 1
```

Recommended workers: `(2 × CPU cores) + 1`

---

## 💼 Use Cases

### AI Agents & LLM Integration

```python
# Example: LangChain tool integration
import requests

def get_stock_analysis(symbol: str, start: str, end: str) -> dict:
    """Tool for AI agent to analyze stocks."""
    response = requests.post(
        "http://localhost:5000/api/analyze",
        json={"symbol": symbol, "start_date": start, "end_date": end}
    )
    return response.json()
```

### Trading Bots

```python
# Example: Automated trading signal
def check_buy_signal(symbol: str) -> bool:
    analysis = requests.post(
        "http://localhost:5000/api/analyze",
        json={"symbol": symbol, "start_date": "2024-01-01", "end_date": "2024-12-01"}
    ).json()
    
    return (
        analysis["trend"] == "bullish" and
        analysis["volatility"] < 30 and
        analysis["max_drawdown"] > -15
    )
```

### Financial Dashboards

```javascript
// Example: React dashboard integration
async function fetchStockData(symbol) {
    const response = await fetch(`http://localhost:5000/api/stock/${symbol}`);
    return response.json();
}
```

### Data Science & Research

```python
# Example: Pandas integration
import pandas as pd
import requests

def get_stock_dataframe(symbol, start, end):
    response = requests.post(
        "http://localhost:5000/api/history",
        json={"symbol": symbol, "start_date": start, "end_date": end}
    )
    data = response.json()["data"]
    return pd.DataFrame(data).set_index("date")
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for public functions
- Maintain test coverage

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 AHILL-0121

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Author

<div align="center">

**Developed & Maintained by**

### [AHILL-0121](https://github.com/AHILL-0121)

[![GitHub](https://img.shields.io/badge/GitHub-AHILL--0121-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AHILL-0121)

*Building intelligent systems for financial data analysis*

---

<sub>⭐ Star this repository if you found it helpful!</sub>

</div>
