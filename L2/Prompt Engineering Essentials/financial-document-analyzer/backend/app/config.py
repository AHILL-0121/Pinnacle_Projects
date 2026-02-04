"""
Application Configuration
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Financial Document Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    TEMP_DIR: Path = Path("./temp")
    
    # OCR Settings
    TESSERACT_CMD: Optional[str] = None  # Auto-detect if None
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300
    
    # LLM Settings
    LLM_PROVIDER: str = "groq"  # Options: gemini, groq, ollama
    
    # Google Gemini Settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Groq Settings (default - fast & free)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Ollama (Local) Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    
    # General LLM Settings
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    
    # Vision Model Settings (for chart analysis)
    VISION_PROVIDER: str = "gemini"  # Options: gemini (vision requires capable model)
    VISION_MODEL: str = "gemini-1.5-flash"
    
    # Processing Settings
    MAX_PAGES: int = 50
    TABLE_DETECTION_CONFIDENCE: float = 0.7
    CHART_DETECTION_CONFIDENCE: float = 0.6
    
    # Summary Settings
    SUMMARY_MAX_LENGTH: int = 2000
    
    class Config:
        env_file = "../.env"  # Look for .env in parent (project root) directory
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env


# Global settings instance
settings = Settings()

# Ensure temp directory exists
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Role-specific prompt configurations
ROLE_PROMPTS = {
    "investor": {
        "focus": ["growth potential", "profitability", "dividend policy", "market position", "risk factors"],
        "metrics": ["Revenue Growth", "Net Profit Margin", "EPS", "ROE", "Dividend Yield", "P/E Ratio"],
        "tone": "decision-oriented, highlighting investment attractiveness and risks"
    },
    "analyst": {
        "focus": ["financial ratios", "trend analysis", "peer comparison", "valuation metrics", "operational efficiency"],
        "metrics": ["ROE", "ROA", "Debt-to-Equity", "Current Ratio", "Operating Margin", "Asset Turnover"],
        "tone": "technical, data-driven with detailed ratio analysis"
    },
    "auditor": {
        "focus": ["compliance indicators", "accounting anomalies", "internal controls", "regulatory adherence", "red flags"],
        "metrics": ["Asset Quality", "Provision Coverage", "Capital Adequacy", "CET1 Ratio", "NPL Ratio"],
        "tone": "scrutinizing, compliance-focused with attention to irregularities"
    },
    "executive": {
        "focus": ["strategic performance", "competitive position", "operational highlights", "key achievements", "future outlook"],
        "metrics": ["Revenue", "EBITDA", "Market Share", "Cost Efficiency", "Strategic Initiatives"],
        "tone": "high-level, strategic with actionable insights"
    }
}


# Financial entity patterns for extraction
FINANCIAL_PATTERNS = {
    "currency": r"[\$€£¥₹]\s*[\d,]+\.?\d*|\d+\.?\d*\s*(million|billion|mn|bn|m|b|crore|lakh)",
    "percentage": r"\d+\.?\d*\s*%",
    "ratio": r"\d+\.?\d*\s*[xX:]?\s*\d*\.?\d*",
    "date": r"\b(Q[1-4]\s*\d{4}|FY\s*\d{2,4}|\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    "growth": r"(increase|decrease|growth|decline|rise|fall|up|down)\s*(?:of\s*)?\d+\.?\d*\s*%?",
}


# Document section identifiers
DOCUMENT_SECTIONS = [
    "balance sheet",
    "income statement",
    "profit and loss",
    "cash flow statement",
    "statement of financial position",
    "notes to accounts",
    "auditor's report",
    "management discussion",
    "key highlights",
    "financial highlights",
    "quarterly results",
    "segment performance",
]


# Standard financial metrics mapping
METRIC_ALIASES = {
    "revenue": ["total revenue", "net revenue", "sales", "turnover", "income from operations"],
    "net_profit": ["net income", "profit after tax", "pat", "bottom line", "net earnings"],
    "gross_profit": ["gross income", "gross margin", "gross earnings"],
    "operating_profit": ["operating income", "ebit", "operating earnings"],
    "ebitda": ["earnings before interest tax depreciation amortization"],
    "eps": ["earnings per share", "basic eps", "diluted eps"],
    "total_assets": ["assets", "total asset base"],
    "total_liabilities": ["liabilities", "total debt", "borrowings"],
    "equity": ["shareholders equity", "net worth", "stockholders equity"],
    "roe": ["return on equity"],
    "roa": ["return on assets"],
    "npm": ["net profit margin", "profit margin"],
    "current_ratio": ["liquidity ratio"],
    "debt_equity": ["leverage ratio", "gearing ratio"],
}
