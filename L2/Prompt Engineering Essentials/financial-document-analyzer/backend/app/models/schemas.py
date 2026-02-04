"""
Pydantic Schemas for API Request/Response Models
"""
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime


class UserRole(str, Enum):
    """Supported user roles for context-aware summarization"""
    INVESTOR = "investor"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    EXECUTIVE = "executive"


class DocumentType(str, Enum):
    """Types of financial documents"""
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    INVESTOR_PRESENTATION = "investor_presentation"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    """Trend direction indicators"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


# ============== Request Models ==============

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    role: UserRole = Field(default=UserRole.INVESTOR, description="User role for context-aware analysis")
    document_type: Optional[DocumentType] = Field(default=None, description="Type of financial document")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to focus on")
    company_name: Optional[str] = Field(default=None, description="Company name if known")
    fiscal_period: Optional[str] = Field(default=None, description="Fiscal period (e.g., Q3 2024, FY2023)")


class URLUploadRequest(BaseModel):
    """Request model for URL-based document upload"""
    url: HttpUrl = Field(..., description="URL of the document to analyze")
    role: UserRole = Field(default=UserRole.INVESTOR)
    document_type: Optional[DocumentType] = None
    focus_areas: Optional[List[str]] = None


# ============== Extracted Data Models ==============

class ExtractedMetric(BaseModel):
    """A single extracted financial metric"""
    name: str = Field(..., description="Metric name")
    value: Any = Field(..., description="Extracted value")
    unit: Optional[str] = Field(default=None, description="Unit (e.g., USD, %, ratio)")
    period: Optional[str] = Field(default=None, description="Time period")
    change: Optional[float] = Field(default=None, description="Change from previous period")
    change_type: Optional[str] = Field(default=None, description="YoY, QoQ, etc.")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Extraction confidence")


class ExtractedTable(BaseModel):
    """Extracted table data"""
    title: Optional[str] = Field(default=None, description="Table title/header")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[Any]] = Field(default_factory=list, description="Table rows")
    page_number: Optional[int] = Field(default=None, description="Page where table was found")
    table_type: Optional[str] = Field(default=None, description="Type of financial table")


class ChartInsight(BaseModel):
    """Insights extracted from a chart or graph"""
    chart_type: str = Field(..., description="Type of chart (bar, line, pie, etc.)")
    title: Optional[str] = Field(default=None, description="Chart title")
    trend: TrendDirection = Field(..., description="Overall trend direction")
    key_values: Dict[str, Any] = Field(default_factory=dict, description="Key data points")
    insight: str = Field(..., description="Textual interpretation of the chart")
    page_number: Optional[int] = None
    
    @field_validator('trend', mode='before')
    @classmethod
    def normalize_trend(cls, v):
        """Normalize trend to lowercase for enum matching"""
        if isinstance(v, str):
            return v.lower()
        return v


class DocumentSection(BaseModel):
    """Identified document section"""
    name: str = Field(..., description="Section name")
    start_page: int = Field(..., description="Starting page")
    end_page: Optional[int] = Field(default=None, description="Ending page")
    content_summary: Optional[str] = Field(default=None, description="Brief summary of section content")


# ============== Analysis Results ==============

class FinancialMetrics(BaseModel):
    """Comprehensive financial metrics extracted"""
    revenue: Optional[ExtractedMetric] = None
    net_profit: Optional[ExtractedMetric] = None
    gross_profit: Optional[ExtractedMetric] = None
    operating_profit: Optional[ExtractedMetric] = None
    ebitda: Optional[ExtractedMetric] = None
    eps: Optional[ExtractedMetric] = None
    total_assets: Optional[ExtractedMetric] = None
    total_liabilities: Optional[ExtractedMetric] = None
    equity: Optional[ExtractedMetric] = None
    roe: Optional[ExtractedMetric] = None
    roa: Optional[ExtractedMetric] = None
    current_ratio: Optional[ExtractedMetric] = None
    debt_equity_ratio: Optional[ExtractedMetric] = None
    other_metrics: List[ExtractedMetric] = Field(default_factory=list)


class RiskItem(BaseModel):
    """Identified risk item"""
    category: str = Field(..., description="Risk category")
    description: str = Field(..., description="Risk description")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Risk severity")
    mitigation: Optional[str] = Field(default=None, description="Suggested mitigation")
    
    @field_validator('severity', mode='before')
    @classmethod
    def normalize_severity(cls, v):
        """Normalize severity to lowercase"""
        if isinstance(v, str):
            return v.lower()
        return v


class ActionableInsight(BaseModel):
    """An actionable insight from the analysis"""
    category: str = Field(..., description="Insight category")
    insight: str = Field(..., description="The insight")
    action: Optional[str] = Field(default=None, description="Recommended action")
    priority: Literal["low", "medium", "high"] = Field(default="medium")
    supporting_data: Optional[str] = Field(default=None, description="Data supporting this insight")
    
    @field_validator('priority', mode='before')
    @classmethod
    def normalize_priority(cls, v):
        """Normalize priority to lowercase"""
        if isinstance(v, str):
            return v.lower()
        return v


# ============== Summary Output ==============

class ExecutiveSummary(BaseModel):
    """Executive summary section"""
    overview: str = Field(..., description="High-level overview")
    key_highlights: List[str] = Field(default_factory=list, description="Key highlights")
    period_covered: Optional[str] = None
    company_name: Optional[str] = None


class PerformanceAnalysis(BaseModel):
    """Performance analysis section"""
    revenue_analysis: Optional[str] = None
    profitability_analysis: Optional[str] = None
    operational_efficiency: Optional[str] = None
    segment_performance: Optional[str] = None
    yoy_comparison: Optional[str] = None
    qoq_comparison: Optional[str] = None


class StructuredSummary(BaseModel):
    """Complete structured financial summary output"""
    executive_summary: ExecutiveSummary
    key_financial_metrics: FinancialMetrics
    performance_analysis: PerformanceAnalysis
    risks_and_outlook: List[RiskItem] = Field(default_factory=list)
    actionable_insights: List[ActionableInsight] = Field(default_factory=list)
    chart_insights: List[ChartInsight] = Field(default_factory=list)
    extracted_tables: List[ExtractedTable] = Field(default_factory=list)
    document_sections: List[DocumentSection] = Field(default_factory=list)
    
    # Metadata
    analysis_role: UserRole
    document_type: DocumentType
    processing_time_seconds: float
    confidence_score: float = Field(ge=0, le=1)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ============== API Response Models ==============

class ProcessingStatus(BaseModel):
    """Status of document processing"""
    status: Literal["pending", "processing", "completed", "completed_with_errors", "failed"]
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    current_step: Optional[str] = None
    error_message: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    success: bool
    message: str
    data: Optional[StructuredSummary] = None
    processing_status: ProcessingStatus
    raw_text: Optional[str] = Field(default=None, description="Raw extracted text (if requested)")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, bool]
