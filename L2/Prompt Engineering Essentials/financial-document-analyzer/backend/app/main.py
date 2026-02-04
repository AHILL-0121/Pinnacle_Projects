"""
FastAPI Application - Financial Document Analyzer API
"""
import io
import logging
import tempfile
from typing import Optional, List
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .config import settings
from .models.schemas import (
    UserRole, DocumentType, StructuredSummary,
    AnalysisResponse, ProcessingStatus, HealthCheckResponse,
    URLUploadRequest
)
from .services.summarizer import FinancialSummarizer
from .utils.validators import validate_file, validate_url, detect_file_type
from .utils.pdf_handler import PDFHandler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ollama_status(base_url: str) -> dict:
    """Check if Ollama is running and get available models"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "unknown") for m in models]
            return {"available": True, "models": model_names}
        return {"available": False, "models": [], "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"available": False, "models": [], "error": str(e)}


def print_startup_status():
    """Print startup status showing API key and service availability"""
    print("\n" + "="*60)
    print("🚀 FINANCIAL DOCUMENT ANALYZER - STARTUP STATUS")
    print("="*60)
    
    # LLM Provider Status
    print(f"\n📦 LLM PROVIDER: {settings.LLM_PROVIDER.upper()}")
    print("-"*40)
    
    if settings.LLM_PROVIDER == "gemini":
        if settings.GEMINI_API_KEY:
            masked_key = settings.GEMINI_API_KEY[:10] + "..." + settings.GEMINI_API_KEY[-4:]
            print(f"   ✅ Gemini API Key: {masked_key}")
        else:
            print("   ❌ Gemini API Key: NOT SET")
        print(f"   📋 Model: {settings.GEMINI_MODEL}")
        
    elif settings.LLM_PROVIDER == "groq":
        if settings.GROQ_API_KEY:
            masked_key = settings.GROQ_API_KEY[:10] + "..." + settings.GROQ_API_KEY[-4:]
            print(f"   ✅ Groq API Key: {masked_key}")
        else:
            print("   ❌ Groq API Key: NOT SET")
        print(f"   📋 Model: {settings.GROQ_MODEL}")
        
    elif settings.LLM_PROVIDER == "ollama":
        print(f"   🏠 Ollama URL: {settings.OLLAMA_BASE_URL}")
        ollama_status = check_ollama_status(settings.OLLAMA_BASE_URL)
        if ollama_status["available"]:
            print(f"   ✅ Ollama Status: RUNNING")
            print(f"   📋 Available Models: {', '.join(ollama_status['models'][:5])}")
            if settings.OLLAMA_MODEL in [m.split(':')[0] for m in ollama_status['models']]:
                print(f"   ✅ Target Model ({settings.OLLAMA_MODEL}): AVAILABLE")
            else:
                print(f"   ⚠️  Target Model ({settings.OLLAMA_MODEL}): NOT FOUND - Run: ollama pull {settings.OLLAMA_MODEL}")
        else:
            print(f"   ❌ Ollama Status: NOT RUNNING - {ollama_status.get('error', 'Unknown error')}")
            print(f"   💡 Start Ollama: Run 'ollama serve' in terminal")
    
    # Vision Provider Status
    print(f"\n👁️  VISION PROVIDER: {settings.VISION_PROVIDER.upper()}")
    print("-"*40)
    
    if settings.VISION_PROVIDER == "gemini":
        if settings.GEMINI_API_KEY:
            print(f"   ✅ Gemini Vision: CONFIGURED")
        else:
            print(f"   ❌ Gemini Vision: API KEY NOT SET")
        print(f"   📋 Model: {settings.VISION_MODEL}")
    
    # Summary
    print("\n" + "="*60)
    llm_ready = False
    vision_ready = False
    
    if settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        llm_ready = True
    elif settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        llm_ready = True
    elif settings.LLM_PROVIDER == "ollama":
        ollama_status = check_ollama_status(settings.OLLAMA_BASE_URL)
        llm_ready = ollama_status["available"]
    
    if settings.VISION_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        vision_ready = True
    
    if llm_ready and vision_ready:
        print("✅ ALL SERVICES READY")
    elif llm_ready:
        print("⚠️  LLM READY | VISION NOT CONFIGURED")
    else:
        print("❌ SERVICES NOT READY - Check configuration above")
    
    print("="*60 + "\n")


# Print startup status
print_startup_status()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    AI-powered Financial Document Analyzer API.
    
    Upload financial documents (scanned PDFs, images) and receive:
    - Structured financial summaries
    - Key metrics extraction
    - Chart and graph interpretation
    - Role-aware insights (Investor, Analyst, Auditor)
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services with multi-provider support
summarizer = FinancialSummarizer(
    ocr_language=settings.OCR_LANGUAGE,
    # LLM Provider settings
    llm_provider=settings.LLM_PROVIDER,
    gemini_api_key=settings.GEMINI_API_KEY,
    gemini_model=settings.GEMINI_MODEL,
    groq_api_key=settings.GROQ_API_KEY,
    groq_model=settings.GROQ_MODEL,
    ollama_base_url=settings.OLLAMA_BASE_URL,
    ollama_model=settings.OLLAMA_MODEL,
    # Vision settings
    vision_provider=settings.VISION_PROVIDER,
    vision_model=settings.VISION_MODEL,
    # Common LLM settings
    llm_temperature=settings.LLM_TEMPERATURE,
    llm_max_tokens=settings.LLM_MAX_TOKENS
)
pdf_handler = PDFHandler(dpi=settings.OCR_DPI, max_pages=settings.MAX_PAGES)


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """API root - health check"""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        services={
            "ocr": True,
            "llm": summarizer.llm_service.is_available(),
            "llm_provider": settings.LLM_PROVIDER,
            "vision": summarizer.chart_analyzer._vision_available(),
            "vision_provider": settings.VISION_PROVIDER
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        services={
            "ocr": True,
            "llm": summarizer.llm_service.is_available(),
            "llm_provider": settings.LLM_PROVIDER,
            "vision": summarizer.chart_analyzer._vision_available(),
            "vision_provider": settings.VISION_PROVIDER
        }
    )


# Store available providers info
AVAILABLE_PROVIDERS = {
    "gemini": {
        "name": "Google Gemini",
        "available": bool(settings.GEMINI_API_KEY),
        "model": settings.GEMINI_MODEL
    },
    "groq": {
        "name": "Groq (Llama)",
        "available": bool(settings.GROQ_API_KEY),
        "model": settings.GROQ_MODEL
    },
    "ollama": {
        "name": "Ollama (Local)",
        "available": False,  # Will check dynamically
        "model": settings.OLLAMA_MODEL
    }
}


def check_ollama_available() -> bool:
    """Check if Ollama is running"""
    try:
        import requests as req
        response = req.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


@app.get("/providers")
async def get_providers():
    """Get available LLM providers and their status"""
    providers = {}
    for key, info in AVAILABLE_PROVIDERS.items():
        providers[key] = {
            "name": info["name"],
            "model": info["model"],
            "available": info["available"] if key != "ollama" else check_ollama_available()
        }
    
    return {
        "current_provider": settings.LLM_PROVIDER,
        "providers": providers
    }


def create_summarizer_for_provider(provider: str) -> FinancialSummarizer:
    """Create a summarizer instance with the specified LLM provider."""
    return FinancialSummarizer(
        ocr_language=settings.OCR_LANGUAGE,
        llm_provider=provider,
        gemini_api_key=settings.GEMINI_API_KEY,
        gemini_model=settings.GEMINI_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        groq_model=settings.GROQ_MODEL,
        ollama_base_url=settings.OLLAMA_BASE_URL,
        ollama_model=settings.OLLAMA_MODEL,
        vision_provider=settings.VISION_PROVIDER,
        vision_model=settings.VISION_MODEL,
        llm_temperature=settings.LLM_TEMPERATURE,
        llm_max_tokens=settings.LLM_MAX_TOKENS
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    file: UploadFile = File(..., description="Financial document (PDF, image, or JSON)"),
    role: UserRole = Form(default=UserRole.INVESTOR, description="User role for analysis context"),
    document_type: Optional[str] = Form(default=None, description="Document type (auto-detected if not provided)"),
    company_name: Optional[str] = Form(default=None, description="Company name"),
    fiscal_period: Optional[str] = Form(default=None, description="Fiscal period (e.g., Q3 2024)"),
    focus_areas: Optional[str] = Form(default=None, description="Comma-separated focus areas"),
    llm_provider: Optional[str] = Form(default=None, description="LLM provider: groq, gemini, or ollama")
):
    """
    Analyze a financial document and generate a structured summary.
    
    **Supported formats:** PDF, PNG, JPG, JPEG, TIFF, BMP, JSON
    
    **JSON input:** Can be raw financial data (will be analyzed by AI) or 
    a pre-analyzed result (will be returned directly if it contains 'executive_summary').
    
    **LLM Providers:** groq (fast), gemini (Google), ollama (local)
    
    **User Roles:**
    - `investor`: Focus on growth, profitability, risks
    - `analyst`: Focus on ratios, trends, comparisons
    - `auditor`: Focus on compliance, anomalies, red flags
    - `executive`: Focus on strategic performance
    """
    try:
        # Select the appropriate summarizer based on provider
        if llm_provider and llm_provider.lower() in ['groq', 'gemini', 'ollama']:
            selected_provider = llm_provider.lower()
            logger.info(f"Using user-selected LLM provider: {selected_provider}")
            active_summarizer = create_summarizer_for_provider(selected_provider)
        else:
            logger.info(f"Using default LLM provider: {settings.LLM_PROVIDER}")
            active_summarizer = summarizer
        
        # Read file content
        content = await file.read()
        
        # Validate file
        is_valid, error = validate_file(
            file_bytes=content,
            file_name=file.filename
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Detect file type
        file_type = detect_file_type(content)
        
        # Handle JSON input - directly parse as pre-extracted data
        if file_type == 'json':
            logger.info(f"Processing JSON file: {file.filename}")
            import json
            try:
                json_data = json.loads(content.decode('utf-8'))
                
                # If it's already a complete analysis result, return it
                if isinstance(json_data, dict) and 'executive_summary' in json_data:
                    status = ProcessingStatus(
                        status="completed",
                        progress=100,
                        current_step="Loaded from JSON"
                    )
                    return AnalysisResponse(
                        success=True,
                        message="JSON data loaded successfully",
                        data=json_data,
                        processing_status=status
                    )
                
                # Otherwise, treat it as raw financial data to analyze
                # Convert JSON to text for LLM analysis
                json_text = json.dumps(json_data, indent=2)
                
                # Parse optional parameters
                doc_type = None
                if document_type:
                    try:
                        doc_type = DocumentType(document_type)
                    except ValueError:
                        pass
                
                focus_list = None
                if focus_areas:
                    focus_list = [f.strip() for f in focus_areas.split(',')]
                
                # Analyze JSON data using LLM
                summary, progress = active_summarizer.analyze_json_data(
                    json_data=json_data,
                    json_text=json_text,
                    role=role,
                    document_type=doc_type,
                    company_name=company_name,
                    fiscal_period=fiscal_period,
                    focus_areas=focus_list
                )
                
                status = ProcessingStatus(
                    status="completed" if not progress.errors else "completed_with_errors",
                    progress=100,
                    current_step="Complete",
                    error_message="; ".join(progress.errors) if progress.errors else None
                )
                
                return AnalysisResponse(
                    success=True,
                    message="JSON data analyzed successfully",
                    data=summary,
                    processing_status=status
                )
                
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
        
        # Convert to images for PDF/image files
        if file_type == 'pdf':
            logger.info(f"Processing PDF: {file.filename}")
            images = pdf_handler.pdf_to_images(content)
        else:
            logger.info(f"Processing image: {file.filename}")
            images = [Image.open(io.BytesIO(content))]
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not extract any pages from document")
        
        logger.info(f"Extracted {len(images)} pages for analysis")
        
        # Parse optional parameters
        doc_type = None
        if document_type:
            try:
                doc_type = DocumentType(document_type)
            except ValueError:
                pass
        
        focus_list = None
        if focus_areas:
            focus_list = [f.strip() for f in focus_areas.split(',')]
        
        # Analyze document (pass pdf_bytes for direct table extraction)
        summary, progress = active_summarizer.analyze_document(
            images=images,
            role=role,
            document_type=doc_type,
            company_name=company_name,
            fiscal_period=fiscal_period,
            focus_areas=focus_list,
            pdf_bytes=content if file_type == 'pdf' else None
        )
        
        # Build response
        status = ProcessingStatus(
            status="completed" if not progress.errors else "completed_with_errors",
            progress=100,
            current_step="Complete",
            error_message="; ".join(progress.errors) if progress.errors else None
        )
        
        return AnalysisResponse(
            success=True,
            message="Document analyzed successfully",
            data=summary,
            processing_status=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        return AnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            data=None,
            processing_status=ProcessingStatus(
                status="failed",
                progress=0,
                error_message=str(e)
            )
        )


@app.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_from_url(request: URLUploadRequest):
    """
    Analyze a financial document from a URL.
    
    Provide a URL to a PDF or image file for analysis.
    """
    try:
        # Validate URL
        is_valid, error = validate_url(str(request.url))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Download file
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.url), follow_redirects=True, timeout=60.0)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to download file: HTTP {response.status_code}"
                )
            
            content = response.content
        
        # Validate content
        is_valid, error = validate_file(file_bytes=content)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Detect and process
        file_type = detect_file_type(content)
        
        if file_type == 'pdf':
            images = pdf_handler.pdf_to_images(content)
        else:
            images = [Image.open(io.BytesIO(content))]
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not extract pages from document")
        
        # Analyze (pass pdf_bytes for direct table extraction)
        summary, progress = summarizer.analyze_document(
            images=images,
            role=request.role,
            document_type=request.document_type,
            focus_areas=request.focus_areas,
            pdf_bytes=content if file_type == 'pdf' else None
        )
        
        status = ProcessingStatus(
            status="completed" if not progress.errors else "completed_with_errors",
            progress=100,
            current_step="Complete",
            error_message="; ".join(progress.errors) if progress.errors else None
        )
        
        return AnalysisResponse(
            success=True,
            message="Document analyzed successfully",
            data=summary,
            processing_status=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"URL analysis failed: {e}")
        return AnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            data=None,
            processing_status=ProcessingStatus(
                status="failed",
                progress=0,
                error_message=str(e)
            )
        )


@app.get("/roles")
async def get_available_roles():
    """Get list of available user roles and their descriptions"""
    return {
        "roles": [
            {
                "id": "investor",
                "name": "Investor",
                "description": "Focus on growth potential, profitability, dividend policy, and investment risks",
                "key_metrics": ["Revenue Growth", "Net Profit Margin", "EPS", "ROE", "Dividend Yield"]
            },
            {
                "id": "analyst",
                "name": "Financial Analyst",
                "description": "Focus on financial ratios, trend analysis, and detailed metrics",
                "key_metrics": ["ROE", "ROA", "Debt-to-Equity", "Operating Margin", "Asset Turnover"]
            },
            {
                "id": "auditor",
                "name": "Auditor",
                "description": "Focus on compliance, accounting anomalies, and red flags",
                "key_metrics": ["Asset Quality", "Provision Coverage", "Capital Adequacy", "NPL Ratio"]
            },
            {
                "id": "executive",
                "name": "Executive",
                "description": "Focus on strategic performance and high-level insights",
                "key_metrics": ["Revenue", "EBITDA", "Market Share", "Cost Efficiency"]
            }
        ]
    }


@app.get("/document-types")
async def get_document_types():
    """Get list of supported document types"""
    return {
        "document_types": [
            {"id": "annual_report", "name": "Annual Report"},
            {"id": "quarterly_report", "name": "Quarterly Report"},
            {"id": "balance_sheet", "name": "Balance Sheet"},
            {"id": "income_statement", "name": "Income Statement"},
            {"id": "cash_flow", "name": "Cash Flow Statement"},
            {"id": "investor_presentation", "name": "Investor Presentation"},
            {"id": "unknown", "name": "Auto-detect"}
        ]
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "data": None
        }
    )
