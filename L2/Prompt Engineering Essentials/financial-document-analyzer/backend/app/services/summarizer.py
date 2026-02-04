"""
Financial Summarizer Service
Orchestrates the complete document analysis pipeline and generates structured summaries
"""
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image

from .ocr_service import OCRService, OCRResult
from .table_extractor import TableExtractor, ExtractedTable
from .chart_analyzer import ChartAnalyzer, ChartInsight
from .entity_extractor import FinancialEntityExtractor, FinancialEntity, FinancialContext, DataValidationResult
from .llm_service import LLMService

from ..models.schemas import (
    StructuredSummary, ExecutiveSummary, FinancialMetrics,
    PerformanceAnalysis, RiskItem, ActionableInsight,
    ExtractedMetric, DocumentSection, UserRole, DocumentType,
    TrendDirection as SchemaTrend
)
from ..utils.image_processor import ImageProcessor
from ..utils.pdf_handler import PDFHandler

logger = logging.getLogger(__name__)


@dataclass
class ProcessingProgress:
    """Track processing progress"""
    total_steps: int = 10
    current_step: int = 0
    current_stage: str = ""
    errors: List[str] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> int:
        return int((self.current_step / self.total_steps) * 100)
    
    def advance(self, stage: str):
        self.current_step += 1
        self.current_stage = stage
        logger.info(f"[{self.progress_percent}%] {stage}")


class FinancialSummarizer:
    """
    Main orchestrator for financial document analysis.
    Combines OCR, table extraction, chart analysis, entity extraction,
    and LLM reasoning to produce structured financial summaries.
    """
    
    def __init__(
        self,
        ocr_language: str = "eng",
        # LLM Provider settings
        llm_provider: str = "groq",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-1.5-pro",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        # Vision settings (Gemini only)
        vision_provider: str = "gemini",
        vision_model: str = "gemini-1.5-pro",
        # Common LLM settings
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 4096
    ):
        # Initialize all services
        self.image_processor = ImageProcessor()
        self.pdf_handler = PDFHandler()
        self.ocr_service = OCRService(language=ocr_language)
        self.table_extractor = TableExtractor()
        self.chart_analyzer = ChartAnalyzer(
            vision_model=vision_model,
            vision_provider=vision_provider,
            gemini_api_key=gemini_api_key
        )
        self.entity_extractor = FinancialEntityExtractor()
        
        # Initialize LLM service with selected provider
        self.llm_service = LLMService(
            provider=llm_provider,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            groq_api_key=groq_api_key,
            groq_model=groq_model,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )
        
        logger.info(f"FinancialSummarizer initialized with LLM provider: {llm_provider}")
    
    def analyze_document(
        self,
        images: List[Image.Image],
        role: UserRole = UserRole.INVESTOR,
        document_type: Optional[DocumentType] = None,
        company_name: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        pdf_bytes: Optional[bytes] = None
    ) -> Tuple[StructuredSummary, ProcessingProgress]:
        """
        Analyze a financial document and produce a structured summary.
        
        NEW ARCHITECTURE (prevents hallucination cascade):
        PDF → pdfplumber tables + OCR → Deterministic Validation → Structured JSON → ONE LLM (report only)
        
        NO analysis LLM in between. Code decides if data exists, not LLM.
        
        Args:
            images: List of PIL Images (document pages)
            role: User role for context-aware summarization
            document_type: Type of financial document
            company_name: Company name if known
            fiscal_period: Fiscal period (e.g., "Q3 2024")
            focus_areas: Specific areas to focus on
            
        Returns:
            Tuple of (StructuredSummary, ProcessingProgress)
        """
        start_time = time.time()
        progress = ProcessingProgress()
        
        try:
            # Step 1: Preprocess images
            progress.advance("Preprocessing images")
            processed_images = self._preprocess_images(images)
            
            # Step 2: OCR extraction + pdfplumber table extraction
            progress.advance("Extracting text via OCR")
            ocr_results = self._extract_text(processed_images)
            full_text = self.ocr_service.get_full_document_text(ocr_results)
            
            # Step 2b: Try pdfplumber for direct table extraction (more reliable than OCR for tables)
            pdfplumber_text = ""
            if pdf_bytes:
                progress.advance("Extracting tables directly from PDF (pdfplumber)")
                try:
                    pdfplumber_text = self.pdf_handler.get_financial_table_text(pdf_bytes)
                    if pdfplumber_text:
                        logger.info("=" * 60)
                        logger.info("PDFPLUMBER TABLE EXTRACTION:")
                        logger.info("=" * 60)
                        logger.info(pdfplumber_text[:2000])
                        logger.info("=" * 60)
                        # Prepend table data to full_text so it gets priority
                        full_text = pdfplumber_text + "\n\n--- OCR TEXT ---\n" + full_text
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {e}")
            
            # DEBUG: Log raw OCR output for troubleshooting
            logger.info("=" * 60)
            logger.info("RAW OCR OUTPUT (first 3000 chars):")
            logger.info("=" * 60)
            logger.info(full_text[:3000])
            logger.info("=" * 60)
            
            # Step 3: DETERMINISTIC DATA VALIDATION (CODE decides, not LLM!)
            progress.advance("Validating financial data (deterministic)")
            validation_result = self.entity_extractor.validate_financial_data(full_text)
            
            logger.info(f"Data validation: has_data={validation_result.has_financial_data}, "
                       f"confidence={validation_result.confidence_score:.2f}")
            
            if not validation_result.has_financial_data:
                logger.warning("No financial data detected by deterministic validation")
                progress.errors.append("No financial data patterns detected in document")
            
            # Step 4: Build STRUCTURED JSON (Source of Truth)
            progress.advance("Building structured data (deterministic)")
            structured_data = self.entity_extractor.build_structured_json(full_text, validation_result)
            
            # Add company name if provided
            if company_name:
                structured_data['company'] = company_name
            
            # Step 5: Detect document sections (for additional context)
            progress.advance("Detecting document sections")
            sections = self._detect_sections(ocr_results)
            
            # Auto-detect document type if not provided
            if not document_type:
                document_type = self._detect_document_type(full_text, sections)
            
            # Step 6: Extract tables (for additional context)
            progress.advance("Extracting tables")
            tables = self._extract_tables(processed_images, ocr_results)
            tables_md = self.table_extractor.tables_to_markdown(tables)
            
            # Step 7: Analyze charts (optional)
            progress.advance("Analyzing charts and graphs")
            chart_insights = self._analyze_charts(images, full_text)
            
            # Step 8: SINGLE LLM CALL - Report generation only
            # NO analysis LLM! Structured JSON is the source of truth.
            progress.advance("Generating report from structured data")
            
            if validation_result.has_financial_data:
                # Pass structured JSON directly to report generator
                summary_response = self.llm_service.generate_report_from_structured_json(
                    structured_data=structured_data,
                    role=role.value,
                    document_type=document_type.value if document_type else "financial_report",
                    raw_text_excerpt=full_text[:3000]  # Only for additional context
                )
            else:
                # Fallback: still try to generate report
                progress.errors.append("Low confidence in extracted data")
                summary_response = self.llm_service.generate_report_from_structured_json(
                    structured_data=structured_data,
                    role=role.value,
                    document_type=document_type.value if document_type else "financial_report",
                    raw_text_excerpt=full_text[:3000]
                )
            
            # Step 9: Build final summary object
            progress.advance("Building final summary")
            
            # Also extract entities for backward compatibility
            context = FinancialContext(
                company_name=company_name,
                fiscal_period=fiscal_period,
                document_type=document_type.value if document_type else None
            )
            entities = self.entity_extractor.extract_entities(full_text, context)
            
            summary = self._build_summary(
                summary_response=summary_response,
                entities=entities,
                tables=tables,
                chart_insights=chart_insights,
                sections=sections,
                role=role,
                document_type=document_type or DocumentType.UNKNOWN,
                start_time=start_time
            )
            
            progress.advance("Complete")
            
            return summary, progress
            
        except Exception as e:
            logger.exception(f"Document analysis failed: {e}")
            progress.errors.append(str(e))
            
            # Return minimal summary on error
            return self._create_error_summary(
                str(e), role, document_type or DocumentType.UNKNOWN, start_time
            ), progress
    
    def analyze_json_data(
        self,
        json_data: Dict[str, Any],
        json_text: str,
        role: UserRole = UserRole.INVESTOR,
        document_type: Optional[DocumentType] = None,
        company_name: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Tuple[StructuredSummary, ProcessingProgress]:
        """
        Analyze financial data from JSON input.
        
        NEW ARCHITECTURE: JSON → Deterministic Validation → Structured JSON → ONE LLM
        
        Args:
            json_data: Parsed JSON data as dict/list
            json_text: JSON as formatted text string
            role: User role for context-aware summarization
            document_type: Type of financial document
            company_name: Company name if known
            fiscal_period: Fiscal period (e.g., "Q3 2024")
            focus_areas: Specific areas to focus on
            
        Returns:
            Tuple of (StructuredSummary, ProcessingProgress)
        """
        start_time = time.time()
        progress = ProcessingProgress(total_steps=5)
        
        try:
            # Step 1: Deterministic validation on JSON text
            progress.advance("Validating financial data from JSON")
            validation_result = self.entity_extractor.validate_financial_data(json_text)
            
            # Step 2: Build structured data
            progress.advance("Building structured data")
            structured_data = self.entity_extractor.build_structured_json(json_text, validation_result)
            
            if company_name:
                structured_data['company'] = company_name
            
            # Also extract entities for backward compatibility
            context = FinancialContext(
                company_name=company_name,
                fiscal_period=fiscal_period,
                document_type=document_type.value if document_type else None
            )
            entities = self.entity_extractor.extract_entities(json_text, context)
            
            # Auto-detect document type if not provided
            if not document_type:
                document_type = self._detect_document_type_from_json(json_data)
            
            # Step 3: SINGLE LLM CALL - Report generation only
            progress.advance("Generating report from structured data")
            summary_response = self.llm_service.generate_report_from_structured_json(
                structured_data=structured_data,
                role=role.value,
                document_type=document_type.value if document_type else "financial_report",
                raw_text_excerpt=json_text[:3000]
            )
            
            # Step 4: Build final summary object
            progress.advance("Building final summary")
            summary = self._build_summary(
                summary_response=summary_response,
                entities=entities,
                tables=[],  # No tables from JSON
                chart_insights=[],  # No charts from JSON
                sections=[],  # No sections from JSON
                role=role,
                document_type=document_type or DocumentType.UNKNOWN,
                start_time=start_time
            )
            
            progress.advance("Complete")
            
            return summary, progress
            
        except Exception as e:
            logger.exception(f"JSON analysis failed: {e}")
            progress.errors.append(str(e))
            
            return self._create_error_summary(
                str(e), role, document_type or DocumentType.UNKNOWN, start_time
            ), progress
    
    def _detect_document_type_from_json(self, json_data: Dict[str, Any]) -> DocumentType:
        """Detect document type from JSON structure"""
        if isinstance(json_data, dict):
            keys = set(str(k).lower() for k in json_data.keys())
            
            # Check for specific patterns
            if any(k in keys for k in ['balance_sheet', 'balancesheet', 'assets', 'liabilities']):
                return DocumentType.BALANCE_SHEET
            if any(k in keys for k in ['income', 'revenue', 'profit', 'loss', 'earnings']):
                return DocumentType.INCOME_STATEMENT
            if any(k in keys for k in ['cash_flow', 'cashflow', 'operating_activities']):
                return DocumentType.CASH_FLOW
            if any(k in keys for k in ['annual', 'yearly', '10-k', '10k']):
                return DocumentType.ANNUAL_REPORT
            if any(k in keys for k in ['quarterly', 'q1', 'q2', 'q3', 'q4', '10-q', '10q']):
                return DocumentType.QUARTERLY_REPORT
        
        return DocumentType.UNKNOWN
    
    def _fallback_json_analysis(self, json_data: Dict[str, Any], entities: List[FinancialEntity]) -> str:
        """Generate fallback analysis when LLM fails for JSON data"""
        parts = ["Financial data analysis from JSON:"]
        
        # Try to extract key metrics from JSON
        if isinstance(json_data, dict):
            for key, value in list(json_data.items())[:10]:
                if isinstance(value, (int, float)):
                    parts.append(f"- {key}: {value:,.2f}")
                elif isinstance(value, str) and len(value) < 100:
                    parts.append(f"- {key}: {value}")
        
        # Add entity info
        if entities:
            parts.append("\nExtracted financial entities:")
            for entity in entities[:10]:
                parts.append(f"- {entity.entity_type}: {entity.value}")
        
        return "\n".join(parts)
    
    def _preprocess_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Preprocess images for OCR"""
        processed = []
        for img in images:
            processed_img = self.image_processor.preprocess_for_ocr(img)
            processed.append(processed_img)
        return processed
    
    def _extract_text(self, images: List[Image.Image]) -> List[OCRResult]:
        """Extract text from all pages"""
        return self.ocr_service.extract_from_multiple_pages(images, preserve_layout=True)
    
    def _detect_sections(self, ocr_results: List[OCRResult]) -> List[DocumentSection]:
        """Detect document sections from OCR results"""
        sections = []
        current_section = None
        
        for result in ocr_results:
            for detected in result.detected_sections:
                if current_section and current_section.name != detected:
                    current_section.end_page = result.page_number
                    sections.append(current_section)
                
                current_section = DocumentSection(
                    name=detected,
                    start_page=result.page_number + 1
                )
        
        if current_section:
            current_section.end_page = ocr_results[-1].page_number + 1
            sections.append(current_section)
        
        return sections
    
    def _detect_document_type(
        self, 
        text: str, 
        sections: List[DocumentSection]
    ) -> DocumentType:
        """Auto-detect document type"""
        text_lower = text.lower()
        section_names = [s.name.lower() for s in sections]
        
        if 'annual report' in text_lower or 'annual review' in text_lower:
            return DocumentType.ANNUAL_REPORT
        
        if any(q in text_lower for q in ['quarterly', 'q1', 'q2', 'q3', 'q4']):
            return DocumentType.QUARTERLY_REPORT
        
        if 'balance sheet' in ' '.join(section_names):
            return DocumentType.BALANCE_SHEET
        
        if any(s in ' '.join(section_names) for s in ['income statement', 'profit and loss']):
            return DocumentType.INCOME_STATEMENT
        
        if 'cash flow' in ' '.join(section_names):
            return DocumentType.CASH_FLOW
        
        if 'investor' in text_lower and 'presentation' in text_lower:
            return DocumentType.INVESTOR_PRESENTATION
        
        return DocumentType.UNKNOWN
    
    def _extract_tables(
        self, 
        images: List[Image.Image],
        ocr_results: List[OCRResult]
    ) -> List[ExtractedTable]:
        """Extract tables from all pages"""
        all_tables = []
        
        for i, (img, ocr) in enumerate(zip(images, ocr_results)):
            tables = self.table_extractor.extract_tables(
                image=img,
                ocr_text=ocr.raw_text,
                page_number=i
            )
            all_tables.extend(tables)
        
        return all_tables
    
    def _analyze_charts(
        self, 
        images: List[Image.Image],
        context: str
    ) -> List[ChartInsight]:
        """Analyze charts from all pages"""
        all_insights = []
        
        for i, img in enumerate(images):
            # Preprocess for chart analysis (preserve colors)
            processed = self.image_processor.preprocess_for_chart_analysis(img)
            
            insights = self.chart_analyzer.analyze_charts(
                image=processed,
                page_number=i,
                context=context[:500]
            )
            all_insights.extend(insights)
        
        return all_insights
    
    def _fallback_analysis(
        self,
        entities: List[FinancialEntity],
        tables: List[ExtractedTable],
        chart_insights: List[ChartInsight]
    ) -> str:
        """Generate basic analysis when LLM fails"""
        parts = ["## Financial Analysis (Basic)\n"]
        
        # Summarize entities
        if entities:
            parts.append("### Key Metrics Found")
            for entity in entities[:10]:
                parts.append(f"- {entity.name}: {entity.value} {entity.unit or ''}")
            parts.append("")
        
        # Summarize tables
        if tables:
            parts.append(f"### Tables Extracted: {len(tables)}")
            for table in tables[:5]:
                if table.title:
                    parts.append(f"- {table.title}")
            parts.append("")
        
        # Summarize charts
        if chart_insights:
            parts.append("### Chart Insights")
            for insight in chart_insights:
                parts.append(f"- {insight.insight}")
            parts.append("")
        
        return '\n'.join(parts)
    
    def _build_summary(
        self,
        summary_response,
        entities: List[FinancialEntity],
        tables: List[ExtractedTable],
        chart_insights: List[ChartInsight],
        sections: List[DocumentSection],
        role: UserRole,
        document_type: DocumentType,
        start_time: float
    ) -> StructuredSummary:
        """Build the final StructuredSummary object"""
        
        processing_time = time.time() - start_time
        
        # Parse LLM response
        if summary_response.success:
            try:
                data = json.loads(summary_response.content)
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
        
        # Build executive summary
        exec_summary = ExecutiveSummary(
            overview=data.get('executive_summary', {}).get('overview', 'Summary not available'),
            key_highlights=data.get('executive_summary', {}).get('key_highlights', []),
            period_covered=data.get('executive_summary', {}).get('period_covered'),
            company_name=data.get('executive_summary', {}).get('company_name')
        )
        
        # Build financial metrics from LLM response (key_financial_metrics) or entities as fallback
        llm_metrics = data.get('key_financial_metrics', data.get('key_metrics', {}))
        financial_metrics = self._build_financial_metrics(entities, llm_metrics)
        
        # Build performance analysis
        perf_data = data.get('performance_analysis', {})
        performance = PerformanceAnalysis(
            revenue_analysis=perf_data.get('revenue_analysis'),
            profitability_analysis=perf_data.get('profitability_analysis'),
            operational_efficiency=perf_data.get('operational_efficiency'),
            segment_performance=perf_data.get('segment_performance'),
            yoy_comparison=perf_data.get('yoy_comparison'),
            qoq_comparison=perf_data.get('qoq_comparison')
        )
        
        # Build risks (try both 'risks' and 'risks_and_outlook' keys)
        risks = []
        risks_data = data.get('risks_and_outlook', data.get('risks', []))
        for risk_data in risks_data:
            risks.append(RiskItem(
                category=risk_data.get('category', 'General'),
                description=risk_data.get('description', ''),
                severity=risk_data.get('severity', 'medium').lower(),  # Ensure lowercase
                mitigation=risk_data.get('mitigation')
            ))
        
        # Build actionable insights
        insights = []
        for insight_data in data.get('actionable_insights', []):
            insights.append(ActionableInsight(
                category=insight_data.get('category', 'General'),
                insight=insight_data.get('insight', ''),
                action=insight_data.get('action'),
                priority=insight_data.get('priority', 'medium').lower(),  # Ensure lowercase
                supporting_data=insight_data.get('supporting_data')
            ))
        
        # Convert chart insights to schema format
        schema_charts = []
        for ci in chart_insights:
            from ..models.schemas import ChartInsight as SchemaChartInsight
            schema_charts.append(SchemaChartInsight(
                chart_type=ci.chart_type.value,
                title=ci.title,
                trend=SchemaTrend(ci.trend.value),
                key_values=ci.key_values,
                insight=ci.insight,
                page_number=ci.page_number
            ))
        
        # Convert tables to schema format
        from ..models.schemas import ExtractedTable as SchemaTable
        schema_tables = []
        for t in tables:
            schema_tables.append(SchemaTable(
                title=t.title,
                headers=t.headers,
                rows=t.rows,
                page_number=t.page_number,
                table_type=t.table_type
            ))
        
        # Calculate confidence score
        confidence = self._calculate_confidence(entities, tables, chart_insights, summary_response.success)
        
        return StructuredSummary(
            executive_summary=exec_summary,
            key_financial_metrics=financial_metrics,
            performance_analysis=performance,
            risks_and_outlook=risks,
            actionable_insights=insights,
            chart_insights=schema_charts,
            extracted_tables=schema_tables,
            document_sections=sections,
            analysis_role=role,
            document_type=document_type,
            processing_time_seconds=processing_time,
            confidence_score=confidence,
            generated_at=datetime.utcnow()
        )
    
    def _build_financial_metrics(
        self, 
        entities: List[FinancialEntity],
        llm_metrics: Dict
    ) -> FinancialMetrics:
        """Build FinancialMetrics from LLM response, with entities as fallback.
        
        PRIORITY: LLM metrics (from structured JSON) > extracted entities
        This ensures we use the correct period-keyed values.
        """
        
        def find_entity(name: str) -> Optional[FinancialEntity]:
            for e in entities:
                if e.name == name:
                    return e
            return None
        
        def to_extracted_metric(entity: Optional[FinancialEntity], llm_data: Dict = None) -> Optional[ExtractedMetric]:
            # PRIORITY: Use LLM data (from structured JSON) first
            if llm_data and llm_data.get('value') is not None:
                val = llm_data.get('value')
                # Handle dict or scalar value
                if isinstance(val, dict):
                    val = val.get('value')
                return ExtractedMetric(
                    name=llm_data.get('name', ''),
                    value=val,
                    unit=llm_data.get('unit', ''),
                    period=llm_data.get('period'),
                    confidence=0.95  # High confidence for structured data
                )
            # Fallback to entity extraction
            if entity:
                return ExtractedMetric(
                    name=entity.name,
                    value=entity.value,
                    unit=entity.unit,
                    period=entity.period,
                    change=entity.change_value,
                    change_type=entity.change_type,
                    confidence=entity.confidence
                )
            return None
        
        # Get other metrics not in standard list
        standard_metrics = {'revenue', 'net_profit', 'gross_profit', 'operating_profit',
                          'ebitda', 'eps', 'total_assets', 'total_liabilities', 'equity',
                          'roe', 'roa', 'current_ratio', 'debt_equity'}
        other = []
        for e in entities:
            if e.name not in standard_metrics:
                other.append(ExtractedMetric(
                    name=e.name,
                    value=e.value,
                    unit=e.unit,
                    period=e.period,
                    change=e.change_value,
                    change_type=e.change_type,
                    confidence=e.confidence
                ))
        
        return FinancialMetrics(
            revenue=to_extracted_metric(find_entity('revenue'), llm_metrics.get('revenue')),
            net_profit=to_extracted_metric(find_entity('net_profit'), llm_metrics.get('net_profit')),
            gross_profit=to_extracted_metric(find_entity('gross_profit')),
            operating_profit=to_extracted_metric(find_entity('operating_profit')),
            ebitda=to_extracted_metric(find_entity('ebitda')),
            eps=to_extracted_metric(find_entity('eps'), llm_metrics.get('eps')),
            total_assets=to_extracted_metric(find_entity('total_assets')),
            total_liabilities=to_extracted_metric(find_entity('total_liabilities')),
            equity=to_extracted_metric(find_entity('equity'), llm_metrics.get('equity')),
            roe=to_extracted_metric(find_entity('roe'), llm_metrics.get('roe')),
            roa=to_extracted_metric(find_entity('roa')),
            current_ratio=to_extracted_metric(find_entity('current_ratio')),
            debt_equity_ratio=to_extracted_metric(find_entity('debt_equity')),
            other_metrics=other
        )
    
    def _calculate_confidence(
        self,
        entities: List[FinancialEntity],
        tables: List[ExtractedTable],
        charts: List[ChartInsight],
        llm_success: bool
    ) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        # Entity extraction confidence
        if entities:
            entity_conf = sum(e.confidence for e in entities) / len(entities)
            scores.append(entity_conf)
        else:
            scores.append(0.3)
        
        # Table extraction (binary)
        scores.append(0.8 if tables else 0.5)
        
        # Chart analysis
        if charts:
            chart_conf = sum(c.confidence for c in charts) / len(charts)
            scores.append(chart_conf)
        else:
            scores.append(0.5)
        
        # LLM success
        scores.append(0.9 if llm_success else 0.4)
        
        return sum(scores) / len(scores)
    
    def _create_error_summary(
        self,
        error_message: str,
        role: UserRole,
        document_type: DocumentType,
        start_time: float
    ) -> StructuredSummary:
        """Create a minimal summary on error"""
        return StructuredSummary(
            executive_summary=ExecutiveSummary(
                overview=f"Analysis failed: {error_message}",
                key_highlights=["Processing error occurred"]
            ),
            key_financial_metrics=FinancialMetrics(),
            performance_analysis=PerformanceAnalysis(),
            risks_and_outlook=[],
            actionable_insights=[],
            chart_insights=[],
            extracted_tables=[],
            document_sections=[],
            analysis_role=role,
            document_type=document_type,
            processing_time_seconds=time.time() - start_time,
            confidence_score=0.0
        )
