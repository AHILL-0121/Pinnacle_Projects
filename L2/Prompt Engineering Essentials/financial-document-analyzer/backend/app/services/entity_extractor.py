"""
Financial Entity Extractor Service
Extracts and normalizes financial metrics from text
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Result of deterministic data validation"""
    has_financial_data: bool
    signals: Dict[str, bool]
    extracted_values: Dict[str, Any]
    confidence_score: float
    raw_matches: List[str]


@dataclass
class FinancialEntity:
    """A single extracted financial entity"""
    name: str
    value: Any
    raw_text: str
    unit: Optional[str] = None
    period: Optional[str] = None
    change_value: Optional[float] = None
    change_type: Optional[str] = None  # YoY, QoQ, MoM
    confidence: float = 1.0
    category: str = "general"  # revenue, profit, ratio, etc.


@dataclass
class FinancialContext:
    """Context information for entity extraction"""
    company_name: Optional[str] = None
    fiscal_period: Optional[str] = None
    currency: str = "USD"
    scale: str = "units"  # units, thousands, millions, billions
    document_type: Optional[str] = None


class FinancialEntityExtractor:
    """
    Service for extracting financial entities from text.
    Identifies metrics, normalizes values, and detects period comparisons.
    """
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction"""
        
        # Currency patterns
        self.currency_symbols = {
            '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY',
            '₹': 'INR', '₩': 'KRW', 'R$': 'BRL', 'A$': 'AUD'
        }
        
        # Number pattern with optional currency and scale
        self.number_pattern = re.compile(
            r'[\$€£¥₹]?\s*\(?\s*'  # Optional currency and opening paren
            r'([\d,]+\.?\d*)'  # Number
            r'\s*\)?'  # Optional closing paren
            r'\s*(mn|bn|m|b|million|billion|thousand|k|cr|crore|lakh)?'  # Scale
            r'\s*(%)?',  # Percentage
            re.IGNORECASE
        )
        
        # Period patterns
        self.period_pattern = re.compile(
            r'\b(Q[1-4]\s*\'?\d{2,4}|'
            r'FY\s*\'?\d{2,4}|'
            r'\d{4}|'
            r'H[1-2]\s*\'?\d{2,4}|'
            r'YTD\s*\d{2,4}|'
            r'TTM|LTM|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\'?\d{2,4})\b',
            re.IGNORECASE
        )
        
        # Change/comparison patterns
        self.change_pattern = re.compile(
            r'(increase|decrease|growth|decline|rise|fall|up|down|grew|dropped|improved|deteriorated)'
            r'\s*(?:of|by)?\s*'
            r'([\d,]+\.?\d*)\s*(%)?'
            r'\s*(YoY|QoQ|MoM|year[\s-]over[\s-]year|quarter[\s-]over[\s-]quarter)?',
            re.IGNORECASE
        )
        
        # Financial metric patterns
        self.metric_patterns = {
            'revenue': re.compile(
                r'(?:total\s+)?(?:net\s+)?(?:revenue|sales|turnover|income\s+from\s+operations)'
                r'[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
                r'\s*(mn|bn|m|b|million|billion|thousand)?',
                re.IGNORECASE
            ),
            'net_profit': re.compile(
                r'(?:net\s+)?(?:profit|income|earnings)(?:\s+after\s+tax)?(?:\s+\(PAT\))?'
                r'[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
                r'\s*(mn|bn|m|b|million|billion)?',
                re.IGNORECASE
            ),
            'ebitda': re.compile(
                r'EBITDA[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
                r'\s*(mn|bn|m|b|million|billion)?',
                re.IGNORECASE
            ),
            'eps': re.compile(
                r'(?:basic\s+)?(?:diluted\s+)?(?:EPS|earnings\s+per\s+share)'
                r'[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?',
                re.IGNORECASE
            ),
            'total_assets': re.compile(
                r'total\s+assets?[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
                r'\s*(mn|bn|m|b|million|billion)?',
                re.IGNORECASE
            ),
            'total_liabilities': re.compile(
                r'total\s+liabilit(?:y|ies)[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
                r'\s*(mn|bn|m|b|million|billion)?',
                re.IGNORECASE
            ),
            'equity': re.compile(
                r'(?:shareholders?\s*\'?\s*)?(?:total\s+)?equity'
                r'[:\s]+[\$€£¥₹]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
                r'\s*(mn|bn|m|b|million|billion)?',
                re.IGNORECASE
            ),
            'roe': re.compile(
                r'(?:ROE|return\s+on\s+equity)[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'roa': re.compile(
                r'(?:ROA|return\s+on\s+assets?)[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'npm': re.compile(
                r'(?:net\s+profit\s+margin|NPM)[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'gross_margin': re.compile(
                r'gross\s+(?:profit\s+)?margin[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'operating_margin': re.compile(
                r'(?:operating|EBIT)\s+margin[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'debt_equity': re.compile(
                r'(?:debt[/\-\s](?:to[/\-\s])?equity|D/E)\s*(?:ratio)?[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?',
                re.IGNORECASE
            ),
            'current_ratio': re.compile(
                r'current\s+ratio[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?',
                re.IGNORECASE
            ),
            'cet1': re.compile(
                r'CET[1I]\s*(?:ratio)?[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'npl': re.compile(
                r'(?:NPL|non[\s-]?performing\s+loan)\s*(?:ratio)?[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'cost_income': re.compile(
                r'cost[/\-\s](?:to[/\-\s])?income\s*(?:ratio)?[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
            'dividend_yield': re.compile(
                r'dividend\s+yield[:\s]+\(?\s*([\d,]+\.?\d*)\s*\)?[:\s]*%?',
                re.IGNORECASE
            ),
        }
    
    def extract_entities(
        self, 
        text: str,
        context: Optional[FinancialContext] = None
    ) -> List[FinancialEntity]:
        """
        Extract all financial entities from text.
        
        Args:
            text: Document text to analyze
            context: Optional context about the document
            
        Returns:
            List of FinancialEntity objects
        """
        entities = []
        context = context or FinancialContext()
        
        # Detect document scale (millions, billions, etc.)
        detected_scale = self._detect_scale(text)
        if detected_scale:
            context.scale = detected_scale
        
        # Detect currency
        detected_currency = self._detect_currency(text)
        if detected_currency:
            context.currency = detected_currency
        
        # Extract specific metrics
        for metric_name, pattern in self.metric_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entity = self._create_entity_from_match(
                    match, metric_name, context, text
                )
                if entity:
                    entities.append(entity)
        
        # Extract period comparisons
        comparison_entities = self._extract_comparisons(text, context)
        entities.extend(comparison_entities)
        
        # Deduplicate and validate
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _detect_scale(self, text: str) -> Optional[str]:
        """Detect the scale used in the document"""
        text_lower = text.lower()
        
        scale_patterns = [
            (r'\(?\s*(?:in|amounts?\s+in)\s*(?:USD\s+)?millions?\s*\)?', 'millions'),
            (r'\(?\s*(?:in|amounts?\s+in)\s*(?:USD\s+)?billions?\s*\)?', 'billions'),
            (r'\(?\s*(?:in|amounts?\s+in)\s*(?:USD\s+)?thousands?\s*\)?', 'thousands'),
            (r'\(?\s*(?:₹|Rs\.?|INR)\s*(?:in\s+)?(?:crores?|cr)\s*\)?', 'crores'),
            (r'\(?\s*(?:₹|Rs\.?|INR)\s*(?:in\s+)?(?:lakhs?|lacs?)\s*\)?', 'lakhs'),
            (r'\$\s*(?:in\s+)?mn', 'millions'),
            (r'\$\s*(?:in\s+)?bn', 'billions'),
        ]
        
        for pattern, scale in scale_patterns:
            if re.search(pattern, text_lower):
                return scale
        
        return None
    
    def _detect_currency(self, text: str) -> Optional[str]:
        """Detect primary currency used"""
        currency_counts = {}
        
        for symbol, code in self.currency_symbols.items():
            count = text.count(symbol)
            if count > 0:
                currency_counts[code] = count
        
        # Also check for spelled out currencies
        if 'USD' in text.upper() or 'dollar' in text.lower():
            currency_counts['USD'] = currency_counts.get('USD', 0) + 5
        if 'EUR' in text.upper() or 'euro' in text.lower():
            currency_counts['EUR'] = currency_counts.get('EUR', 0) + 5
        if 'INR' in text.upper() or 'rupee' in text.lower():
            currency_counts['INR'] = currency_counts.get('INR', 0) + 5
        
        if currency_counts:
            return max(currency_counts, key=currency_counts.get)
        
        return None
    
    def _create_entity_from_match(
        self,
        match: re.Match,
        metric_name: str,
        context: FinancialContext,
        full_text: str
    ) -> Optional[FinancialEntity]:
        """Create FinancialEntity from regex match"""
        try:
            raw_value = match.group(1).replace(',', '')
            value = float(raw_value)
            
            # Get scale from match if present
            scale_group = match.group(2) if match.lastindex >= 2 else None
            scale = self._normalize_scale(scale_group) or context.scale
            
            # Apply scale
            value = self._apply_scale(value, scale)
            
            # Determine unit
            unit = self._get_unit_for_metric(metric_name, context.currency)
            
            # Look for period near the match
            period = self._find_nearby_period(full_text, match.start())
            
            # Look for change info
            change_value, change_type = self._find_change_info(
                full_text, match.start(), match.end()
            )
            
            return FinancialEntity(
                name=metric_name,
                value=value,
                raw_text=match.group(0),
                unit=unit,
                period=period,
                change_value=change_value,
                change_type=change_type,
                category=self._get_metric_category(metric_name),
                confidence=0.9
            )
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse entity: {e}")
            return None
    
    def _normalize_scale(self, scale: Optional[str]) -> Optional[str]:
        """Normalize scale string"""
        if not scale:
            return None
        
        scale_lower = scale.lower()
        
        if scale_lower in ['mn', 'm', 'million', 'millions']:
            return 'millions'
        if scale_lower in ['bn', 'b', 'billion', 'billions']:
            return 'billions'
        if scale_lower in ['k', 'thousand', 'thousands']:
            return 'thousands'
        if scale_lower in ['cr', 'crore', 'crores']:
            return 'crores'
        if scale_lower in ['lakh', 'lakhs', 'lac', 'lacs']:
            return 'lakhs'
        
        return None
    
    def _apply_scale(self, value: float, scale: str) -> float:
        """Apply scale multiplier to value"""
        multipliers = {
            'units': 1,
            'thousands': 1_000,
            'millions': 1_000_000,
            'billions': 1_000_000_000,
            'lakhs': 100_000,
            'crores': 10_000_000,
        }
        
        return value * multipliers.get(scale, 1)
    
    def _get_unit_for_metric(self, metric_name: str, currency: str) -> str:
        """Get appropriate unit for a metric"""
        ratio_metrics = {
            'roe', 'roa', 'npm', 'gross_margin', 'operating_margin',
            'debt_equity', 'current_ratio', 'cet1', 'npl', 'cost_income',
            'dividend_yield'
        }
        
        if metric_name in ratio_metrics:
            if 'ratio' in metric_name or metric_name in ['debt_equity', 'current_ratio']:
                return 'ratio'
            return '%'
        
        return currency
    
    def _get_metric_category(self, metric_name: str) -> str:
        """Categorize a metric"""
        categories = {
            'revenue': 'revenue',
            'net_profit': 'profitability',
            'ebitda': 'profitability',
            'eps': 'profitability',
            'gross_margin': 'profitability',
            'npm': 'profitability',
            'operating_margin': 'profitability',
            'total_assets': 'balance_sheet',
            'total_liabilities': 'balance_sheet',
            'equity': 'balance_sheet',
            'roe': 'ratio',
            'roa': 'ratio',
            'debt_equity': 'ratio',
            'current_ratio': 'ratio',
            'cet1': 'capital',
            'npl': 'asset_quality',
            'cost_income': 'efficiency',
            'dividend_yield': 'returns',
        }
        
        return categories.get(metric_name, 'general')
    
    def _find_nearby_period(self, text: str, position: int, window: int = 100) -> Optional[str]:
        """Find fiscal period mentioned near a position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]
        
        match = self.period_pattern.search(context)
        if match:
            return match.group(1)
        
        return None
    
    def _find_change_info(
        self, 
        text: str, 
        start: int, 
        end: int
    ) -> Tuple[Optional[float], Optional[str]]:
        """Find change/comparison information near a metric"""
        # Look in window after the metric
        window = text[end:end + 150]
        
        match = self.change_pattern.search(window)
        if match:
            direction = match.group(1).lower()
            value = float(match.group(2).replace(',', ''))
            is_percentage = match.group(3) is not None
            change_type = match.group(4) or 'YoY'
            
            if is_percentage:
                value = value / 100
            
            if direction in ['decrease', 'decline', 'fall', 'down', 'dropped', 'deteriorated']:
                value = -value
            
            # Normalize change type
            change_type = change_type.upper()
            if 'YEAR' in change_type:
                change_type = 'YoY'
            elif 'QUARTER' in change_type:
                change_type = 'QoQ'
            elif 'MONTH' in change_type:
                change_type = 'MoM'
            
            return value, change_type
        
        return None, None
    
    def _extract_comparisons(
        self, 
        text: str,
        context: FinancialContext
    ) -> List[FinancialEntity]:
        """Extract period comparison statements"""
        entities = []
        
        # Pattern for "X grew/declined by Y% YoY"
        comparison_patterns = [
            (
                r'(\w+(?:\s+\w+)?)\s+(grew|increased|rose|improved|declined|decreased|fell|dropped)'
                r'\s+(?:by\s+)?([\d,]+\.?\d*)\s*%?\s*(YoY|QoQ|MoM)?',
                'comparison'
            ),
            (
                r'([\d,]+\.?\d*)\s*%\s+(growth|decline|increase|decrease)\s+in\s+(\w+(?:\s+\w+)?)'
                r'\s*(YoY|QoQ)?',
                'growth_rate'
            ),
        ]
        
        for pattern, pattern_type in comparison_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if pattern_type == 'comparison':
                    metric_name = match.group(1).lower().replace(' ', '_')
                    direction = match.group(2).lower()
                    value = float(match.group(3).replace(',', ''))
                    change_type = match.group(4) or 'YoY'
                    
                    if direction in ['declined', 'decreased', 'fell', 'dropped']:
                        value = -value
                    
                else:  # growth_rate
                    value = float(match.group(1).replace(',', ''))
                    direction = match.group(2).lower()
                    metric_name = match.group(3).lower().replace(' ', '_')
                    change_type = match.group(4) or 'YoY'
                    
                    if direction in ['decline', 'decrease']:
                        value = -value
                
                entities.append(FinancialEntity(
                    name=f"{metric_name}_change",
                    value=value,
                    raw_text=match.group(0),
                    unit='%',
                    change_value=value,
                    change_type=change_type.upper(),
                    category='comparison',
                    confidence=0.8
                ))
        
        return entities
    
    def _deduplicate_entities(
        self, 
        entities: List[FinancialEntity]
    ) -> List[FinancialEntity]:
        """Remove duplicate entities, keeping highest confidence"""
        seen = {}
        
        for entity in entities:
            key = (entity.name, entity.period)
            
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        
        return list(seen.values())
    
    def entities_to_dict(self, entities: List[FinancialEntity]) -> Dict[str, Any]:
        """Convert entities to structured dictionary"""
        result = {
            'metrics': {},
            'comparisons': {},
            'ratios': {}
        }
        
        for entity in entities:
            data = {
                'value': entity.value,
                'unit': entity.unit,
                'period': entity.period,
                'raw': entity.raw_text
            }
            
            if entity.change_value is not None:
                data['change'] = entity.change_value
                data['change_type'] = entity.change_type
            
            if entity.category == 'comparison':
                result['comparisons'][entity.name] = data
            elif entity.category == 'ratio':
                result['ratios'][entity.name] = data
            else:
                result['metrics'][entity.name] = data
        
        return result
    
    def format_entities_summary(self, entities: List[FinancialEntity]) -> str:
        """Format entities as readable summary"""
        if not entities:
            return "No financial metrics extracted."
        
        lines = ["## Extracted Financial Metrics\n"]
        
        # Group by category
        by_category = {}
        for entity in entities:
            category = entity.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(entity)
        
        category_labels = {
            'revenue': 'Revenue & Sales',
            'profitability': 'Profitability',
            'balance_sheet': 'Balance Sheet',
            'ratio': 'Financial Ratios',
            'capital': 'Capital Metrics',
            'asset_quality': 'Asset Quality',
            'efficiency': 'Efficiency Ratios',
            'returns': 'Returns',
            'comparison': 'Period Comparisons',
            'general': 'Other Metrics'
        }
        
        for category, cat_entities in by_category.items():
            label = category_labels.get(category, category.title())
            lines.append(f"### {label}")
            
            for entity in cat_entities:
                formatted_value = self._format_value(entity.value, entity.unit)
                line = f"- **{entity.name.replace('_', ' ').title()}**: {formatted_value}"
                
                if entity.period:
                    line += f" ({entity.period})"
                
                if entity.change_value is not None:
                    change_str = f"{entity.change_value:+.1%}" if entity.unit == '%' else f"{entity.change_value:+,.0f}"
                    line += f" | Change: {change_str} {entity.change_type or ''}"
                
                lines.append(line)
            
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_value(self, value: Any, unit: Optional[str]) -> str:
        """Format a value for display"""
        if value is None:
            return "N/A"
        
        if not isinstance(value, (int, float)):
            return str(value)
        
        if unit == '%':
            return f"{value:.2f}%"
        elif unit == 'ratio':
            return f"{value:.2f}x"
        elif abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B {unit or ''}"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M {unit or ''}"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.2f}K {unit or ''}"
        else:
            return f"{value:,.2f} {unit or ''}"

    def validate_financial_data(self, text: str) -> DataValidationResult:
        """
        DETERMINISTIC data validation gate.
        NO LLM is allowed to decide whether data exists - only this function.
        
        Args:
            text: OCR extracted text
            
        Returns:
            DataValidationResult with has_financial_data flag
        """
        # Comprehensive patterns to detect ANY financial data
        validation_patterns = {
            # Standard formats
            'revenue_standard': r'(?:revenue|sales|turnover)[:\s]*[\$€£]?\s*[\d,]+\.?\d*\s*(?:bn|mn|billion|million)?',
            'profit_standard': r'(?:net\s+)?profit[:\s]*[\$€£]?\s*[\d,]+\.?\d*\s*(?:bn|mn|billion|million)?',
            'equity_standard': r'equity[:\s]*[\$€£]?\s*[\d,]+\.?\d*\s*(?:bn|mn|billion|million)?',
            'roe_standard': r'(?:ROE|return\s+on\s+equity)[:\s]*[\d,]+\.?\d*\s*%?',
            'cet1_standard': r'CET[1I]\s*(?:ratio)?[:\s]*[\d,]+\.?\d*\s*%?',
            
            # Table format: "Metric (Unit) | Value"
            'revenue_table': r'Revenue\s*\([^)]*USD[^)]*\)\s*[\d\.]+',
            'profit_table': r'(?:Net\s+)?Profit\s*\([^)]*USD[^)]*\)\s*[\d\.]+',
            'equity_table': r'(?:Total\s+)?Equity\s*\([^)]*USD[^)]*\)\s*[\d\.]+',
            'roe_table': r'ROE\s*\([^)]*%[^)]*\)\s*[\d\.]+',
            'cet1_table': r'CET[1I]\s*(?:Ratio)?\s*\([^)]*%[^)]*\)\s*[\d\.]+',
            
            # Any numeric value after financial keywords
            'any_revenue': r'revenue[^0-9]*(\d+\.?\d*)',
            'any_profit': r'profit[^0-9]*(\d+\.?\d*)',
            'any_roe': r'ROE[^0-9]*(\d+\.?\d*)',
            'any_cet1': r'CET1[^0-9]*(\d+\.?\d*)',
            
            # Financial table detection
            'financial_table': r'(?:Q[1-4]\s*20\d{2}|FY\s*20\d{2})[^\n]*\d+\.?\d*',
            'period_with_value': r'20\d{2}[^\n]*(?:billion|million|USD|\$|%)',
        }
        
        signals = {}
        raw_matches = []
        extracted_values = {}
        
        for pattern_name, pattern in validation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            signals[pattern_name] = len(matches) > 0
            if matches:
                raw_matches.extend([f"{pattern_name}: {m}" for m in matches[:3]])
        
        # Extract actual values using enhanced patterns
        extracted_values = self._extract_values_deterministic(text)
        
        # Data exists if ANY pattern matches
        has_financial_data = any(signals.values()) or bool(extracted_values)
        
        # Calculate confidence based on how many patterns matched
        matched_count = sum(1 for v in signals.values() if v)
        confidence_score = min(1.0, matched_count / 5)  # 5+ matches = 100%
        
        result = DataValidationResult(
            has_financial_data=has_financial_data,
            signals=signals,
            extracted_values=extracted_values,
            confidence_score=confidence_score,
            raw_matches=raw_matches[:20]  # Limit to 20 examples
        )
        
        logger.info(f"Data validation: has_data={has_financial_data}, confidence={confidence_score:.2f}, matches={len(raw_matches)}")
        
        return result
    
    def _extract_values_deterministic(self, text: str) -> Dict[str, Any]:
        """
        Extract actual financial values using deterministic regex.
        This is the SOURCE OF TRUTH for the pipeline.
        
        CRITICAL: Preserves metric→period association.
        Uses period-keyed data structure when available.
        
        Handles:
        1. Period-keyed format: "[PERIOD: Q1 2023] revenue: 3.8..."
        2. Inline: "Revenue: 5.3 billion"
        3. Table row: "Revenue (USD Billion) 3.8 4.2 4.5..."
        """
        values = {}
        
        # Helper function to filter out year numbers (2019-2030)
        def is_valid_financial_number(num_str):
            """Check if a number is a valid financial value (not a year)"""
            try:
                num = float(num_str)
                if 2019 <= num <= 2030:
                    return False
                if num <= 0:
                    return False
                return True
            except:
                return False
        
        # STRATEGY 0 (PRIORITY): Parse period-keyed structure from pdfplumber
        # This is the most reliable method when available
        period_data = {}  # {"Q1 2023": {"revenue": 3.8, ...}, ...}
        
        # Look for period blocks: [PERIOD: Q1 2023]
        period_block_pattern = r'\[PERIOD:\s*(Q[1-4]\s*20\d{2})\](.*?)(?=\[PERIOD:|LATEST_PERIOD:|$)'
        period_matches = re.findall(period_block_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if period_matches:
            logger.info(f"Found {len(period_matches)} period blocks in text")
            
            for period, block_content in period_matches:
                period = period.upper().replace('  ', ' ')
                if period not in period_data:
                    period_data[period] = {}
                
                # Parse metric lines: "revenue: 3.8 billion USD"
                metric_lines = re.findall(r'(\w+):\s*([\d.]+)', block_content)
                for metric_key, value_str in metric_lines:
                    metric_key = metric_key.lower().strip()
                    try:
                        value = float(value_str)
                        if is_valid_financial_number(value_str):
                            period_data[period][metric_key] = value
                    except ValueError:
                        pass
        
        # Also check for [LATEST VALUES] section
        latest_section = re.search(r'\[LATEST VALUES\](.*?)(?=\[|$)', text, re.IGNORECASE | re.DOTALL)
        latest_values_explicit = {}
        if latest_section:
            metric_lines = re.findall(r'(\w+):\s*([\d.]+)', latest_section.group(1))
            for metric_key, value_str in metric_lines:
                metric_key = metric_key.lower().strip()
                try:
                    value = float(value_str)
                    if is_valid_financial_number(value_str):
                        latest_values_explicit[metric_key] = value
                except ValueError:
                    pass
        
        # Sort periods chronologically
        def period_sort_key(p):
            match = re.match(r'Q(\d)\s*(\d{4})', p, re.IGNORECASE)
            if match:
                return (int(match.group(2)), int(match.group(1)))
            return (0, 0)
        
        sorted_periods = sorted(period_data.keys(), key=period_sort_key)
        
        if sorted_periods:
            logger.info(f"Periods found (chronological): {sorted_periods}")
            values['periods'] = sorted_periods
            values['earliest_period'] = sorted_periods[0]
            values['latest_period'] = sorted_periods[-1]
            
            # Store period_data for later use
            values['period_data'] = period_data
            
            # Build metric→values mapping with correct latest/earliest
            all_metrics = set()
            for period_metrics in period_data.values():
                all_metrics.update(period_metrics.keys())
            
            for metric_key in all_metrics:
                # Collect values in chronological order
                metric_values = []
                for period in sorted_periods:
                    if metric_key in period_data.get(period, {}):
                        metric_values.append(period_data[period][metric_key])
                
                if metric_values:
                    # Use explicit latest if available, otherwise last in chronological order
                    latest_val = latest_values_explicit.get(metric_key, metric_values[-1])
                    values[metric_key] = {
                        'all_values': metric_values,  # In chronological order
                        'latest': latest_val,
                        'earliest': metric_values[0],
                        'count': len(metric_values),
                        'latest_period': values['latest_period'],
                        'earliest_period': values['earliest_period'],
                    }
                    logger.info(f"  {metric_key}: earliest={metric_values[0]} ({values['earliest_period']}) -> latest={latest_val} ({values['latest_period']})")
        
        # If period-keyed parsing worked, we're done
        if values.get('revenue') or values.get('net_profit'):
            logger.info("Using period-keyed extraction (most reliable)")
            return self._extract_company_name(text, values)
        
        # FALLBACK STRATEGIES (for documents without clear period structure)
        logger.info("Falling back to pattern-based extraction")
        
        # STRATEGY 1: Look for numbers that appear after metric labels
        flexible_patterns = {
            'revenue': [
                r'Revenue\s*\([^)]*(?:Billion|USD)[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'Revenue[:\s]+(\d+\.?\d*)',
                r'Revenue.*?(\d+\.?\d*)\s*(?:billion|bn)',
            ],
            'net_profit': [
                r'Net\s*Profit\s*\([^)]*(?:Million|USD)[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'Net\s*Profit[:\s]+(\d+\.?\d*)',
                r'Profit.*?(\d+\.?\d*)\s*(?:million|mn)',
            ],
            'equity': [
                r'Equity\s*\([^)]*(?:Billion|USD)[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'(?:Total\s+)?Equity[:\s]+(\d+\.?\d*)',
            ],
            'roe': [
                r'ROE\s*\([^)]*%[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'ROE[:\s]+(\d+\.?\d*)',
                r'Return\s+on\s+Equity[:\s]+(\d+\.?\d*)',
            ],
            'cet1_ratio': [
                r'CET1\s*(?:Ratio)?\s*\([^)]*%[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'CET1[:\s]+(\d+\.?\d*)',
            ],
            'eps': [
                r'EPS\s*\([^)]*\$?[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'EPS[:\s]+(\d+\.?\d*)',
                r'Earnings\s+per\s+Share[:\s]+(\d+\.?\d*)',
            ],
            'cost_income': [
                r'Cost[\s/-]+(?:to[\s/-]+)?Income\s*(?:Ratio)?\s*\([^)]*%[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'Cost[\s/-]+Income[:\s]+(\d+\.?\d*)',
            ],
            'npl': [
                r'NPL\s*(?:Ratio)?\s*\([^)]*%[^)]*\)[\s\n]*((?:\d+\.?\d*[\s\n]*)+)',
                r'NPL[:\s]+(\d+\.?\d*)',
            ],
        }
        
        for metric, patterns in flexible_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    numbers_str = match.group(1)
                    numbers = re.findall(r'(\d+\.?\d*)', numbers_str)
                    float_values = [float(n) for n in numbers if is_valid_financial_number(n)]
                    if float_values:
                        values[metric] = {
                            'all_values': float_values,
                            'latest': float_values[-1],  # Last in document order (risky!)
                            'earliest': float_values[0],
                            'count': len(float_values)
                        }
                        break
        
        # Detect time periods if not already found
        if 'periods' not in values:
            periods = re.findall(r'(Q[1-4]\s*20\d{2})', text, re.IGNORECASE)
            if periods:
                periods = [re.sub(r'\s+', ' ', p) for p in periods]
                unique_periods = list(dict.fromkeys(periods))
                sorted_periods = sorted(unique_periods, key=period_sort_key)
                values['periods'] = sorted_periods
                values['latest_period'] = sorted_periods[-1] if sorted_periods else None
                values['earliest_period'] = sorted_periods[0] if sorted_periods else None
        
        return self._extract_company_name(text, values)
    
    def _extract_company_name(self, text: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Extract company name from text and add to values dict."""
        company_patterns = [
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*(?:Digital\s+Finance|Financial|Bank)',
            r'([A-Z][a-zA-Z]+Bank)',
            r'([A-Z][a-zA-Z]+)\s*—\s*Financial',
        ]
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                values['company_name'] = match.group(1).strip()
                break
        
        logger.info(f"Deterministic extraction found: {list(values.keys())}")
        
        if not any(k in values for k in ['revenue', 'net_profit', 'roe', 'equity']):
            logger.warning("No financial metrics extracted from text - may need table extraction from PDF")
        
        return values
    
    def build_structured_json(self, text: str, validation: DataValidationResult) -> Dict[str, Any]:
        """
        Build the STRUCTURED JSON that becomes the SINGLE SOURCE OF TRUTH.
        This is passed directly to the report generator LLM.
        NO analysis LLM in between.
        
        CRITICAL: Includes period_data for full period→metric mapping.
        """
        extracted = validation.extracted_values
        
        # Determine units from text
        is_revenue_billions = 'billion' in text.lower() or 'USD Billion' in text
        is_profit_millions = 'million' in text.lower() or 'USD Million' in text
        
        structured = {
            'data_validation': {
                'has_financial_data': validation.has_financial_data,
                'confidence': validation.confidence_score,
                'data_verified_by_code': True,  # CRITICAL FLAG
            },
            'company': extracted.get('company_name', 'Unknown'),
            'periods': {
                'all': extracted.get('periods', []),
                'latest': extracted.get('latest_period'),
                'earliest': extracted.get('earliest_period'),
            },
            'metrics': {},
            # Include period_data for full context (NON-NEGOTIABLE per user requirement)
            'period_data': extracted.get('period_data', {}),
        }
        
        # Build metrics with proper units - USE CHRONOLOGICAL LATEST, NOT DOCUMENT ORDER
        if 'revenue' in extracted:
            rev = extracted['revenue']
            latest_val = rev.get('latest')
            earliest_val = rev.get('earliest')
            trend = rev.get('all_values', [])
            
            structured['metrics']['revenue'] = {
                'value': latest_val,
                'unit': 'billion USD' if is_revenue_billions else 'million USD',
                'period': extracted.get('latest_period'),
                'earliest_value': earliest_val,
                'earliest_period': extracted.get('earliest_period'),
                'trend': trend,  # Chronological order
            }
            logger.info(f"Revenue: {earliest_val} ({extracted.get('earliest_period')}) -> {latest_val} ({extracted.get('latest_period')})")
        
        if 'net_profit' in extracted:
            profit = extracted['net_profit']
            latest_val = profit.get('latest')
            earliest_val = profit.get('earliest')
            trend = profit.get('all_values', [])
            
            structured['metrics']['net_profit'] = {
                'value': latest_val,
                'unit': 'million USD' if is_profit_millions else 'billion USD',
                'period': extracted.get('latest_period'),
                'earliest_value': earliest_val,
                'earliest_period': extracted.get('earliest_period'),
                'trend': trend,
            }
            logger.info(f"Net Profit: {earliest_val} ({extracted.get('earliest_period')}) -> {latest_val} ({extracted.get('latest_period')})")
        
        if 'equity' in extracted:
            eq = extracted['equity']
            structured['metrics']['equity'] = {
                'value': eq.get('latest'),
                'unit': 'billion USD',
                'period': extracted.get('latest_period'),
                'earliest_value': eq.get('earliest'),
                'earliest_period': extracted.get('earliest_period'),
                'trend': eq.get('all_values', []),
            }
        
        if 'roe' in extracted:
            roe = extracted['roe']
            structured['metrics']['roe'] = {
                'value': roe.get('latest'),
                'unit': '%',
                'period': extracted.get('latest_period'),
                'earliest_value': roe.get('earliest'),
                'earliest_period': extracted.get('earliest_period'),
                'trend': roe.get('all_values', []),
            }
        
        if 'cet1_ratio' in extracted:
            cet1 = extracted['cet1_ratio']
            structured['metrics']['cet1_ratio'] = {
                'value': cet1.get('latest'),
                'unit': '%',
                'period': extracted.get('latest_period'),
                'earliest_value': cet1.get('earliest'),
                'earliest_period': extracted.get('earliest_period'),
                'trend': cet1.get('all_values', []),
            }
        
        if 'eps' in extracted:
            eps = extracted['eps']
            structured['metrics']['eps'] = {
                'value': eps.get('latest'),
                'unit': 'USD',
                'period': extracted.get('latest_period'),
                'earliest_value': eps.get('earliest'),
                'earliest_period': extracted.get('earliest_period'),
                'trend': eps.get('all_values', []),
            }
        
        if 'cost_income' in extracted or 'cost_income_ratio' in extracted:
            ci = extracted.get('cost_income') or extracted.get('cost_income_ratio')
            structured['metrics']['cost_income_ratio'] = {
                'value': ci.get('latest'),
                'unit': '%',
                'period': extracted.get('latest_period'),
            }
        
        if 'npl' in extracted or 'npl_ratio' in extracted:
            npl = extracted.get('npl') or extracted.get('npl_ratio')
            structured['metrics']['npl_ratio'] = {
                'value': npl.get('latest'),
                'unit': '%',
                'period': extracted.get('latest_period'),
            }
        
        if 'assets' in extracted:
            assets = extracted['assets']
            structured['metrics']['assets'] = {
                'value': assets.get('latest'),
                'unit': 'billion USD',
                'period': extracted.get('latest_period'),
                'earliest_value': assets.get('earliest'),
                'earliest_period': extracted.get('earliest_period'),
                'trend': assets.get('all_values', []),
            }
        
        # Log final structured data for debugging
        logger.info("=" * 60)
        logger.info("FINAL STRUCTURED JSON (Source of Truth):")
        logger.info(f"  Company: {structured['company']}")
        logger.info(f"  Period: {structured['periods']['earliest']} to {structured['periods']['latest']}")
        for metric_name, metric_data in structured['metrics'].items():
            logger.info(f"  {metric_name}: {metric_data.get('value')} {metric_data.get('unit')} ({metric_data.get('period')})")
        logger.info("=" * 60)
        
        return structured
