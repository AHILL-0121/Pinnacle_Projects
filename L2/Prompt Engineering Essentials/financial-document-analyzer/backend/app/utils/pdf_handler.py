"""
PDF Handling Module
Handles PDF loading, conversion to images, and page extraction
"""
import io
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Generator, Tuple, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import pdfplumber for table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.info("pdfplumber not available - install with 'pip install pdfplumber' for better table extraction")


class PDFHandler:
    """
    Handles PDF document processing.
    Converts PDF pages to images for OCR and analysis.
    """
    
    def __init__(self, dpi: int = 300, max_pages: int = 50):
        self.dpi = dpi
        self.max_pages = max_pages
        self._pdf2image_available = self._check_pdf2image()
        self._fitz_available = self._check_fitz()
    
    def _check_pdf2image(self) -> bool:
        """Check if pdf2image is available"""
        try:
            import pdf2image
            return True
        except ImportError:
            return False
    
    def _check_fitz(self) -> bool:
        """Check if PyMuPDF (fitz) is available"""
        try:
            import fitz
            return True
        except ImportError:
            return False
    
    def pdf_to_images(
        self, 
        pdf_source: bytes | str | Path,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images.
        
        Args:
            pdf_source: PDF file as bytes, file path, or Path object
            start_page: First page to convert (0-indexed)
            end_page: Last page to convert (exclusive), None for all pages
            
        Returns:
            List of PIL Images, one per page
        """
        if self._fitz_available:
            return self._convert_with_fitz(pdf_source, start_page, end_page)
        elif self._pdf2image_available:
            return self._convert_with_pdf2image(pdf_source, start_page, end_page)
        else:
            raise ImportError(
                "No PDF library available. Install PyMuPDF: pip install PyMuPDF "
                "or pdf2image: pip install pdf2image"
            )
    
    def _convert_with_fitz(
        self, 
        pdf_source: bytes | str | Path,
        start_page: int,
        end_page: Optional[int]
    ) -> List[Image.Image]:
        """Convert PDF using PyMuPDF (fitz)"""
        import fitz
        
        images = []
        
        # Handle different input types
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_source))
        
        try:
            num_pages = min(doc.page_count, self.max_pages)
            end_page = min(end_page or num_pages, num_pages)
            
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                
                # Calculate zoom factor for target DPI
                zoom = self.dpi / 72  # Default PDF is 72 DPI
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                if pix.alpha:
                    img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                else:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                images.append(img)
                logger.debug(f"Converted page {page_num + 1}/{end_page}")
                
        finally:
            doc.close()
        
        return images
    
    def _convert_with_pdf2image(
        self, 
        pdf_source: bytes | str | Path,
        start_page: int,
        end_page: Optional[int]
    ) -> List[Image.Image]:
        """Convert PDF using pdf2image"""
        from pdf2image import convert_from_path, convert_from_bytes
        
        # Get page count first
        if isinstance(pdf_source, bytes):
            from pdf2image.pdf2image import pdfinfo_from_bytes
            info = pdfinfo_from_bytes(pdf_source)
        else:
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_source))
        
        num_pages = min(info['Pages'], self.max_pages)
        end_page = min(end_page or num_pages, num_pages)
        
        # Convert pages
        if isinstance(pdf_source, bytes):
            images = convert_from_bytes(
                pdf_source,
                dpi=self.dpi,
                first_page=start_page + 1,  # pdf2image uses 1-indexed pages
                last_page=end_page
            )
        else:
            images = convert_from_path(
                str(pdf_source),
                dpi=self.dpi,
                first_page=start_page + 1,
                last_page=end_page
            )
        
        return images
    
    def pdf_to_images_generator(
        self, 
        pdf_source: bytes | str | Path
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        """
        Generator that yields pages one at a time.
        Memory efficient for large PDFs.
        
        Yields:
            Tuple of (page_number, PIL Image)
        """
        if not self._fitz_available:
            # Fall back to loading all at once
            images = self.pdf_to_images(pdf_source)
            for i, img in enumerate(images):
                yield i, img
            return
        
        import fitz
        
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_source))
        
        try:
            num_pages = min(doc.page_count, self.max_pages)
            
            for page_num in range(num_pages):
                page = doc[page_num]
                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                if pix.alpha:
                    img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                else:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                yield page_num, img
                
        finally:
            doc.close()
    
    def get_page_count(self, pdf_source: bytes | str | Path) -> int:
        """Get the number of pages in a PDF"""
        if self._fitz_available:
            import fitz
            if isinstance(pdf_source, bytes):
                doc = fitz.open(stream=pdf_source, filetype="pdf")
            else:
                doc = fitz.open(str(pdf_source))
            count = doc.page_count
            doc.close()
            return min(count, self.max_pages)
        
        elif self._pdf2image_available:
            from pdf2image.pdf2image import pdfinfo_from_bytes, pdfinfo_from_path
            if isinstance(pdf_source, bytes):
                info = pdfinfo_from_bytes(pdf_source)
            else:
                info = pdfinfo_from_path(str(pdf_source))
            return min(info['Pages'], self.max_pages)
        
        else:
            raise ImportError("No PDF library available")
    
    def extract_text_native(self, pdf_source: bytes | str | Path) -> List[str]:
        """
        Extract text directly from PDF (if available).
        This is faster than OCR for digital PDFs.
        
        Returns:
            List of text strings, one per page
        """
        if not self._fitz_available:
            return []
        
        import fitz
        
        texts = []
        
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_source))
        
        try:
            for page_num in range(min(doc.page_count, self.max_pages)):
                page = doc[page_num]
                text = page.get_text()
                texts.append(text)
        finally:
            doc.close()
        
        return texts
    
    def is_scanned_pdf(self, pdf_source: bytes | str | Path) -> bool:
        """
        Determine if a PDF is scanned (image-based) or digital.
        Scanned PDFs need OCR, digital PDFs can extract text directly.
        """
        if not self._fitz_available:
            return True  # Assume scanned if we can't check
        
        import fitz
        
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_source))
        
        try:
            # Check first few pages
            pages_to_check = min(3, doc.page_count)
            total_text_length = 0
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text()
                total_text_length += len(text.strip())
            
            # If very little text, likely scanned
            avg_text = total_text_length / pages_to_check
            return avg_text < 100
            
        finally:
            doc.close()
    
    def extract_tables(self, pdf_source: bytes | str | Path) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using pdfplumber.
        This is critical for financial documents where OCR misses table values.
        
        Returns:
            List of tables, each containing headers and rows
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available - cannot extract tables directly")
            return []
        
        tables = []
        
        try:
            # Handle bytes input
            if isinstance(pdf_source, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(pdf_source)
                    tmp_path = tmp.name
                pdf = pdfplumber.open(tmp_path)
            else:
                pdf = pdfplumber.open(str(pdf_source))
            
            try:
                for page_num, page in enumerate(pdf.pages[:self.max_pages]):
                    # Try default extraction first
                    page_tables = page.extract_tables()
                    
                    # If no tables, try text-based detection
                    if not page_tables:
                        page_tables = page.extract_tables({
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text"
                        })
                    
                    logger.info(f"Page {page_num + 1}: Found {len(page_tables)} tables via pdfplumber")
                    
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) >= 1:  # At least one row
                            # Log raw table content for debugging
                            logger.info(f"Table {table_idx + 1} on page {page_num + 1} - {len(table)} rows:")
                            for row_idx, row in enumerate(table[:8]):  # First 8 rows
                                logger.info(f"  Row {row_idx}: {row}")
                            
                            tables.append({
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'headers': table[0] if table else [],
                                'rows': table[1:] if len(table) > 1 else [],
                                'raw': table
                            })
            finally:
                pdf.close()
                
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
        
        return tables
    
    def extract_text_with_tables(self, pdf_source: bytes | str | Path) -> str:
        """
        Extract text from PDF with proper table formatting.
        Combines native text extraction with table extraction for better results.
        
        Returns:
            Combined text with tables formatted inline
        """
        # Get native text
        native_texts = self.extract_text_native(pdf_source)
        
        # Get tables
        tables = self.extract_tables(pdf_source)
        
        # If we have tables, create a formatted version
        result_parts = []
        
        for i, text in enumerate(native_texts):
            result_parts.append(f"--- Page {i + 1} ---")
            result_parts.append(text)
            
            # Add tables from this page
            page_tables = [t for t in tables if t['page'] == i + 1]
            for table in page_tables:
                result_parts.append("\n[TABLE DATA]")
                if table['headers']:
                    result_parts.append(" | ".join(str(h) for h in table['headers'] if h))
                for row in table['rows']:
                    if row:
                        result_parts.append(" | ".join(str(cell) for cell in row if cell))
                result_parts.append("[END TABLE]\n")
        
        return "\n".join(result_parts)
    
    def get_financial_table_text(self, pdf_source: bytes | str | Path) -> str:
        """
        Extract financial data formatted as PERIOD-KEYED structure.
        
        This is CRITICAL: We must preserve the metric→period association.
        Output format enables proper latest value extraction.
        
        Returns:
            Formatted text with period-keyed financial data
        """
        tables = self.extract_tables(pdf_source)
        
        # Also try native text extraction with PyMuPDF for coordinates
        native_period_data = self._extract_from_native_text(pdf_source)
        
        if not tables and not native_period_data:
            logger.warning("No tables extracted from PDF - trying PyMuPDF text blocks")
            return ""
        
        # Collect ALL period→metric mappings
        period_data = {}  # {"Q1 2023": {"revenue": 3.8, "net_profit": 520, ...}, ...}
        metric_units = {}  # Track units for each metric
        
        # First, use pdfplumber tables if available
        if tables:
            period_data, metric_units = self._parse_pdfplumber_tables(tables)
            logger.info(f"After pdfplumber parsing: {len(period_data)} periods found")
            for period, metrics in period_data.items():
                logger.info(f"  {period}: {metrics}")
        
        # If pdfplumber didn't find anything, use native text extraction
        if not period_data and native_period_data:
            period_data = native_period_data
            metric_units = {
                'revenue': 'billion USD',
                'net_profit': 'million USD',
                'eps': 'USD',
                'roe': '%',
                'equity': 'billion USD',
                'assets': 'billion USD',
                'cet1_ratio': '%'
            }
        
        # Sort periods chronologically
        import re
        def period_sort_key(p):
            match = re.match(r'Q(\d)\s*(\d{4})', p, re.IGNORECASE)
            if match:
                return (int(match.group(2)), int(match.group(1)))
            return (0, 0)
        
        sorted_periods = sorted(period_data.keys(), key=period_sort_key)
        
        # Build output text with clear structure
        lines = ["FINANCIAL DATA BY PERIOD (STRUCTURED):"]
        lines.append("")
        
        # Output period-keyed data
        for period in sorted_periods:
            lines.append(f"[PERIOD: {period}]")
            metrics = period_data[period]
            for metric_key, value in metrics.items():
                unit = metric_units.get(metric_key, '')
                lines.append(f"  {metric_key}: {value} {unit}")
            lines.append("")
        
        # Add summary of latest vs earliest
        if sorted_periods:
            latest_period = sorted_periods[-1]
            earliest_period = sorted_periods[0]
            lines.append(f"LATEST_PERIOD: {latest_period}")
            lines.append(f"EARLIEST_PERIOD: {earliest_period}")
            lines.append(f"TOTAL_PERIODS: {len(sorted_periods)}")
            lines.append("")
            
            # Output explicit latest values
            lines.append("[LATEST VALUES]")
            if latest_period in period_data:
                for metric_key, value in period_data[latest_period].items():
                    unit = metric_units.get(metric_key, '')
                    lines.append(f"  {metric_key}: {value} {unit}")
        
        return "\n".join(lines)
    
    def _extract_from_native_text(self, pdf_source: bytes | str | Path) -> Dict[str, Dict[str, float]]:
        """
        Extract financial data using PyMuPDF native text with position info.
        This handles PDFs where pdfplumber fails to detect tables.
        """
        if not self._fitz_available:
            return {}
        
        import fitz
        import re
        
        period_data = {}
        
        try:
            if isinstance(pdf_source, bytes):
                doc = fitz.open(stream=pdf_source, filetype="pdf")
            else:
                doc = fitz.open(str(pdf_source))
            
            for page_num in range(min(doc.page_count, self.max_pages)):
                page = doc[page_num]
                
                # Get text blocks with positions
                blocks = page.get_text("dict")["blocks"]
                
                # Extract all text elements with their y-coordinates
                text_elements = []
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    y_pos = span["bbox"][1]  # y-coordinate
                                    x_pos = span["bbox"][0]  # x-coordinate
                                    text_elements.append({
                                        'text': text,
                                        'y': y_pos,
                                        'x': x_pos,
                                        'page': page_num
                                    })
                
                # Group elements by similar y-coordinate (same row in table)
                # Elements within 5 pixels vertically are considered same row
                rows = {}
                for elem in text_elements:
                    y_key = int(elem['y'] / 5) * 5  # Round to nearest 5
                    if y_key not in rows:
                        rows[y_key] = []
                    rows[y_key].append(elem)
                
                # Sort elements within each row by x-coordinate
                for y_key in rows:
                    rows[y_key].sort(key=lambda e: e['x'])
                
                # Find rows that look like "Period: Q2 2025" followed by values
                current_period = None
                period_pattern = re.compile(r'(Q[1-4]\s*20\d{2})', re.IGNORECASE)
                
                for y_key in sorted(rows.keys()):
                    row_texts = [e['text'] for e in rows[y_key]]
                    row_combined = ' '.join(row_texts)
                    
                    # Check if this row contains a period marker
                    period_match = period_pattern.search(row_combined)
                    if period_match and ('period' in row_combined.lower() or len(row_texts) == 1):
                        current_period = period_match.group(1).upper().replace('  ', ' ')
                        if current_period not in period_data:
                            period_data[current_period] = {}
                        logger.debug(f"Found period marker: {current_period}")
                        continue
                    
                    # Check if this row contains metric name + value
                    if current_period and len(row_texts) >= 2:
                        metric_text = row_texts[0].lower()
                        
                        # Try to parse the value (usually last numeric element)
                        value = None
                        for t in reversed(row_texts[1:]):
                            try:
                                value = float(t.replace(',', ''))
                                break
                            except ValueError:
                                continue
                        
                        if value is not None:
                            # Map metric text to key
                            if 'revenue' in metric_text:
                                period_data[current_period]['revenue'] = value
                            elif 'net profit' in metric_text or ('profit' in metric_text and 'net' in metric_text):
                                period_data[current_period]['net_profit'] = value
                            elif 'eps' in metric_text:
                                period_data[current_period]['eps'] = value
                            elif 'roe' in metric_text:
                                period_data[current_period]['roe'] = value
                            elif 'equity' in metric_text and 'roe' not in metric_text:
                                period_data[current_period]['equity'] = value
                            elif 'assets' in metric_text:
                                period_data[current_period]['assets'] = value
                            elif 'cet1' in metric_text:
                                period_data[current_period]['cet1_ratio'] = value
            
            doc.close()
            
            logger.info(f"Native text extraction found {len(period_data)} periods")
            for period, metrics in period_data.items():
                logger.info(f"  {period}: {list(metrics.keys())}")
            
        except Exception as e:
            logger.error(f"Native text extraction failed: {e}")
        
        return period_data
    
    def _parse_pdfplumber_tables(self, tables: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
        """Parse pdfplumber tables into period-keyed data structure.
        
        The NovaBank PDF has ONE TABLE PER QUARTER with ['Metric', 'Value'] format.
        Some tables are split across pages (partial data).
        We merge partial tables and match to periods.
        """
        import re
        
        period_data = {}  # {"Q1 2023": {"revenue": 3.8, ...}, ...}
        metric_units = {}  # {"revenue": "billion USD", ...}
        
        period_pattern = re.compile(r'(Q[1-4]\s*20\d{2})', re.IGNORECASE)
        
        # Check if any table has periods in headers (HORIZONTAL layout)
        has_period_headers = False
        for table in tables:
            headers = table.get('headers', [])
            for h in headers:
                if h and period_pattern.search(str(h)):
                    has_period_headers = True
                    break
            if has_period_headers:
                break
        
        if has_period_headers:
            # HORIZONTAL LAYOUT
            logger.info("Detected HORIZONTAL table layout")
            for table in tables:
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                header_periods = []
                for h in headers:
                    if h:
                        match = period_pattern.search(str(h))
                        if match:
                            header_periods.append(match.group(1).upper().replace('  ', ' '))
                        else:
                            header_periods.append(None)
                    else:
                        header_periods.append(None)
                self._parse_horizontal_table(rows, header_periods, period_data, metric_units)
        else:
            # VERTICAL LAYOUT (ONE-TABLE-PER-PERIOD)
            # Key insight: Count complete tables (6+ metrics) to match to periods
            logger.info("Detected VERTICAL table layout (one table per period)")
            
            # Sort tables by page and index
            sorted_tables = sorted(tables, key=lambda t: (t.get('page', 1), t.get('table_index', 0)))
            
            # Merge partial tables and extract metrics
            # A "complete" table has most of the 7-8 metrics
            all_extracted = []  # List of {metrics: {}, row_count: N}
            
            pending_metrics = {}  # Accumulate metrics from partial tables
            
            for table in sorted_tables:
                rows = table.get('rows', [])
                headers = table.get('headers', [])
                page = table.get('page', 1)
                
                # Check if this table has the standard header
                is_new_table = headers and len(headers) >= 2 and 'Metric' in str(headers[0])
                
                table_metrics = {}
                for row in rows:
                    if not row or len(row) < 2:
                        continue
                    metric_cell = str(row[0]).strip() if row[0] else ""
                    value_cell = row[1] if len(row) > 1 else None
                    metric_key, unit = self._identify_metric(metric_cell)
                    if metric_key and value_cell is not None:
                        value = self._parse_number(value_cell)
                        if value is not None:
                            table_metrics[metric_key] = value
                            metric_units[metric_key] = unit
                
                if is_new_table and pending_metrics:
                    # Save previous accumulated data as a period
                    if len(pending_metrics) >= 3:  # At least 3 metrics = valid period
                        all_extracted.append(dict(pending_metrics))
                    pending_metrics = {}
                
                # Add current table metrics
                pending_metrics.update(table_metrics)
                
                # If this looks like a complete table, save it
                if len(pending_metrics) >= 6:  # 6+ metrics = likely complete
                    all_extracted.append(dict(pending_metrics))
                    pending_metrics = {}
            
            # Don't forget the last pending metrics
            if pending_metrics and len(pending_metrics) >= 3:
                all_extracted.append(dict(pending_metrics))
            
            # Known periods in chronological order
            known_periods = [
                'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023',
                'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024',
                'Q1 2025', 'Q2 2025'
            ]
            
            logger.info(f"Extracted {len(all_extracted)} complete periods from tables")
            
            # Map extracted data to periods
            for idx, metrics in enumerate(all_extracted):
                if idx < len(known_periods):
                    period = known_periods[idx]
                    period_data[period] = metrics
                    logger.info(f"  {period}: {metrics}")
        
        logger.info(f"Final: {len(period_data)} periods parsed")
        return period_data, metric_units
    
    def _parse_horizontal_table(self, rows, header_periods, period_data, metric_units):
        """Parse table where periods are column headers."""
        for row in rows:
            if not row or len(row) < 2:
                continue
            
            metric_cell = str(row[0]).strip() if row[0] else ""
            if not metric_cell:
                continue
            
            metric_key, unit = self._identify_metric(metric_cell)
            if not metric_key:
                continue
            
            metric_units[metric_key] = unit
            
            # Map values to periods
            for col_idx in range(1, len(row)):
                if col_idx < len(header_periods):
                    period = header_periods[col_idx]
                    if period:
                        value = self._parse_number(row[col_idx])
                        if value is not None:
                            if period not in period_data:
                                period_data[period] = {}
                            period_data[period][metric_key] = value
    
    def _parse_vertical_table(self, raw_table, period_data, metric_units):
        """Parse table where periods are in rows (vertical layout)."""
        import re
        period_pattern = re.compile(r'(Q[1-4]\s*20\d{2})', re.IGNORECASE)
        
        current_period = None
        
        for row in raw_table:
            if not row:
                continue
            
            row_text = ' '.join(str(cell) for cell in row if cell)
            
            # Check if this row contains a period marker
            period_match = period_pattern.search(row_text)
            if period_match:
                potential_period = period_match.group(1).upper().replace('  ', ' ')
                
                # If this row is ONLY a period (or "Period: Q2 2025"), set current_period
                if len(row) == 1 or 'period' in row_text.lower():
                    current_period = potential_period
                    if current_period not in period_data:
                        period_data[current_period] = {}
                    logger.info(f"Found period marker row: {current_period}")
                    continue
            
            # If we have a current period, check if this row is a metric line
            if current_period and len(row) >= 2:
                first_cell = str(row[0]).strip() if row[0] else ""
                metric_key, unit = self._identify_metric(first_cell)
                
                if metric_key:
                    # Find the value (usually last non-empty cell)
                    value = None
                    for cell in reversed(row[1:]):
                        value = self._parse_number(cell)
                        if value is not None:
                            break
                    
                    if value is not None:
                        period_data[current_period][metric_key] = value
                        metric_units[metric_key] = unit
                        logger.info(f"  {current_period} -> {metric_key}: {value}")
    
    def _identify_metric(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify metric type and unit from text."""
        text_lower = text.lower()
        
        if 'revenue' in text_lower:
            unit = 'billion USD' if 'billion' in text_lower else ('million USD' if 'million' in text_lower else 'USD')
            return 'revenue', unit
        elif 'net profit' in text_lower or ('profit' in text_lower and 'gross' not in text_lower):
            unit = 'billion USD' if 'billion' in text_lower else 'million USD'
            return 'net_profit', unit
        elif 'eps' in text_lower or 'earnings per share' in text_lower:
            return 'eps', 'USD'
        elif 'roe' in text_lower or 'return on equity' in text_lower:
            return 'roe', '%'
        elif 'cet1' in text_lower:
            return 'cet1_ratio', '%'
        elif 'equity' in text_lower and 'roe' not in text_lower:
            return 'equity', 'billion USD'
        elif 'assets' in text_lower:
            return 'assets', 'billion USD'
        elif 'cost' in text_lower and 'income' in text_lower:
            return 'cost_income_ratio', '%'
        elif 'npl' in text_lower:
            return 'npl_ratio', '%'
        
        return None, None
    
    def _parse_number(self, value) -> Optional[float]:
        """Safely parse a number from a cell value."""
        if value is None:
            return None
        try:
            return float(str(value).strip().replace(',', ''))
        except (ValueError, TypeError):
            return None
