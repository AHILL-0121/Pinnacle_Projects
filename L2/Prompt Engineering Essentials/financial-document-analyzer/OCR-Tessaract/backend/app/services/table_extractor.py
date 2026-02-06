"""
Table Extraction Service
Extracts and parses tables from financial documents
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedCell:
    """A single table cell"""
    value: Any
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    is_numeric: bool = False


@dataclass
class ExtractedTable:
    """A complete extracted table"""
    title: Optional[str]
    headers: List[str]
    rows: List[List[Any]]
    cells: List[ExtractedCell] = field(default_factory=list)
    page_number: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    table_type: Optional[str] = None
    confidence: float = 0.0


class TableExtractor:
    """
    Service for extracting tables from financial documents.
    Uses multiple strategies: OCR-based, structure detection, and ML-based.
    """
    
    def __init__(self):
        self._table_transformer_available = False
        self._init_table_detection()
    
    def _init_table_detection(self):
        """Initialize table detection models if available"""
        try:
            # Try to load table-transformer for better detection
            from transformers import TableTransformerForObjectDetection, DetrImageProcessor
            self._table_transformer_available = True
            logger.info("Table Transformer available for enhanced detection")
        except ImportError:
            logger.info("Table Transformer not available, using rule-based detection")
    
    def extract_tables(
        self, 
        image: Image.Image,
        ocr_text: str,
        page_number: int = 0
    ) -> List[ExtractedTable]:
        """
        Extract tables from an image.
        
        Args:
            image: PIL Image containing tables
            ocr_text: OCR-extracted text from the image
            page_number: Page number for reference
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        # Strategy 1: Parse tables from OCR text
        text_tables = self._extract_tables_from_text(ocr_text, page_number)
        tables.extend(text_tables)
        
        # Strategy 2: Detect table regions in image
        # (More advanced - uses visual features)
        visual_tables = self._detect_table_regions(image, page_number)
        
        # Merge and deduplicate
        tables = self._merge_tables(tables, visual_tables)
        
        # Classify table types
        for table in tables:
            table.table_type = self._classify_table(table)
        
        return tables
    
    def _extract_tables_from_text(
        self, 
        text: str, 
        page_number: int
    ) -> List[ExtractedTable]:
        """Extract tables by parsing structured text"""
        tables = []
        lines = text.split('\n')
        
        current_table_lines = []
        in_table = False
        table_title = None
        
        for i, line in enumerate(lines):
            # Check if line looks like table data
            is_table_line = self._is_table_line(line)
            
            # Check for table title
            if not in_table and self._is_table_title(line):
                table_title = line.strip()
                continue
            
            if is_table_line:
                if not in_table:
                    in_table = True
                current_table_lines.append(line)
            else:
                if in_table and len(current_table_lines) >= 2:
                    # End of table - parse it
                    table = self._parse_table_lines(
                        current_table_lines, 
                        table_title,
                        page_number
                    )
                    if table:
                        tables.append(table)
                
                in_table = False
                current_table_lines = []
                table_title = None
        
        # Handle last table
        if in_table and len(current_table_lines) >= 2:
            table = self._parse_table_lines(
                current_table_lines,
                table_title,
                page_number
            )
            if table:
                tables.append(table)
        
        return tables
    
    def _is_table_line(self, line: str) -> bool:
        """Determine if a line is part of a table"""
        line = line.strip()
        
        if not line:
            return False
        
        # Count potential column separators
        tab_count = line.count('\t')
        space_runs = len(re.findall(r'\s{2,}', line))
        pipe_count = line.count('|')
        
        # Count numbers (financial tables have many numbers)
        numbers = re.findall(r'[\d,]+\.?\d*', line)
        
        # Heuristics for table detection
        if tab_count >= 2:
            return True
        if space_runs >= 2 and len(numbers) >= 2:
            return True
        if pipe_count >= 2:
            return True
        if len(numbers) >= 3:
            return True
        
        return False
    
    def _is_table_title(self, line: str) -> bool:
        """Check if line is a potential table title"""
        line = line.strip().lower()
        
        title_patterns = [
            r'balance sheet',
            r'income statement',
            r'cash flow',
            r'statement of',
            r'financial (position|results|summary)',
            r'(quarterly|annual) results',
            r'key (financial )?metrics',
            r'consolidated',
            r'segment',
            r'(in |amounts in ).*(million|billion|thousand)',
        ]
        
        return any(re.search(p, line) for p in title_patterns)
    
    def _parse_table_lines(
        self, 
        lines: List[str],
        title: Optional[str],
        page_number: int
    ) -> Optional[ExtractedTable]:
        """Parse table lines into structured table"""
        if len(lines) < 2:
            return None
        
        # Detect column positions
        columns = self._detect_columns(lines)
        
        if not columns or len(columns) < 2:
            return None
        
        # Parse rows
        rows = []
        for line in lines:
            row = self._parse_row(line, columns)
            if row:
                rows.append(row)
        
        if len(rows) < 2:
            return None
        
        # First row is usually header
        headers = rows[0]
        data_rows = rows[1:]
        
        # Clean headers
        headers = [str(h).strip() for h in headers]
        
        return ExtractedTable(
            title=title,
            headers=headers,
            rows=data_rows,
            page_number=page_number,
            confidence=0.7
        )
    
    def _detect_columns(self, lines: List[str]) -> List[Tuple[int, int]]:
        """Detect column boundaries from multiple lines"""
        # Find consistent whitespace positions
        all_spaces = []
        
        for line in lines:
            spaces = []
            in_space = False
            space_start = 0
            
            for i, char in enumerate(line):
                if char in ' \t':
                    if not in_space:
                        in_space = True
                        space_start = i
                else:
                    if in_space:
                        if i - space_start >= 2:  # Significant gap
                            spaces.append((space_start, i))
                        in_space = False
            
            all_spaces.append(spaces)
        
        if not all_spaces:
            return []
        
        # Find common column boundaries
        # Use the first non-empty line as reference
        reference_spaces = next((s for s in all_spaces if s), [])
        
        if not reference_spaces:
            return []
        
        # Create columns from gaps
        columns = []
        prev_end = 0
        
        for start, end in reference_spaces:
            columns.append((prev_end, start))
            prev_end = end
        
        # Add last column
        max_len = max(len(line) for line in lines)
        columns.append((prev_end, max_len))
        
        return columns
    
    def _parse_row(self, line: str, columns: List[Tuple[int, int]]) -> List[Any]:
        """Parse a single row using column positions"""
        row = []
        
        for start, end in columns:
            if start < len(line):
                cell_text = line[start:min(end, len(line))].strip()
                cell_value = self._parse_cell_value(cell_text)
                row.append(cell_value)
            else:
                row.append("")
        
        return row
    
    def _parse_cell_value(self, text: str) -> Any:
        """Parse cell text into appropriate type"""
        text = text.strip()
        
        if not text or text in ['-', '--', 'N/A', 'n/a', '—']:
            return None
        
        # Remove currency symbols and thousand separators
        cleaned = re.sub(r'[\$€£¥₹,]', '', text)
        
        # Try to parse as number
        try:
            # Handle percentages
            if '%' in text:
                return float(cleaned.replace('%', '')) / 100
            
            # Handle parentheses for negative numbers
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            # Handle millions/billions abbreviations
            multiplier = 1
            if cleaned.lower().endswith('m') or cleaned.lower().endswith('mn'):
                cleaned = re.sub(r'[mM][nN]?$', '', cleaned)
                multiplier = 1_000_000
            elif cleaned.lower().endswith('b') or cleaned.lower().endswith('bn'):
                cleaned = re.sub(r'[bB][nN]?$', '', cleaned)
                multiplier = 1_000_000_000
            
            value = float(cleaned) * multiplier
            return value
            
        except ValueError:
            return text
    
    def _detect_table_regions(
        self, 
        image: Image.Image, 
        page_number: int
    ) -> List[ExtractedTable]:
        """Detect table regions using visual analysis"""
        # Simplified visual table detection
        # In production, use table-transformer or similar
        
        img_array = np.array(image.convert('L'))
        
        # Detect horizontal and vertical lines
        h_lines = self._detect_lines(img_array, axis=0)
        v_lines = self._detect_lines(img_array, axis=1)
        
        # If sufficient grid structure detected, mark as table region
        if len(h_lines) >= 3 and len(v_lines) >= 2:
            # Could extract more details with OCR on region
            pass
        
        return []  # Visual extraction handled separately
    
    def _detect_lines(self, img_array: np.ndarray, axis: int) -> List[int]:
        """Detect line positions in image"""
        # Simple line detection using projection
        projection = np.mean(img_array, axis=axis)
        
        # Find valleys (dark lines)
        threshold = np.mean(projection) * 0.7
        lines = np.where(projection < threshold)[0]
        
        # Cluster nearby positions
        if len(lines) == 0:
            return []
        
        clustered = []
        current = [lines[0]]
        
        for pos in lines[1:]:
            if pos - current[-1] <= 3:
                current.append(pos)
            else:
                clustered.append(int(np.mean(current)))
                current = [pos]
        
        if current:
            clustered.append(int(np.mean(current)))
        
        return clustered
    
    def _merge_tables(
        self, 
        text_tables: List[ExtractedTable],
        visual_tables: List[ExtractedTable]
    ) -> List[ExtractedTable]:
        """Merge tables from different extraction methods"""
        # For now, prioritize text-based tables
        # Could implement IoU-based merging for overlapping tables
        return text_tables + visual_tables
    
    def _classify_table(self, table: ExtractedTable) -> str:
        """Classify the type of financial table"""
        title = (table.title or '').lower()
        headers = ' '.join(table.headers).lower()
        combined = title + ' ' + headers
        
        if any(kw in combined for kw in ['balance sheet', 'assets', 'liabilities', 'equity']):
            return 'balance_sheet'
        
        if any(kw in combined for kw in ['income', 'revenue', 'profit', 'loss', 'earnings']):
            return 'income_statement'
        
        if any(kw in combined for kw in ['cash flow', 'operating', 'investing', 'financing']):
            return 'cash_flow'
        
        if any(kw in combined for kw in ['ratio', 'roe', 'roa', 'margin']):
            return 'ratios'
        
        if any(kw in combined for kw in ['segment', 'region', 'geography']):
            return 'segment'
        
        return 'general'
    
    def tables_to_markdown(self, tables: List[ExtractedTable]) -> str:
        """Convert tables to markdown format"""
        md_parts = []
        
        for i, table in enumerate(tables):
            if table.title:
                md_parts.append(f"### {table.title}\n")
            else:
                md_parts.append(f"### Table {i + 1}\n")
            
            if table.headers:
                md_parts.append('| ' + ' | '.join(str(h) for h in table.headers) + ' |')
                md_parts.append('| ' + ' | '.join(['---'] * len(table.headers)) + ' |')
            
            for row in table.rows:
                formatted_row = []
                for cell in row:
                    if cell is None:
                        formatted_row.append('-')
                    elif isinstance(cell, float):
                        if abs(cell) >= 1_000_000:
                            formatted_row.append(f"{cell/1_000_000:.1f}M")
                        elif abs(cell) >= 1_000:
                            formatted_row.append(f"{cell/1_000:.1f}K")
                        else:
                            formatted_row.append(f"{cell:.2f}")
                    else:
                        formatted_row.append(str(cell))
                
                md_parts.append('| ' + ' | '.join(formatted_row) + ' |')
            
            md_parts.append('')
        
        return '\n'.join(md_parts)
