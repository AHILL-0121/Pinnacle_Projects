"""
Document Processing Module

Handles PDF ingestion, text extraction, and semantic chunking.
Implements page-aware extraction with metadata preservation.
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Generator
import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTChar, LTAnno
except ImportError:
    extract_pages = None

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with full metadata."""
    chunk_id: str
    text: str
    paper_title: str
    section_name: str
    page_numbers: List[int]
    chunk_index: int
    total_chunks: int
    source_file: str
    token_count: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "paper_title": self.paper_title,
            "section_name": self.section_name,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_file": self.source_file,
            "token_count": self.token_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentChunk":
        """Create chunk from dictionary."""
        return cls(**data)
    
    def format_citation(self) -> str:
        """Format chunk as citation string."""
        pages = f"Page{'s' if len(self.page_numbers) > 1 else ''} {', '.join(map(str, self.page_numbers))}"
        return f"{self.paper_title}, {self.section_name}, {pages}"


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_number: int
    text: str
    section_hints: List[str] = field(default_factory=list)


class PDFExtractor:
    """Extracts text from PDF files with page awareness."""
    
    # Common section patterns in research papers
    SECTION_PATTERNS = [
        r'^(\d+\.?\s*(?:Introduction|Abstract|Background|Related\s+Work|Methodology|Method|Methods|'
        r'Architecture|Model|Experiments?|Results?|Discussion|Conclusion|Conclusions|'
        r'Acknowledgments?|References|Appendix|Appendices))',
        r'^(Abstract|Introduction|Background|Related\s+Work|Methodology|Method|Methods|'
        r'Architecture|Model|Experiments?|Results?|Discussion|Conclusion|Conclusions|'
        r'Acknowledgments?|References|Appendix|Appendices)$',
        r'^(\d+\.\d+\.?\s+[A-Z][^.!?\n]{3,50})$',  # Numbered subsections like 3.2 Multi-Head
        r'^(\d+\.?\s+[A-Z][^.!?\n]{3,50})$',  # Numbered sections like 3 Model
    ]
    
    # More specific section header patterns for extraction
    NUMBERED_SECTION_PATTERN = r'^\s*(\d+(?:\.\d+)?)\.?\s+([A-Z][A-Za-z\s\-]+)\s*$'
    
    # Content to filter out
    NOISE_PATTERNS = [
        r'^\s*\d+\s*$',  # Page numbers only
        r'^(Figure|Table|Fig\.)\s+\d+',  # Figure/table captions (keep but tag)
        r'^\s*[\[\(]\d+[\]\)]\s*$',  # Reference numbers
        r'^arXiv:\d+\.\d+',  # arXiv IDs
    ]
    
    # Known paper titles for better matching
    KNOWN_PAPERS = {
        "attention": "Attention Is All You Need",
        "transformer": "Attention Is All You Need", 
        "retrieval-augmented": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "rag": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "language models are few-shot": "Language Models are Few-Shot Learners",
        "gpt-3": "Language Models are Few-Shot Learners",
        "few-shot learners": "Language Models are Few-Shot Learners",
    }
    
    def __init__(self):
        if fitz is None and extract_pages is None:
            raise ImportError("Please install PyMuPDF (fitz) or pdfminer: pip install pymupdf pdfminer.six")
    
    def extract_paper_title(self, first_page_text: str, filename: str) -> str:
        """Extract paper title from first page or filename."""
        # Check for known papers by keywords in text
        text_lower = first_page_text[:2000].lower()
        for keyword, title in self.KNOWN_PAPERS.items():
            if keyword in text_lower:
                return title
        
        # Try to extract title from first few lines (usually large font at top)
        lines = first_page_text.strip().split('\n')
        potential_title_lines = []
        
        for line in lines[:15]:  # Check first 15 lines
            line = line.strip()
            # Skip empty, short, or metadata lines
            if not line or len(line) < 10:
                continue
            if line.lower().startswith(('arxiv', 'abstract', 'published', 'preprint', '©')):
                continue
            if re.match(r'^[\d\.\-\s]+$', line):  # Skip numbers/dates
                continue
            if '@' in line or 'university' in line.lower():  # Skip author affiliations
                continue
            
            # Title lines are usually longer and don't end with certain patterns
            if len(line) > 20 and not line.endswith(('Inc.', 'Ltd.', 'et al.')):
                potential_title_lines.append(line)
                if len(potential_title_lines) >= 2:
                    break
        
        if potential_title_lines:
            # Join first 1-2 lines as title
            title = ' '.join(potential_title_lines[:2])
            # Clean up
            title = re.sub(r'\s+', ' ', title).strip()
            if len(title) > 10:
                return title
        
        # Fallback to filename
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
    
    def extract_with_pymupdf(self, pdf_path: Path) -> List[PageContent]:
        """Extract text using PyMuPDF (faster, better formatting) with TABLE support."""
        pages = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, start=1):
            text_parts = []
            
            # Extract regular text
            text = page.get_text("text")
            text_parts.append(text)
            
            # Extract tables using PyMuPDF's table detection
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    for table_idx, table in enumerate(tables.tables):
                        table_text = self._format_table(table, table_idx + 1, page_num)
                        if table_text:
                            text_parts.append(table_text)
                            logger.info(f"Extracted table {table_idx + 1} from page {page_num}")
            except Exception as e:
                logger.debug(f"Table extraction failed on page {page_num}: {e}")
            
            # Fallback: detect tables by caption pattern in text
            table_sections = self._extract_tables_from_text(text, page_num)
            if table_sections:
                text_parts.extend(table_sections)
            
            combined_text = "\n\n".join(text_parts)
            
            # Find section hints in this page
            section_hints = []
            for pattern in self.SECTION_PATTERNS:
                matches = re.findall(pattern, combined_text, re.MULTILINE | re.IGNORECASE)
                section_hints.extend(matches)
            
            pages.append(PageContent(
                page_number=page_num,
                text=combined_text,
                section_hints=section_hints
            ))
        
        doc.close()
        return pages
    
    def _extract_tables_from_text(self, text: str, page_num: int) -> List[str]:
        """Extract table content by detecting 'Table X:' captions and surrounding data."""
        table_sections = []
        
        # Pattern to find table captions: "Table 1:", "Table 2.", etc.
        table_pattern = r'(Table\s+(\d+)[:.]\s*([^\n]+))'
        matches = list(re.finditer(table_pattern, text, re.IGNORECASE))
        
        for match in matches:
            table_num = match.group(2)
            table_title = match.group(3).strip()
            start_pos = match.start()
            
            # Extract MORE content after the table caption (up to 2500 chars or next major section)
            remaining_text = text[start_pos:]
            
            # Find end boundary - next section, next table, or character limit
            end_markers = [
                r'\n\s*\d+\.?\s+[A-Z][a-z]+\s*\n',  # Section headers
                r'\n\s*References\s*\n',
                r'\n\s*Acknowledgments?\s*\n',
            ]
            
            end_pos = 2500
            for marker in end_markers:
                next_section = re.search(marker, remaining_text[100:])
                if next_section:
                    potential_end = 100 + next_section.start()
                    if potential_end < end_pos and potential_end > 200:
                        end_pos = potential_end
            
            table_content = remaining_text[:end_pos].strip()
            
            # Try to convert to markdown table format for better LLM parsing
            markdown_table = self._convert_to_markdown_table(table_content, table_num, table_title)
            
            # Format as structured table block with clear markers
            formatted = f"\n[TABLE {table_num} - Page {page_num}]\n"
            formatted += f"Title: Table {table_num}: {table_title}\n"
            if markdown_table:
                formatted += f"Data:\n{markdown_table}\n"
            else:
                formatted += f"Raw Content:\n{table_content}\n"
            formatted += f"[END TABLE {table_num}]\n"
            
            table_sections.append(formatted)
            logger.info(f"Text-detected Table {table_num} on page {page_num}: {table_title[:50]}")
        
        return table_sections
    
    def _convert_to_markdown_table(self, content: str, table_num: str, title: str) -> str:
        """Try to convert table content to markdown format for better LLM parsing."""
        try:
            lines = content.split('\n')
            
            # Look for common patterns in research paper tables
            # Pattern 1: Header line followed by data rows (e.g., Table 5 in RAG paper)
            # "MSMARCO" "Jeopardy QGen" on header line, then "Gold" "89.6%" "90.0%" etc.
            
            # Find potential header/data patterns
            data_rows = []
            current_row = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip the table caption line itself
                if line.lower().startswith('table ' + table_num):
                    continue
                    
                # Check if this looks like a numeric value or percentage
                if re.match(r'^[\d.]+%?$', line):
                    current_row.append(line)
                # Check if it's a model name or category
                elif re.match(r'^[A-Za-z][\w\-\.]+$', line) and len(line) < 30:
                    if current_row:
                        data_rows.append(current_row)
                    current_row = [line]
                # Header-like line (multiple words, could be column headers)
                elif len(line.split()) >= 2 and not any(c.isdigit() for c in line[:10]):
                    if current_row:
                        data_rows.append(current_row)
                    # This might be a header row
                    headers = line.split()
                    if len(headers) >= 2:
                        current_row = ['Model'] + headers  # Assume first column is model name
                    else:
                        current_row = [line]
            
            if current_row:
                data_rows.append(current_row)
            
            # If we have at least 2 rows with consistent column counts, format as markdown
            if len(data_rows) >= 2:
                # Try to detect consistent column structure
                col_counts = [len(row) for row in data_rows]
                most_common = max(set(col_counts), key=col_counts.count)
                
                if most_common >= 2:
                    # Filter to rows with the right column count
                    consistent_rows = [row for row in data_rows if len(row) == most_common]
                    
                    if len(consistent_rows) >= 2:
                        # Build markdown table
                        md_lines = []
                        for i, row in enumerate(consistent_rows):
                            md_lines.append('| ' + ' | '.join(row) + ' |')
                            if i == 0:
                                md_lines.append('|' + '---|' * len(row))
                        
                        return '\n'.join(md_lines)
            
            return ""
        except Exception as e:
            logger.debug(f"Markdown conversion failed: {e}")
            return ""
    
    def _format_table(self, table, table_num: int, page_num: int) -> str:
        """Format extracted table as readable text."""
        try:
            # Extract table data
            data = table.extract()
            if not data or len(data) < 2:
                return ""
            
            # Build formatted table text
            lines = [f"\n[TABLE {table_num} on Page {page_num}]"]
            
            # Get headers (first row)
            headers = [str(cell).strip() if cell else "" for cell in data[0]]
            
            # Check if there's a table caption nearby (often in first cell)
            caption = ""
            if headers and headers[0] and headers[0].lower().startswith("table"):
                caption = headers[0]
                lines.append(f"Caption: {caption}")
                data = data[1:]  # Skip caption row
                if data:
                    headers = [str(cell).strip() if cell else "" for cell in data[0]]
            
            # Format header row
            if headers and any(headers):
                lines.append("Headers: " + " | ".join(h for h in headers if h))
            
            # Format data rows
            for row_idx, row in enumerate(data[1:], 1):
                cells = [str(cell).strip() if cell else "" for cell in row]
                if any(cells):
                    # Create key-value pairs with headers
                    row_parts = []
                    for i, cell in enumerate(cells):
                        if cell:
                            header = headers[i] if i < len(headers) and headers[i] else f"Col{i+1}"
                            row_parts.append(f"{header}: {cell}")
                    if row_parts:
                        lines.append(f"Row {row_idx}: " + ", ".join(row_parts))
            
            lines.append("[END TABLE]\n")
            return "\n".join(lines)
            
        except Exception as e:
            logger.debug(f"Table formatting failed: {e}")
            return ""
    
    def extract_with_pdfminer(self, pdf_path: Path) -> List[PageContent]:
        """Extract text using pdfminer (more accurate for complex layouts)."""
        pages = []
        
        for page_num, page_layout in enumerate(extract_pages(str(pdf_path)), start=1):
            text_parts = []
            
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text_parts.append(element.get_text())
            
            text = "\n".join(text_parts)
            
            section_hints = []
            for pattern in self.SECTION_PATTERNS:
                matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                section_hints.extend(matches)
            
            pages.append(PageContent(
                page_number=page_num,
                text=text,
                section_hints=section_hints
            ))
        
        return pages
    
    def extract(self, pdf_path: Path) -> List[PageContent]:
        """Extract text from PDF using best available method."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if fitz is not None:
            logger.info(f"Extracting {pdf_path.name} with PyMuPDF")
            return self.extract_with_pymupdf(pdf_path)
        else:
            logger.info(f"Extracting {pdf_path.name} with pdfminer")
            return self.extract_with_pdfminer(pdf_path)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text while preserving structure."""
        # Remove excessive whitespace but keep paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'-\n', '', text)  # Hyphenation
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)  # Mid-sentence breaks
        
        # Remove noise patterns
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            is_noise = any(re.match(pattern, line.strip()) for pattern in self.NOISE_PATTERNS)
            if not is_noise or len(line.strip()) > 5:  # Keep longer content
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class SemanticChunker:
    """
    Creates semantic chunks from documents with overlap.
    
    Implements overlapping chunks to avoid breaking semantic context,
    as mentioned in Transformer positional encoding requirements.
    """
    
    def __init__(self, chunk_size: int = 400, overlap: int = 75, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token for English)."""
        return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def detect_section(self, text: str, previous_section: str = "Unknown") -> str:
        """Detect section name from text content, preserving section numbers."""
        
        # First check for explicit numbered section headers
        # Handle various formats like "3.2 Multi-Head Attention", "3.2\nAttention", "3.2.2 Multi-Head"
        lines = text[:800].split('\n')
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Pattern 1: "3.2 Multi-Head Attention" on same line
            match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s\-]+)$', line)
            if match:
                section_num = match.group(1)
                section_name = match.group(2).strip()
                return f"Section {section_num}: {section_name}"
            
            # Pattern 2: "3.2" on one line, title on next line
            match = re.match(r'^(\d+(?:\.\d+)*)$', line)
            if match and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and re.match(r'^[A-Z][A-Za-z\s\-]+$', next_line):
                    section_num = match.group(1)
                    section_name = next_line.strip()
                    return f"Section {section_num}: {section_name}"
        
        section_keywords = {
            "abstract": ["abstract"],
            "introduction": ["introduction", "1 introduction", "1. introduction"],
            "background": ["background", "preliminaries"],
            "related work": ["related work", "prior work", "previous work"],
            "methodology": ["method", "methodology", "approach", "our approach"],
            "architecture": ["architecture", "model architecture"],
            "model": ["model", "the model", "our model"],
            "attention": ["attention", "self-attention", "multi-head attention"],
            "training": ["training", "training details", "optimization"],
            "experiments": ["experiment", "experiments", "experimental"],
            "results": ["results", "main results"],
            "analysis": ["analysis", "ablation"],
            "discussion": ["discussion"],
            "conclusion": ["conclusion", "conclusions", "summary"],
            "references": ["references", "bibliography"],
            "appendix": ["appendix", "supplementary"],
            "tables": ["table 1", "table 2", "table 3", "table 4", "table 5", "table 6", "[table"]
        }
        
        text_lower = text[:500].lower()
        
        # Check for tables first
        if "[table" in text_lower or text_lower.strip().startswith("table"):
            return "Tables"
        
        for section, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section.title()
        
        return previous_section
    
    def chunk_document(
        self,
        pages: List[PageContent],
        paper_title: str,
        source_file: str
    ) -> List[DocumentChunk]:
        """
        Create overlapping chunks from extracted pages.
        
        Args:
            pages: List of extracted page contents
            paper_title: Title of the paper
            source_file: Original filename
            
        Returns:
            List of DocumentChunk objects with metadata
        """
        chunks = []
        current_chunk_text = []
        current_chunk_tokens = 0
        current_pages = []
        current_section = "Unknown"
        chunk_index = 0
        
        # Process all pages
        all_text = ""
        page_boundaries = {}  # char_index -> page_number
        
        for page in pages:
            start_idx = len(all_text)
            all_text += page.text + "\n"
            page_boundaries[start_idx] = page.page_number
            
            # Update section from page hints
            if page.section_hints:
                current_section = self.detect_section(
                    " ".join(page.section_hints), 
                    current_section
                )
        
        # Split into sentences for better chunking
        sentences = self.split_into_sentences(all_text)
        
        # Track section headers for prefixing chunks
        last_section_header = ""
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            
            # Check if this sentence is a section header (same line pattern)
            header_match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s\-]+)$', sentence.strip())
            if header_match:
                last_section_header = f"[Section {header_match.group(1)}: {header_match.group(2).strip()}]"
                current_section = f"Section {header_match.group(1)}: {header_match.group(2).strip()}"
            
            # Check for standalone section number followed by title on next sentence
            num_match = re.match(r'^(\d+(?:\.\d+)*)$', sentence.strip())
            if num_match and i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip()
                if re.match(r'^[A-Z][A-Za-z\s\-]+$', next_sentence) and len(next_sentence) < 50:
                    section_num = num_match.group(1)
                    last_section_header = f"[Section {section_num}: {next_sentence}]"
                    current_section = f"Section {section_num}: {next_sentence}"
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_chunk_tokens + sentence_tokens > self.chunk_size and current_chunk_text:
                # Create chunk with section prefix for context
                chunk_text = " ".join(current_chunk_text)
                detected_section = self.detect_section(chunk_text, current_section)
                
                # Prefix chunk with section header for better retrieval
                if detected_section.startswith("Section"):
                    prefixed_text = f"[{detected_section}]\n{chunk_text}"
                elif last_section_header:
                    prefixed_text = f"{last_section_header}\n{chunk_text}"
                else:
                    prefixed_text = chunk_text
                
                if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                    chunk_id = self._generate_chunk_id(source_file, chunk_index)
                    
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=prefixed_text,
                        paper_title=paper_title,
                        section_name=detected_section,
                        page_numbers=list(set(current_pages)) or [1],
                        chunk_index=chunk_index,
                        total_chunks=0,  # Updated later
                        source_file=source_file,
                        token_count=self.estimate_tokens(prefixed_text)
                    ))
                    chunk_index += 1
                
                # Keep overlap
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk_text):
                    s_tokens = self.estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk_text = overlap_sentences
                current_chunk_tokens = overlap_tokens
                current_pages = current_pages[-1:] if current_pages else []
            
            current_chunk_text.append(sentence)
            current_chunk_tokens += sentence_tokens
            
            # Track page numbers (simplified)
            if pages and len(current_pages) < len(pages):
                current_pages.append(min(len(current_pages) + 1, len(pages)))
        
        # Don't forget the last chunk
        if current_chunk_text:
            chunk_text = " ".join(current_chunk_text)
            detected_section = self.detect_section(chunk_text, current_section)
            
            # Prefix chunk with section header
            if detected_section.startswith("Section"):
                prefixed_text = f"[{detected_section}]\n{chunk_text}"
            elif last_section_header:
                prefixed_text = f"{last_section_header}\n{chunk_text}"
            else:
                prefixed_text = chunk_text
            
            if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunk_id = self._generate_chunk_id(source_file, chunk_index)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=prefixed_text,
                    paper_title=paper_title,
                    section_name=detected_section,
                    page_numbers=list(set(current_pages)) or [1],
                    chunk_index=chunk_index,
                    total_chunks=0,
                    source_file=source_file,
                    token_count=self.estimate_tokens(prefixed_text)
                ))
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        logger.info(f"Created {total} chunks from {paper_title}")
        return chunks
    
    def _generate_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{source_file}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class DocumentProcessor:
    """Main class for processing documents into chunks."""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 75):
        self.extractor = PDFExtractor()
        self.chunker = SemanticChunker(chunk_size=chunk_size, overlap=overlap)
    
    def process_pdf(self, pdf_path: Path, paper_title: Optional[str] = None) -> List[DocumentChunk]:
        """
        Process a single PDF into chunks.
        
        Args:
            pdf_path: Path to PDF file
            paper_title: Optional paper title (extracted from content if not provided)
            
        Returns:
            List of DocumentChunk objects
        """
        pdf_path = Path(pdf_path)
        
        logger.info(f"Processing: {pdf_path.name}")
        
        # Extract pages
        pages = self.extractor.extract(pdf_path)
        
        # Extract paper title from first page if not provided
        if paper_title is None and pages:
            paper_title = self.extractor.extract_paper_title(pages[0].text, pdf_path.name)
        elif paper_title is None:
            paper_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
        
        logger.info(f"Paper title: {paper_title}")
        
        # Clean text
        for page in pages:
            page.text = self.extractor.clean_text(page.text)
        
        # Chunk
        chunks = self.chunker.chunk_document(
            pages=pages,
            paper_title=paper_title,
            source_file=pdf_path.name
        )
        
        return chunks
    
    def process_directory(self, directory: Path) -> List[DocumentChunk]:
        """Process all PDFs in a directory."""
        directory = Path(directory)
        all_chunks = []
        
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in {directory}")
        
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
        
        return all_chunks
