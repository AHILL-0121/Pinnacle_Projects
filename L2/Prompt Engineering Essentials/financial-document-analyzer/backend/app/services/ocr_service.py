"""
OCR Service Module
Handles text extraction from images using Tesseract and layout preservation
"""
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRBlock:
    """Represents a block of text from OCR"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    block_type: str = "text"  # text, heading, table, list
    page_number: int = 0
    line_number: int = 0


@dataclass
class OCRResult:
    """Complete OCR result for a page"""
    raw_text: str
    blocks: List[OCRBlock] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    average_confidence: float = 0.0
    page_number: int = 0
    detected_sections: List[str] = field(default_factory=list)


class OCRService:
    """
    OCR Service for extracting text from financial documents.
    Supports Tesseract and PaddleOCR backends.
    """
    
    def __init__(
        self, 
        language: str = "eng",
        tesseract_cmd: Optional[str] = None,
        use_paddle: bool = False
    ):
        self.language = language
        self.use_paddle = use_paddle
        self._tesseract_available = False
        self._paddle_available = False
        
        # Initialize OCR engine
        if use_paddle:
            self._init_paddle()
        else:
            self._init_tesseract(tesseract_cmd)
    
    def _init_tesseract(self, tesseract_cmd: Optional[str] = None):
        """Initialize Tesseract OCR"""
        try:
            import pytesseract
            
            # Auto-detect Tesseract on Windows if not specified
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            else:
                import os
                import platform
                if platform.system() == "Windows":
                    # Common Windows installation paths
                    windows_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
                    ]
                    for path in windows_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            logger.info(f"Found Tesseract at: {path}")
                            break
            
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            self._pytesseract = pytesseract
            logger.info("Tesseract OCR initialized successfully")
            
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self._tesseract_available = False
    
    def _init_paddle(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
            self._paddle_available = True
            logger.info("PaddleOCR initialized successfully")
            
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")
            self._paddle_available = False
    
    def extract_text(
        self, 
        image: Image.Image,
        page_number: int = 0,
        preserve_layout: bool = True
    ) -> OCRResult:
        """
        Extract text from an image with layout preservation.
        
        Args:
            image: PIL Image to process
            page_number: Page number for reference
            preserve_layout: Whether to preserve document layout
            
        Returns:
            OCRResult with extracted text and metadata
        """
        if self.use_paddle and self._paddle_available:
            return self._extract_with_paddle(image, page_number, preserve_layout)
        elif self._tesseract_available:
            return self._extract_with_tesseract(image, page_number, preserve_layout)
        else:
            raise RuntimeError("No OCR engine available. Install pytesseract or paddleocr.")
    
    def _extract_with_tesseract(
        self, 
        image: Image.Image,
        page_number: int,
        preserve_layout: bool
    ) -> OCRResult:
        """Extract text using Tesseract"""
        
        # Get detailed data with bounding boxes
        data = self._pytesseract.image_to_data(
            image, 
            lang=self.language,
            output_type=self._pytesseract.Output.DICT
        )
        
        blocks = []
        current_block_text = []
        current_block_bbox = None
        current_conf_sum = 0
        current_conf_count = 0
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if conf < 0:  # Invalid confidence
                continue
            
            if text:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Build block
                current_block_text.append(text)
                current_conf_sum += conf
                current_conf_count += 1
                
                if current_block_bbox is None:
                    current_block_bbox = [x, y, x + w, y + h]
                else:
                    current_block_bbox[0] = min(current_block_bbox[0], x)
                    current_block_bbox[1] = min(current_block_bbox[1], y)
                    current_block_bbox[2] = max(current_block_bbox[2], x + w)
                    current_block_bbox[3] = max(current_block_bbox[3], y + h)
            
            # Check for block break (new line or paragraph)
            block_num = data['block_num'][i]
            next_block = data['block_num'][i + 1] if i + 1 < len(data['block_num']) else -1
            
            if block_num != next_block and current_block_text:
                block_text = ' '.join(current_block_text)
                avg_conf = current_conf_sum / current_conf_count if current_conf_count > 0 else 0
                
                # Detect block type
                block_type = self._detect_block_type(block_text)
                
                blocks.append(OCRBlock(
                    text=block_text,
                    confidence=avg_conf / 100,  # Normalize to 0-1
                    bbox=tuple(current_block_bbox),
                    block_type=block_type,
                    page_number=page_number
                ))
                
                # Reset
                current_block_text = []
                current_block_bbox = None
                current_conf_sum = 0
                current_conf_count = 0
        
        # Get full text
        if preserve_layout:
            # Use layout-preserving extraction
            raw_text = self._pytesseract.image_to_string(
                image,
                lang=self.language,
                config='--psm 6'  # Assume uniform block of text
            )
        else:
            raw_text = ' '.join(block.text for block in blocks)
        
        # Calculate average confidence
        confidences = [b.confidence for b in blocks if b.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Detect document sections
        sections = self._detect_sections(raw_text)
        
        return OCRResult(
            raw_text=raw_text,
            blocks=blocks,
            average_confidence=avg_confidence,
            page_number=page_number,
            detected_sections=sections
        )
    
    def _extract_with_paddle(
        self, 
        image: Image.Image,
        page_number: int,
        preserve_layout: bool
    ) -> OCRResult:
        """Extract text using PaddleOCR"""
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Run OCR
        result = self._paddle_ocr.ocr(img_array, cls=True)
        
        blocks = []
        all_text = []
        
        for line in result[0] if result[0] else []:
            bbox_points = line[0]
            text, conf = line[1]
            
            # Convert polygon to rectangle bbox
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            bbox = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords))
            )
            
            block_type = self._detect_block_type(text)
            
            blocks.append(OCRBlock(
                text=text,
                confidence=conf,
                bbox=bbox,
                block_type=block_type,
                page_number=page_number
            ))
            
            all_text.append(text)
        
        # Sort blocks by position (top to bottom, left to right)
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # Reconstruct text preserving layout
        raw_text = '\n'.join(all_text)
        
        # Calculate average confidence
        confidences = [b.confidence for b in blocks]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Detect sections
        sections = self._detect_sections(raw_text)
        
        return OCRResult(
            raw_text=raw_text,
            blocks=blocks,
            average_confidence=avg_confidence,
            page_number=page_number,
            detected_sections=sections
        )
    
    def _detect_block_type(self, text: str) -> str:
        """Detect the type of text block"""
        text_lower = text.lower().strip()
        
        # Check for headings
        heading_patterns = [
            r'^(balance sheet|income statement|cash flow|statement of)',
            r'^(notes to|auditor|management discussion)',
            r'^(financial highlights|key highlights|quarterly)',
            r'^(revenue|expenses|assets|liabilities|equity)',
            r'^(q[1-4]|fy\s*\d{2,4}|fiscal year)',
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text_lower):
                return "heading"
        
        # Check for list items
        if re.match(r'^[\-•●○]\s', text) or re.match(r'^\d+[\.\)]\s', text):
            return "list"
        
        # Check for table-like content
        if text.count('\t') >= 2 or re.search(r'\d+\s{2,}\d+', text):
            return "table"
        
        return "text"
    
    def _detect_sections(self, text: str) -> List[str]:
        """Detect financial document sections in text"""
        sections = []
        text_lower = text.lower()
        
        section_keywords = {
            "Balance Sheet": ["balance sheet", "statement of financial position", "assets and liabilities"],
            "Income Statement": ["income statement", "profit and loss", "p&l", "statement of operations"],
            "Cash Flow": ["cash flow", "statement of cash flows", "cash from operations"],
            "Notes": ["notes to", "accounting policies", "significant accounting"],
            "Auditor Report": ["auditor", "audit opinion", "independent auditor"],
            "Management Discussion": ["management discussion", "md&a", "management's discussion"],
            "Financial Highlights": ["financial highlights", "key highlights", "performance summary"],
            "Segment Performance": ["segment", "business segment", "operating segment"],
        }
        
        for section_name, keywords in section_keywords.items():
            if any(kw in text_lower for kw in keywords):
                sections.append(section_name)
        
        return sections
    
    def extract_from_multiple_pages(
        self, 
        images: List[Image.Image],
        preserve_layout: bool = True
    ) -> List[OCRResult]:
        """
        Extract text from multiple pages.
        
        Args:
            images: List of PIL Images
            preserve_layout: Whether to preserve layout
            
        Returns:
            List of OCRResult objects
        """
        results = []
        
        for i, image in enumerate(images):
            logger.debug(f"Processing page {i + 1}/{len(images)}")
            result = self.extract_text(image, page_number=i, preserve_layout=preserve_layout)
            results.append(result)
        
        return results
    
    def get_full_document_text(self, ocr_results: List[OCRResult]) -> str:
        """Combine OCR results into single document text"""
        pages_text = []
        
        for result in ocr_results:
            page_text = f"\n--- Page {result.page_number + 1} ---\n"
            page_text += result.raw_text
            pages_text.append(page_text)
        
        return '\n'.join(pages_text)
