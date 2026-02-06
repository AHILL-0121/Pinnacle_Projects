"""
Validation Utilities
Input validation for files, URLs, and data
"""
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.json'}

# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed URL schemes
ALLOWED_URL_SCHEMES = {'http', 'https'}


def validate_file(
    file_path: str | Path = None,
    file_bytes: bytes = None,
    file_name: str = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file.
    
    Args:
        file_path: Path to the file
        file_bytes: File content as bytes
        file_name: Original filename (for extension check)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if we have something to validate
    if file_path is None and file_bytes is None:
        return False, "No file provided"
    
    # Validate file path if provided
    if file_path:
        path = Path(file_path)
        
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        if not path.is_file():
            return False, f"Not a file: {file_path}"
        
        # Check extension
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        
        # Check file size
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            return False, f"File too large: {size / (1024*1024):.1f} MB. Maximum: {MAX_FILE_SIZE / (1024*1024):.0f} MB"
        
        if size == 0:
            return False, "File is empty"
    
    # Validate file bytes if provided
    if file_bytes:
        if len(file_bytes) > MAX_FILE_SIZE:
            return False, f"File too large: {len(file_bytes) / (1024*1024):.1f} MB"
        
        if len(file_bytes) == 0:
            return False, "File is empty"
        
        # Check file name extension if provided
        if file_name:
            ext = Path(file_name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported file type: {ext}"
        
        # Try to detect file type from magic bytes
        file_type = detect_file_type(file_bytes)
        if file_type is None:
            return False, "Could not determine file type"
        
        if file_type not in ['pdf', 'png', 'jpg', 'tiff', 'bmp', 'webp', 'json']:
            return False, f"Unsupported file type detected: {file_type}"
    
    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate document URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is required"
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"
    
    # Check scheme
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        return False, f"Unsupported URL scheme: {parsed.scheme}. Use http or https."
    
    # Check host
    if not parsed.netloc:
        return False, "URL must include a host"
    
    # Check for potentially valid file extension
    path = parsed.path.lower()
    has_valid_extension = any(path.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    
    # Allow URLs without extension (might be dynamic)
    # Just warn in logs
    if not has_valid_extension:
        logger.warning(f"URL does not have a recognized file extension: {url}")
    
    return True, None


def detect_file_type(data: bytes) -> Optional[str]:
    """
    Detect file type from magic bytes.
    
    Args:
        data: File content as bytes
        
    Returns:
        File type string or None
    """
    if len(data) < 8:
        return None
    
    # PDF: %PDF
    if data[:4] == b'%PDF':
        return 'pdf'
    
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    
    # JPEG: FF D8 FF
    if data[:3] == b'\xff\xd8\xff':
        return 'jpg'
    
    # TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
    if data[:4] in [b'II*\x00', b'MM\x00*']:
        return 'tiff'
    
    # BMP: 42 4D
    if data[:2] == b'BM':
        return 'bmp'
    
    # WebP: 52 49 46 46 ... 57 45 42 50
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    
    # JSON: starts with { or [ (with optional whitespace)
    try:
        text_start = data[:100].decode('utf-8').lstrip()
        if text_start.startswith('{') or text_start.startswith('['):
            return 'json'
    except (UnicodeDecodeError, AttributeError):
        pass
    
    return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove potentially dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    
    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:max_length - len(ext)] + ext
    
    return filename


def validate_role(role: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user role selection.
    
    Args:
        role: Role string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_roles = {'investor', 'analyst', 'auditor', 'executive'}
    
    if not role:
        return False, "Role is required"
    
    role = role.lower().strip()
    
    if role not in valid_roles:
        return False, f"Invalid role: {role}. Valid roles: {', '.join(valid_roles)}"
    
    return True, None


def validate_financial_period(period: str) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Validate and parse financial period string.
    
    Args:
        period: Period string (e.g., "Q3 2024", "FY2023", "2024")
        
    Returns:
        Tuple of (is_valid, error_message, parsed_data)
    """
    if not period:
        return True, None, None  # Period is optional
    
    period = period.strip().upper()
    
    # Quarterly pattern: Q1 2024, Q2-2024, etc.
    quarter_match = re.match(r'Q([1-4])\s*[-/]?\s*(\d{4})', period)
    if quarter_match:
        return True, None, {
            'type': 'quarterly',
            'quarter': int(quarter_match.group(1)),
            'year': int(quarter_match.group(2))
        }
    
    # Fiscal year pattern: FY2024, FY24, FY 2024
    fy_match = re.match(r'FY\s*(\d{2,4})', period)
    if fy_match:
        year = int(fy_match.group(1))
        if year < 100:
            year += 2000  # Convert YY to 20YY
        return True, None, {
            'type': 'fiscal_year',
            'year': year
        }
    
    # Plain year: 2024
    year_match = re.match(r'^(\d{4})$', period)
    if year_match:
        return True, None, {
            'type': 'year',
            'year': int(year_match.group(1))
        }
    
    return False, f"Invalid period format: {period}. Use formats like 'Q3 2024', 'FY2023', or '2024'", None
